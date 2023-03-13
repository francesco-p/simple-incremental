from opt import OPT
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from torch import optim
from strategies.base import Base
import copy
import os

class Soup(Base):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.lr, weight_decay=OPT.wd)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "Soup"


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):
        self.model.to(OPT.device)

         # Previous frozen model
        old_model = copy.deepcopy(self.model)
        for p in old_model.parameters():
            p.requires_grad = False
        for p in self.model.parameters():
            p.requires_grad = True

        if scheduler:
            sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 0.01, epochs=OPT.epochs, steps_per_epoch=len(train_loader))

        for epoch in range(0, OPT.epochs):
            print(f'    EPOCH {epoch} ')

            ##################
            #### Training ####
            cumul_loss_train = 0
            cumul_acc_train = 0
            seen = 0
            self.model.train()
            for x, y in train_loader:
                # Move to GPU
                x = x.to(OPT.device)
                y = y.to(OPT.device)

                # Forward data to model and compute loss
                y_hat = utils.check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                y_onehot = F.one_hot(y, num_classes=OPT.num_classes).to(torch.float32)
                loss_train = self.loss_fn(y_hat, y_onehot)

                # Backward
                loss_train.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Record & update learning rate
                if scheduler:
                    sched.step()

                # Compute measures
                cumul_loss_train += loss_train.item()
                cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            cumul_acc_train /= seen
            cumul_loss_train /= seen
            # Print measures
            self.log_metrics(cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

            ####################
            #### Validation ####
            if (epoch == 0) or ((epoch % OPT.eval_every) == 0):
                eval_loss, eval_acc = self.eval(val_loader, writer, tag, self.model)
                #torch.save(self.model.state_dict(), OPT.chk_folder+f'/{tag}_{epoch:04}_{OPT.model}.pt')
            
        new_model = copy.deepcopy(self.model)
        for i, (p1, p2) in enumerate(zip(new_model.parameters(), old_model.parameters())):
            p1 =  (p1 + p2)/2.

        eval_loss, eval_acc = self.eval(val_loader, writer, tag, self.model)
        new_eval_loss, new_eval_acc = self.eval(val_loader, writer, tag, new_model)
        old_eval_loss, old_eval_acc = self.eval(val_loader, writer, tag, old_model)
        print(f"before new data: loss {old_eval_loss}, acc {old_eval_acc}")
        print(f"before soup: loss {eval_loss}, acc {eval_acc}")
        print(f"after soup: loss {new_eval_loss}, acc {new_eval_acc}")

        if new_eval_acc > eval_acc:
            if new_eval_acc > old_eval_acc:
                print(f"performance improved through soup. Updating.")
                self.model = copy.deepcopy(new_model)
        if old_eval_acc > eval_acc:
            if old_eval_acc > new_eval_acc:
                print(f"performance not improved through new data. Updating.")
                self.model = copy.deepcopy(old_model)




    def eval(self, val_loader, writer, tag, model):
        """ Evaluate the model on the evaluation set"""
        with torch.no_grad():
            cumul_loss_eval = 0
            cumul_acc_eval = 0
            seen = 0
            model.eval()
            for x, y in val_loader:

                # Move to GPU
                x = x.to(OPT.device)
                y = y.to(OPT.device)

                # Forward to model
                y_hat = utils.check_output(model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                y_onehot = F.one_hot(y, num_classes=OPT.num_classes).to(torch.float32)
                loss_test = self.loss_fn(y_hat, y_onehot)

                # Compute measures
                cumul_loss_eval += loss_test.item()
                cumul_acc_eval += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            cumul_acc_eval /= seen
            cumul_loss_eval /= seen
            # Print measures
            self.log_metrics(cumul_loss_eval, cumul_acc_eval, 0, 'eval', writer, tag)
        
        return cumul_loss_eval, cumul_acc_eval


    def get_name(self):
        fname = os.path.join(OPT.csv_folder, f"{OPT.dataset}_{OPT.num_tasks}tasks_{OPT.strategy}_{OPT.model.replace('_','')}_epochs{OPT.epochs}.csv")
        return fname
    
    def log_metrics(self, loss, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        print(f'        {tag}_{session:<6} - l:{loss:.5f}  a:{acc:.5f}')

        if OPT.tboard:
            writer.add_scalar(f'{tag}/loss/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)
