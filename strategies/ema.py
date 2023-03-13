from opt import OPT
import torch
import copy
import torch.nn.functional as F
from torch import optim
from timm.utils.model_ema import ModelEmaV2
import torch.nn as nn
import utils
from strategies.base import Base

class Ema(Base):
    def __init__(self, model, decay=0.1) -> None:
        self.model = model
        self.ema_model = ModelEmaV2(model, device=OPT.device, decay=decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.lr, weight_decay=OPT.wd)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "Ema"


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):
        self.model.to(OPT.device)

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

                # Losses
                l_c = self.loss_fn(y_hat, y_onehot)
                loss_train = l_c

                # Backward
                loss_train.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Record & update learning rate
                if scheduler:
                    sched.step()

                # Compute measures
                cumul_loss_train += l_c.item()
                cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            # Print measures
            cumul_acc_train /= seen
            cumul_loss_train /= len(train_loader)
            self.log_metrics(cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

            ####################
            #### Validation ####
            if (epoch == 0) or ((epoch % OPT.eval_every) == 0):
                eval_loss, eval_acc = self.eval(val_loader, writer, tag)


        self.ema_model.update(self.model)
        self.model = copy.deepcopy(self.ema_model.module)
    
    
    def eval(self, val_loader, writer, tag):
        """ Evaluate the model on the evaluation set"""
        with torch.no_grad():
            cumul_loss_eval = 0
            cumul_acc_eval = 0
            seen = 0
            self.model.eval()
            for x, y in val_loader:

                # Move to GPU
                x = x.to(OPT.device)
                y = y.to(OPT.device)

                # Forward to model
                y_hat = utils.check_output(self.model(x))['y_hat']
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



    def log_metrics(self, loss, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        print(f'        {session:<6} - l_c:{loss:.5f} a:{acc:.5f}')

        if OPT.tboard:
            writer.add_scalar(f'{tag}/loss_c/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)

