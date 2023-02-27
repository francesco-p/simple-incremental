
from opt import OPT
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from torch import optim
from strategies.base import Base

class SurgicalFT(Base):

    """ https://arxiv.org/pdf/2210.11466.pdf 
    Surgical works with models that have a layer X 
    such as: model.layerX.parameters()
    resnets are fine"""

    def __init__(self, model, layer) -> None:
        super().__init__()
        self.model = model
        self.layer = layer
        self.optimizer = self._set_optim(layer)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "SurgicalFT"


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):
        self.model.to(OPT.DEVICE)

        if scheduler:
            sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 0.01, epochs=OPT.EPOCHS_CONT, steps_per_epoch=len(train_loader))

        for epoch in range(0, OPT.EPOCHS_CONT):
            print(f'    EPOCH {epoch} ')

            ##################
            #### Training ####
            cumul_loss_train = 0
            cumul_acc_train = 0
            seen = 0
            self.model.train()
            for x, y in train_loader:
                # Move to GPU
                x = x.to(OPT.DEVICE)
                y = y.to(OPT.DEVICE)

                # Forward data to model and compute loss
                y_hat = utils.check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
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
            if (epoch == 0) or ((epoch % OPT.EVAL_EVERY_CONT) == 0):
                eval_loss, eval_acc = self.eval(val_loader, writer, tag)
                #torch.save(self.model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}_{OPT.MODEL}.pt')


    def eval(self, val_loader, writer, tag):
        """ Evaluate the model on the evaluation set"""
        with torch.no_grad():
            cumul_loss_eval = 0
            cumul_acc_eval = 0
            seen = 0
            self.model.eval()
            for x, y in val_loader:

                # Move to GPU
                x = x.to(OPT.DEVICE)
                y = y.to(OPT.DEVICE)

                # Forward to model
                y_hat = utils.check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
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
        print(f'        {tag}_{session:<6} - l:{loss:.5f}  a:{acc:.5f}')

        if OPT.TENSORBOARD:
            writer.add_scalar(f'{tag}/loss/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)


    def _set_optim(self, layer):

        if layer == 0:
            tr_params = [{"params": self.model.conv1.parameters()}]
        elif layer == 1:
            tr_params = [{"params": self.model.layer1.parameters()}]
        elif layer == 2:
            tr_params = [{"params": self.model.layer2.parameters()}]
        elif layer == 3:
            tr_params = [{"params": self.model.layer3.parameters()}]
        elif layer == 4:
            tr_params = [{"params": self.model.fc.parameters()}]

        optimizer = optim.Adam(tr_params,lr=OPT.LR_CONT, weight_decay=OPT.WD_CONT)

        return optimizer
    

    def get_csv_name(self):
        return os.path.join(OPT.CSV_FOLDER, f"{OPT.DATASET}_{OPT.NUM_TASKS}tasks_{self.name.replace('_','')}{self.layer}_{OPT.MODEL.replace('_','')}.csv")