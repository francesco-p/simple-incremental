from importlib.util import LazyLoader
from pandas import set_option
from opt import OPT
import torch
import copy
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import utils


class ResNet18_FR(nn.Module):
    def __init__(self, model) -> None:
        super(ResNet18_FR).__init__()
        
        self.backbone = nn.Sequential(*(list(model.children())[:-1]))
        self.backbone_fc = model.fc
        
        self.refiner = nn.Sequential([
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.LayerNorm(64)])

        self.refiner_fc = nn.Linear(512, OPT.NUM_CLASSES)
        

    def forward(self, x):

        _x1 = self.backbone(x)
        _x2 = _x1.clone()
        _x1 = _x1.detach()
        backbone_pred = self.backbone_fc(_x1)
        refiner_pred = self.refiner_fc(self.refiner(_x2))

        return backbone_pred, refiner_pred


class OJKD():
    """ https://arxiv.org/pdf/2210.05657.pdf
    assumes resnet18
    """

    def __init__(self, model) -> None:
        self.model = ResNet18_FR(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.CONTINUAL_LR, weight_decay=OPT.CONTINUAL_WD)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "OJKD"


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):
        self.model.to(OPT.DEVICE)

        if scheduler:
            sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 0.01, epochs=OPT.CONTINUAL_EPOCHS, steps_per_epoch=len(train_loader))

        for epoch in range(0, OPT.CONTINUAL_EPOCHS):
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
            self.log_metrics(seen, cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

            ####################
            #### Validation ####
            if (epoch == 0) or ((epoch % OPT.LOG_EVERY) == 0):
                    with torch.no_grad():
                        cumul_loss_val = 0
                        cumul_acc_val = 0
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

                            l_c = self.loss_fn(y_hat, y_onehot)
                            loss_train = l_c

                            # Compute measures
                            cumul_loss_val += l_c.item()
                            cumul_acc_val += (y_hat.argmax(dim=1) == y).sum().item()
                            seen += len(y)

                        # Print measures
                        self.log_metrics(seen, cumul_loss_val, cumul_acc_val, epoch, 'val', writer, tag)

            # Save the model
            if ((epoch % OPT.CHK_EVERY) == 0) and OPT.CHK:
                torch.save(self.model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}.pt')


    def log_metrics(self, seen, loss, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        loss /= seen
        acc /= seen
        print(f'        {session:<6} - l_c:{loss:.5f} a:{acc:.5f}')

        if OPT.LOG:
            writer.add_scalar(f'{tag}/loss_c/{session}', loss, epoch)
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

        optimizer = optim.Adam(tr_params,lr=OPT.CONTINUAL_LR, weight_decay=OPT.CONTINUAL_WD)

        return optimizer
