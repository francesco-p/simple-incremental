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
    def __init__(self, model, emb_size=128) -> None:
        super(ResNet18_FR, self).__init__()
        self.emb_size = emb_size

        self.backbone = nn.Sequential(*(list(model.children())[:-1]))
        self.backbone_fc = model.fc

        self.refiner = nn.Sequential(
            nn.Linear(512, emb_size),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.LayerNorm(emb_size))

        self.refiner_fc = nn.Linear(emb_size, OPT.NUM_CLASSES)


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
                out_dict = utils.check_output(self.model(x))
                y_hat, y_fr = out_dict['bkb'], out_dict['fr']

                y_hat = y_hat.to(torch.float32)
                y_fr = y_fr.to(torch.float32)

                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)

                # Losses
                l_bkb = self.loss_fn(y_hat, y_onehot)
                l_fr = self.loss_fn(y_fr, y_onehot)

                loss_train = l_bkb + l_fr

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

                            # Forward data to model and compute loss
                            out_dict = utils.check_output(self.model(x))
                            y_hat, y_fr = out_dict['bkb'], out_dict['fr']

                            y_hat = y_hat.to(torch.float32)
                            y_fr = y_fr.to(torch.float32)

                            y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)

                            # Losses
                            l_bkb = self.loss_fn(y_hat, y_onehot)
                            l_fr = self.loss_fn(y_fr, y_onehot)

                            loss_val = l_bkb + l_fr

                            # Compute measures
                            cumul_loss_val += loss_val.item()
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
        print(f'        {session:<6} - l:{loss:.5f} a:{acc:.5f}')

        if OPT.LOG:
            writer.add_scalar(f'{tag}/loss_c/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)

