from opt import OPT
import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from utils import check_output
from strategies.base import Base

class ResNet18_FR(nn.Module):
    """ Feature Refiner for ResNet18 """
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
        """ 
        x: input image
        returns: backbone prediction, refiner prediction
        """

        _x1 = self.backbone(x)
        _x2 = _x1.clone()
        _x1 = _x1.detach()
        backbone_pred = self.backbone_fc(_x1)
        refiner_pred = self.refiner_fc(self.refiner(_x2))

        return backbone_pred, refiner_pred


class OJKD(Base):
    """ https://arxiv.org/pdf/2210.05657.pdf

    This method only with ResNet18 for now
    """

    def __init__(self, model) -> None:
        self.model = ResNet18_FR(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.LR_CONT, weight_decay=OPT.WD_CONT)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "OJKD"


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
                out_dict = check_output(self.model(x))
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
            cumul_loss_train /= seen
            cumul_acc_train /= seen
            self.log_metrics(cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

            ####################
            #### Validation ####
            if (epoch == 0) or ((epoch % OPT.EVAL_EVERY_CONT) == 0):
                eval_loss, eval_acc = self.eval(val_loader, writer, tag)
                #torch.save(self.model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}_{OPT.MODEL}.pt')


    def eval(self, test_loader, writer, tag):
        self.model.to(OPT.DEVICE)
        self.model.eval()
        with torch.no_grad():
            cumul_loss_test = 0
            cumul_acc_test = 0
            seen = 0
            for x, y in test_loader:
                # Move to GPU
                x = x.to(OPT.DEVICE)
                y = y.to(OPT.DEVICE)

                # Forward data to model and compute loss
                out_dict = check_output(self.model(x))
                y_hat, y_fr = out_dict['bkb'], out_dict['fr']

                y_hat = y_hat.to(torch.float32)
                y_fr = y_fr.to(torch.float32)

                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)

                # Losses
                l_bkb = self.loss_fn(y_hat, y_onehot)
                l_fr = self.loss_fn(y_fr, y_onehot)

                loss_test = l_bkb + l_fr

                # Compute measures
                cumul_loss_test += loss_test.item()
                cumul_acc_test += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            cumul_acc_test /= seen
            cumul_loss_test /= seen
            # Print measures (we do not print the feature refiner metrics)
            self.log_metrics(cumul_loss_test, cumul_acc_test, 0, 'test', writer, tag)

        return cumul_loss_test, cumul_acc_test


    def log_metrics(self, loss, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        print(f'        {session:<6} - loss:{loss:.5f} acc:{acc:.5f}')

        if OPT.TENSORBOARD:
            writer.add_scalar(f'{tag}/loss/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)

