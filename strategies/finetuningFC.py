from opt import OPT
import torch
from utils import log_metrics
import torch.nn.functional as F
import torch.nn as nn
import utils
from torch import optim


class FinetuningFC():

    def __init__(self, model) -> None:
        self.model = model
        for p in model.parameters():
            p.requires_grad=False
        
        for p in model.fc.parameters():
            p.requires_grad=True
        
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=OPT.CONTINUAL_LR, weight_decay=OPT.CONTINUAL_WD)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "FinetuningFC"

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

            # Print measures
            log_metrics(seen, cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

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
                            loss_val = self.loss_fn(y_hat, y_onehot)

                            # Compute measures
                            cumul_loss_val += loss_val.item()
                            cumul_acc_val += (y_hat.argmax(dim=1) == y).sum().item()
                            seen += len(y)

                        # Print measures
                        log_metrics(seen, cumul_loss_val, cumul_acc_val, epoch, 'val', writer, tag)

            # Save the model
            if ((epoch % OPT.CHK_EVERY) == 0) and OPT.CHK:
                torch.save(self.model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}_{OPT.MODEL}.pt')
