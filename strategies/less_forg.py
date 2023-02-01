from opt import OPT
import torch
import copy
import torch.nn.functional as F
from torch import optim
import utils
import torch.nn as nn

class LessForg():
    """ Requires resnet32 because of intermediate features
    https://arxiv.org/pdf/1607.00122.pdf
    """

    def __init__(self, model, original_impl, alpha=1., beta=1.) -> None:
        self.model = model
        self.alpha = 1
        self.beta = 0.001
        self.original_implementation = original_impl
        self.optimizer = self._set_optim(original_impl)
        self.name = "LessForg"
        self.loss_fn = nn.CrossEntropyLoss()


    def train(self,train_loader, val_loader, writer, tag, scheduler=False):
        self.model.to(OPT.DEVICE)

        # Previous frozen model
        old_model = copy.deepcopy(self.model)
        for p in old_model.parameters():
            p.requires_grad = False

        # # Freeze current model fc layer (as in original paper implementation)
        # for p in model.fc.parameters():
        #     p.requires_grad = False

        if scheduler:
            sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 0.01, epochs=OPT.CONTINUAL_EPOCHS, steps_per_epoch=len(train_loader))

        for epoch in range(0, OPT.CONTINUAL_EPOCHS):
            print(f'    EPOCH {epoch} ')

            ##################
            #### Training ####
            cumul_loss_train = 0
            cumul_loss_e_train = 0
            cumul_acc_train = 0
            seen = 0
            self.model.train()
            for x, y in train_loader:
                # Move to GPU
                x = x.to(OPT.DEVICE)
                y = y.to(OPT.DEVICE)

                # Forward data to model and compute loss
                out = utils.check_output(self.model(x))
                y_hat, features = out['y_hat'], out['fts']
                y_hat = y_hat.to(torch.float32)
                features = features.to(torch.float32)

                _, old_features = old_model(x)
                old_features = old_features.to(torch.float32)

                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)

                # Losses
                l_c = self.loss_fn(y_hat, y_onehot) * self.alpha

                l_e = 0.5 * (torch.linalg.norm(features - old_features) ** 2)
                l_e = l_e * self.beta

                loss_train = l_c + l_e

                # Backward
                loss_train.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Record & update learning rate
                if scheduler:
                    sched.step()

                # Compute measures
                cumul_loss_train += l_c.item()
                cumul_loss_e_train += l_e.item()
                cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            # Print measures
            self.log_metrics(seen, cumul_loss_train, cumul_loss_e_train, cumul_acc_train, epoch, 'train', writer, tag)

            ####################
            #### Validation ####
            if (epoch == 0) or ((epoch % OPT.LOG_EVERY) == 0):
                    with torch.no_grad():
                        cumul_loss_val = 0
                        cumul_loss_e_val = 0
                        cumul_acc_val = 0
                        seen = 0
                        self.model.eval()
                        for x, y in val_loader:

                            # Move to GPU
                            x = x.to(OPT.DEVICE)
                            y = y.to(OPT.DEVICE)

                            # Forward to model
                            out = utils.check_output(self.model(x))
                            y_hat, features = out['y_hat'], out['fts']
                            y_hat = y_hat.to(torch.float32)
                            features = features.to(torch.float32)

                            _, old_features = old_model(x)
                            old_features = old_features.to(torch.float32)

                            y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)


                            l_c = self.loss_fn(y_hat, y_onehot) * self.alpha
                            l_e = 0.5 * (torch.linalg.norm(features - old_features) ** 2)
                            l_e = l_e * self.beta
                            loss_train = l_c + l_e

                            # Compute measures
                            cumul_loss_e_val += l_e.item()
                            cumul_loss_val += l_c.item()
                            cumul_acc_val += (y_hat.argmax(dim=1) == y).sum().item()
                            seen += len(y)

                        # Print measures
                        self.log_metrics(seen, cumul_loss_val, cumul_loss_e_val, cumul_acc_val, epoch, 'val', writer, tag)

            # Save the model
            if ((epoch % OPT.CHK_EVERY) == 0) and OPT.CHK:
                torch.save(self.model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}.pt')


    def log_metrics(self, seen, loss, loss_e, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        loss /= seen
        acc /= seen
        loss_e /= seen
        print(f'        {session:<6} - l_c:{loss:.5f} l_e:{loss_e:.5f}  a:{acc:.5f}')

        if OPT.LOG:
            writer.add_scalar(f'{tag}/loss_c/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/loss_e/{session}', loss_e, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)


    def _set_optim(self, original_impl):

        if original_impl == True:
            optimizer = optim.Adam(self.model.parameters(), lr=OPT.CONTINUAL_LR, weight_decay=OPT.CONTINUAL_WD)
        else:
            optimizer = optim.Adam(
            [
                {"params": self.model.conv1.parameters()},
                {"params": self.model.layer1.parameters()},
                {"params": self.model.layer2.parameters()},
                {"params": self.model.layer3.parameters()},
                {"params": self.model.fc.parameters(), 
                        "lr": OPT.CONTINUAL_LR*0.1, 
                        "weight_decay":OPT.CONTINUAL_WD*0.1}
            ],
            lr=OPT.CONTINUAL_LR, 
            weight_decay=OPT.CONTINUAL_WD
            )
        return optimizer

