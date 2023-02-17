from opt import OPT
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from torch import optim
from strategies.base import Base
import os

class MemoryBuffer():
    def __init__(self, size=500) -> None:
        self.size = size
        self.buffer = None
        self.labels = None
        self.seen = 0

    def __len__(self):
        if self.buffer is None:
            return 0

        return len(self.labels)
    
    def push(self, samples, labels):
        samples = samples.cpu()
        labels = labels.cpu()

        if self.buffer is None:
            self.buffer = samples
            self.labels = labels
        elif len(self.buffer) < self.size:
            self.buffer = torch.cat((self.buffer, samples), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)
        else:
            self.buffer = torch.cat((self.buffer, samples), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)
            indices = torch.randperm(self.labels.shape[0])
            self.buffer = self.buffer[indices][:self.size]
            self.labels = self.labels[indices][:self.size]

    def get_next_examples(self, len_loader):
        increment = int((len(self.buffer) / len_loader) + 0.5)
        
        replay_x = self.buffer[self.seen:self.seen+increment]
        replay_y = self.labels[self.seen:self.seen+increment]

        if self.seen+increment >= len(self.buffer):
            self.seen = 0
        else:
            self.seen += increment
            
        return replay_x, replay_y


class Replay(Base):

    def __init__(self, model, buffer_size=500) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=OPT.LR_CONT, weight_decay=OPT.WD_CONT)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "replay"
        self.buffer = MemoryBuffer(size=buffer_size)


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
                if epoch == OPT.EPOCHS_CONT - 1:
                    self.buffer.push(x, y)
                # Move to GPU
                x = x.to(OPT.DEVICE)
                y = y.to(OPT.DEVICE)
                
                # Augment the batch with memory data
                if tag != '0':
                    replay_x, replay_y = self.buffer.get_next_examples(len(train_loader))
                    x = torch.cat((x, replay_x.to(OPT.DEVICE)), dim=0)
                    y = torch.cat((y, replay_y.to(OPT.DEVICE)), dim=0)

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


    def get_csv_name(self):
        return os.path.join(OPT.CSV_FOLDER, f"{OPT.DATASET}_{OPT.NUM_TASKS}tasks_{self.name.replace('_','')}{self.buffer.size}_{OPT.MODEL.replace('_','')}.csv")