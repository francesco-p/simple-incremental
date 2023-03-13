from opt import OPT
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from torch import optim
from strategies.base import Base
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import os


class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # required int cast to concatenate the datasets
        # pytorch datasets return int as labels
        return self.x[idx].to(OPT.device), self.y[idx].to(OPT.device)

class TensorDataset_one_hot(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # required int cast to concatenate the datasets
        # pytorch datasets return int as labels
        return self.x[idx][0].to(OPT.device), F.one_hot(torch.tensor(self.x[idx][1]), num_classes = OPT.num_classes).to(torch.float32).to(OPT.device)


class CDD(Base):

    def __init__(self, model, buffer_size) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.lr, weight_decay=OPT.wd)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "CDD"
        self.buffer_images = torch.tensor([])
        self.buffer_labels = torch.tensor([])
        self.buffer_size = buffer_size


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):
        self.model.to(OPT.device)

        if scheduler:
            sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 0.01, epochs=OPT.epochs, steps_per_epoch=len(train_loader))


        if int(tag)>0:

            syn_images_and_labels = torch.load(f"CDD/results/{OPT.dataset}_{OPT.seed}/ConvNet_1_5_4//task_{int(tag) - 1}/res.pth")
            
            if int(tag)==1:
                self.buffer_images = syn_images_and_labels["data"][0][0][:self.buffer_size]
                self.buffer_labels = F.one_hot(syn_images_and_labels["data"][0][1], num_classes = OPT.num_classes)[:self.buffer_size]
            
            if int(tag)>1:
                indices = torch.randperm(self.buffer_labels.shape[0])[:self.buffer_size]
                self.buffer_images = self.buffer_images[indices].to(OPT.device)
                self.buffer_labels = self.buffer_labels[indices].to(OPT.device)

                indices = torch.randperm(syn_images_and_labels["data"][0][1].shape[0])[:self.buffer_size]
                
                #normalize keeping count of how many task have been avaraged so far
                self.buffer_images *= torch.tensor(float(tag) - 1.).long()
                self.buffer_images +=  syn_images_and_labels["data"][0][0].to(OPT.device)[indices]
                self.buffer_images /= torch.tensor(float(tag)).long()

                self.buffer_labels *= torch.tensor(float(tag) - 1.).long()
                self.buffer_labels = self.buffer_labels + F.one_hot(syn_images_and_labels["data"][0][1][indices], num_classes = OPT.num_classes).to(torch.float32).to(OPT.device)
                self.buffer_labels /= torch.tensor(float(tag)).long()

            replay_buffer_dset = TensorDataset(self.buffer_images, self.buffer_labels)

            train_dataset = TensorDataset_one_hot(train_loader.dataset)
            
            # Create a new DataLoader with the concatenated data
            train_loader = DataLoader(
                ConcatDataset((replay_buffer_dset, train_dataset)),
                batch_size=OPT.batch_size,
                shuffle=True # important to shuffle the data
            )
        print("len dataloader: ", len(train_loader))
        print("shape buffer: ", self.buffer_images.shape)

        for epoch in range(0, OPT.epochs):
            print(f'    EPOCH {epoch} ')

            ##################
            #### Training ####
            cumul_loss_train = 0
            cumul_acc_train = 0
            seen = 0
            self.model.train()
            for i, (x, y) in enumerate(train_loader):
                # Move to GPU
                
                x = x.to(OPT.device)
                y = y.to(OPT.device)

                # Forward data to model and compute loss
                y_hat = utils.check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                if int(tag)==0:
                    y = F.one_hot(y, num_classes=OPT.num_classes).to(torch.float32) 
        
                loss_train = self.loss_fn(y_hat, y)

                # Backward
                loss_train.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Record & update learning rate
                if scheduler:
                    sched.step()
               
                # Compute measures
                cumul_loss_train += loss_train.item()
                cumul_acc_train += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()
                seen += len(y)

            cumul_acc_train /= seen
            cumul_loss_train /= seen
            # Print measures
            self.log_metrics(cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

            ####################
            #### Validation ####
            if (epoch == 0) or ((epoch % OPT.eval_every) == 0):
                eval_loss, eval_acc = self.eval(val_loader, writer, tag)
                #torch.save(self.model.state_dict(), OPT.chk_folder+f'/{tag}_{epoch:04}_{OPT.model}.pt')


    def eval(self, val_loader, writer, tag ):
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

    def get_csv_name(self):
        return os.path.join(OPT.csv_folder, f"{OPT.dataset}_{OPT.num_tasks}tasks_{self.name.replace('_','')}{self.buffer_size}_{OPT.model.replace('_','')}_epochs{OPT.epochs}.csv")
    def log_metrics(self, loss, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        print(f'        {tag}_{session:<6} - l:{loss:.5f}  a:{acc:.5f}')

        if OPT.tboard:
            writer.add_scalar(f'{tag}/loss/{session}', loss, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)
