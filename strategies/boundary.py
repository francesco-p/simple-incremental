from opt import OPT
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
from torch import optim
from strategies.base import Base
import torchvision
from functorch import make_functional_with_buffers, vmap, grad
import matplotlib.pyplot as plt
import numpy as np
import os

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class MemoryBuffer():
    def __init__(self, buffer_size=500) -> None:
        self.buffer_size = buffer_size
        self.buffer = None
        self.labels = None
        self.grads = None
        self.seen = 0

    def __len__(self):
        if self.buffer is None:
            return 0

        return len(self.labels)
    
    def push(self, sample, label, grad):
        sample = sample.cpu()
        label = label.cpu()
        grad = grad.cpu()

        if self.buffer is None:
            self.buffer = sample
            self.grads = grad
            self.labels = label
        elif len(self.buffer) < self.buffer_size:
            self.buffer = torch.cat((self.buffer, sample), dim=0)
            self.grads = torch.cat((self.grads, grad), dim=0)
            self.labels = torch.cat((self.labels, label), dim=0)
        else:
            self.buffer = torch.cat((self.buffer, sample), dim=0)
            self.grads = torch.cat((self.grads, grad), dim=0)
            self.labels = torch.cat((self.labels, label), dim=0)
            indices = self.grads.argsort()[-self.buffer_size:]
            self.buffer = self.buffer[indices]
            self.grads = self.grads[indices]
            self.labels = self.labels[indices]

    def get_next_examples(self, len_loader):
        increment = int((len(self.buffer) / len_loader) + 0.5)
        
        replay_x = self.buffer[self.seen:self.seen+increment]
        replay_y = self.labels[self.seen:self.seen+increment]

        if self.seen+increment >= len(self.buffer):
            self.seen = 0
        else:
            self.seen += increment
            
        return replay_x, replay_y

    def viz_mem(self):
        matplotlib_imshow(torchvision.utils.make_grid(self.buffer.cpu()))
        
        plt.show()
    

class Boundary(Base):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=OPT.LR_CONT, weight_decay=OPT.WD_CONT)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "Boundary"
        self.replay_memory = MemoryBuffer()


    def compute_grad(self, sample, target):
        sample = sample.unsqueeze(0)  # prepend batch dimension for processing
        target = target.unsqueeze(0)
        prediction = self.model(sample)
        loss = self.loss_fn(prediction, target)
        return torch.autograd.grad(loss, list(self.model.parameters()))


    def compute_sample_grads(self, data, targets):
        """ manually process each sample with per sample gradient """
        sample_grads = [self.compute_grad(data[i], targets[i]) for i in range(data.shape[0])]
        sample_grads = zip(*sample_grads)
        sample_grads = [torch.stack(shards) for shards in sample_grads]
        return sample_grads


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
            self.model.eval()
            for x, y in train_loader:

                # Move to GPU
                x = x.to(OPT.DEVICE)
                y = y.to(OPT.DEVICE)

                # Augment the batch with memory data
                if tag != '0':
                    replay_x, replay_y = self.replay_memory.get_next_examples(len(train_loader))
                    x = torch.cat((x, replay_x.to(OPT.DEVICE)), dim=0)
                    y = torch.cat((y, replay_y.to(OPT.DEVICE)), dim=0)


                # Forward data to model and compute loss
                y_hat = utils.check_output(self.model(x))['y_hat']  
                y_hat = y_hat.to(torch.float32)
                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
                
                # I check the gradients only at the last epoch to be faster
                if epoch == (OPT.EPOCHS_CONT-1):
                    per_sample_grads = self.compute_sample_grads(x, y_onehot)
                    collapsed_grads = [x.view(x.shape[0], -1).sum(dim=1) for x in per_sample_grads]
                    gradiets_per_example = 0.
                    for gradient in collapsed_grads:
                        gradiets_per_example += gradient

                    # get biggest gradient and keep tot (keep < batch size)
                    sorted_idx = gradiets_per_example.argsort() 
                    example_with_max_grad = x[sorted_idx][:OPT.KEEP]
                    label_with_max_grad = y[sorted_idx][:OPT.KEEP]
                    max_grad = gradiets_per_example[sorted_idx][:OPT.KEEP]

                    self.replay_memory.push(example_with_max_grad, label_with_max_grad, max_grad)
                    

                else:
                    loss_train = self.loss_fn(y_hat, y_onehot)
                    loss_train.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Record & update learning rate
                if scheduler:
                    sched.step()

                # Compute measures
                if epoch == (OPT.EPOCHS_CONT-1):
                    cumul_loss_train +=  0.
                    
                else:
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
                # Move to GPU
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
        return os.path.join(OPT.CSV_FOLDER, f"{OPT.DATASET}_{OPT.NUM_TASKS}tasks_{self.name.replace('_','')}_{OPT.MODEL.replace('_','')}.csv")

