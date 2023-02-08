import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from opt import OPT
from strategies.base import Base
from utils import check_output
from einops import rearrange

class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # required int cast to concatenate the datasets
        # pytorch datasets return int as labels
        return self.x[idx], int(self.y[idx])


class ExamplarMemoryNI:
    """ Memory for iCaRL it does not require to implement the removal of the old classes
    since it is used in a NI setting K=50 because it is VERY slow otherywise.
    """
    def __init__(self, phi, num_classes, img_size, emb_size, K=500) -> None:
        # Feature extractor (should be a resnet32 as in the paper)
        self.phi = phi
        # Embedding size
        self.emb_size = emb_size
        # Total number of examplresnet32t with the examplars """
        labels = torch.arange(self.P.shape[0]).repeat_interleave(self.m)
        data = self.P.view(-1, *self.P.shape[2:])
        return TensorDataset(data, labels)

    def compute_mean_with_dloader(self, the_images, num_workers=8):
        """ Compute the mean of the images of a class 
        it would be easier to use tensor.mean() but it is not possible
        when you have a lot of images, therefore we use a dataloader
        """
        # Convert the tensor to a TensorDataset with dummy labels
        dataloader = DataLoader(the_images, batch_size=OPT.BATCH_SIZE, shuffle=False, num_workers=num_workers)

        # Compute the mean
        mu = torch.zeros(self.emb_size).to(OPT.DEVICE)
        for x in dataloader:
            x = x.to(OPT.DEVICE)
            x = x.to(torch.float32)
            # Compute the embeddings of the images
            embeddings = check_output(self.phi(x))['fts']
            mu += embeddings.sum(dim=0)
        mu /= len(dataloader.dataset)   # len(dataloader.dataset) = the_images.shape[0]

        return mu


    def construct_examplar_set(self, data, labels):
        """ Construct the examplar set for the first task """
        
        # Get the images of the first class
        for i in range(0, OPT.NUM_CLASSES):
            the_images = data[labels == i]
            # Select the examplars
            self.update_examplars(the_images, i)


    def compute_distances(self, the_images, the_class, k, mu):
        """ Compute the distances between the images and the mean"""
        
        dataloader = DataLoader(the_images, batch_size=OPT.BATCH_SIZE, shuffle=False, num_workers=OPT.NUM_WORKERS)
        mu = mu.unsqueeze(0)
        examplars_mu = self.compute_mean_with_dloader(self.P[the_class, :k])
        dist = torch.zeros((the_images.shape[0]))

        for i, x in enumerate(dataloader):
            x = x.to(OPT.DEVICE)
            x = x.to(torch.float32)
            start_idx = i * OPT.BATCH_SIZE
            end_idx = start_idx + OPT.BATCH_SIZE
            # Compute the embeddings of the images
            embeddings = check_output(self.phi(x))['fts']
            if k == 0:
                dist[start_idx:end_idx] = torch.norm(mu - embeddings, dim=1)
            else:
                dist[start_idx:end_idx] = torch.norm(mu - 1/k * (embeddings + examplars_mu ), dim=1)
        return dist


    def update_examplars(self, the_images, the_class):
        """ implements algorithm 4 of the paper"""

        if the_images.shape[0] < self.m:
            raise ValueError(f"the_images.shape[0] = {the_images.shape[0]} < self.m = {self.m}")

        # Real mean
        mu = self.compute_mean_with_dloader(the_images)
        
        # Create the examplar set
        for k in range(self.m):
            # Distances between the images and the mean 
            dist = torch.zeros((the_images.shape[0]))

            # Compute the distances
            dist = self.compute_distances(the_images, the_class, k, mu)
                
            # Find the image with the minimum distance
            idx = torch.argmin(dist, dim=0)
            # Add the image to the memory
            self.P[the_class, k] = the_images[idx] # p_k
            # Remove the image from the dataset 
            the_images = torch.cat((the_images[:idx], the_images[idx+1:]))




class iCaRL(Base):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.LR_CONT, weight_decay=OPT.WD_CONT)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "iCaRL"
        self.replay_buffer = ExamplarMemoryNI(self.model, OPT.NUM_CLASSES, OPT.IMG_SHAPE, OPT.EMB_SIZE, K=OPT.MEMORY_SIZE)


    def compute_distillation_loss(self, old_model):
        """ Compute the output of the old model """
        
        # Get the examplar set
        replay_buffer = self.replay_buffer.get_examplars_dset()
        loader = DataLoader(replay_buffer, batch_size=OPT.BATCH_SIZE, shuffle=True, num_workers=OPT.NUM_WORKERS)
        
        old_model.to(OPT.DEVICE)
        old_model.train()
        seen = 0
        distill_loss = 0.
        for x, y in loader:
            x = x.to(OPT.DEVICE)
            y = y.to(OPT.DEVICE)

            old_outputs = check_output(old_model(x))['y_hat']
            new_outputs = check_output(self.model(x))['y_hat']

            # Compute the distillation loss
            distill_loss += nn.KLDivLoss()(F.log_softmax(old_outputs, dim=1), F.softmax(new_outputs, dim=1))
            seen += x.shape[0]

        return distill_loss, seen


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):
        
        # Move model to GPU
        self.model.to(OPT.DEVICE)

        ## Create a new DataLoader with the concatenated data
        #concat_dataloader = DataLoader(
        #    ConcatDataset((self.replay_buffer.get_examplars_dset(), train_loader.dataset)),
        #    batch_size=OPT.BATCH_SIZE,
        #    shuffle=True # important to shuffle the data
        #)
        
        # Previous frozen model
        old_model = copy.deepcopy(self.model)
        for p in old_model.parameters():
            p.requires_grad = False


        # Update the examplar set from subsa
        indices = train_loader.dataset.indices
        # trnsf = train_loader.dataset.dataset.transform
        data = torch.tensor(train_loader.dataset.dataset.data[indices])
        data = rearrange(data, 'b h w c -> b c h w')
        labels = torch.tensor(train_loader.dataset.dataset.targets)[indices]
        with torch.no_grad():
            self.replay_buffer.construct_examplar_set(data, labels)

        
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

                # NEW
                y_hat = check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                
                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
                loss_train = self.loss_fn(y_hat, y_onehot)

                # Backward
                loss_train.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Compute measures
                cumul_loss_train += loss_train.item()
                cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)


            # Compute the distillation loss
            loss_dstll, seen_dstll = self.compute_distillation_loss(old_model)
            loss_dstll.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            loss_dstll = loss_dstll.item() / seen_dstll
            cumul_acc_train /= seen
            cumul_loss_train /= seen
            # Print measures
            self.log_metrics(cumul_loss_train, loss_dstll, cumul_acc_train, epoch, 'train', writer, tag)

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
                y_hat = check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                y_hat = torch.softmax(y_hat, dim=1)
                
                y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
                loss_test = self.loss_fn(y_hat, y_onehot)

                # Compute measures
                cumul_loss_eval += loss_test.item()
                cumul_acc_eval += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            cumul_acc_eval /= seen
            cumul_loss_eval /= seen
            # Print measures
            self.log_metrics(cumul_loss_eval, -1, cumul_acc_eval, -1, 'eval', writer, tag)
        
        return cumul_loss_eval, cumul_acc_eval


    def log_metrics(self, loss_ce, loss_dstll, acc, epoch, session, writer, tag):
        """ Prints metrics to screen and logs to tensorboard """
        print(f'        {tag}_{session:<6} - l:{loss_dstll+loss_ce:.5f} = l_dstll:{loss_dstll:.5f} + l_ce:{loss_ce:.5f}  a:{acc:.5f}')

        if OPT.TENSORBOARD:
            writer.add_scalar(f'{tag}/loss_ce/{session}', loss_ce, epoch)
            writer.add_scalar(f'{tag}/loss_dstll/{session}', loss_dstll, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)
