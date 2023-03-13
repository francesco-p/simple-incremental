import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from opt import OPT
from strategies.base import Base
from utils import check_output
from tqdm import tqdm
import torchshow as ts
import os


def transform_core50_task_trainloader_to_tensor(train_loader):
    """Transform a trainloader into a tensor"""
    data = []
    labels = []
    for x, y in train_loader:
        data.append(x)
        labels.append(y)
    return torch.cat(data), torch.cat(labels)


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


class ICARL_ReplayMemory:
    """ Memory for iCaRL it does not require to implement the removal
      of the old classes since it is used in a NI settings 
      (not sure of this choice).
    """

    def __init__(self, phi, num_classes, img_size, emb_size, K=500) -> None:
        # Feature extractor (resnet32 in the paper)
        self.phi = phi
        # Embedding size
        self.emb_size = emb_size
        # Total number of examplars (2000 in the paper)
        self.K = K
        # Number of images per class
        self.m = K // num_classes
        # Examplar set
        self.P = torch.zeros((num_classes, self.m, *img_size))


    def get_examplars_dset(self):
        """ This method is used to get the examplars of the memory
        that have been computed in the update method
        This is used in the distillation loss

        Return a TensorDataset with the examplars and their labels
        """
        labels = torch.arange(self.P.shape[0]).repeat_interleave(self.m)
        data = self.P.view(-1, *self.P.shape[2:])
        return TensorDataset(data, labels)


    def compute_mean_feat_with_dloader(self, dataloader, num_workers=8):
        """ Compute the mean of the images of a class. We use this
        when we cannot use tensor.mean() because we have a lot of images
        """
        mu = None 
        for x in dataloader:
            x = x.to(OPT.DEVICE)
            x = x.to(torch.float32)
            #normalization = stats.DSET_NORMALIZATION[OPT.DATASET]
            #x = normalization(x)
            embeddings = self.phi(x)
            if mu == None:
                mu = embeddings.view(embeddings.shape[0], -1).sum(dim=0)
            else:
                mu += embeddings.view(embeddings.shape[0], -1).sum(dim=0)
        # len(dataloader.dataset) == the_images.shape[0]
        mu /= len(dataloader.dataset)
        return mu


    def update(self, data, labels):
        """ Construct the memory buffer examplar set for the current task 
        
        Args:
            data (torch.Tensor): images of the current task
            labels (torch.Tensor): labels of the current task
        """

        # Given a task, it extracts the images of each class
        print("Constructing examplar set...")
        for i in tqdm(range(0, OPT.NUM_CLASSES)):
            the_images = data[labels == i]

            # and select the most representative examplars to be stored
            self.update_class(the_images, i)


    def compute_distances(self, dataloader, the_class, k, mu):
        """ Compute the distances between the images of a class and its mean 
        this is to select the most representative images of the class

        Args:
            dataloader (DataLoader): dataloader of the images of the class
            the_class (int): class index
            k (int): number of examplars of the class currently stored
            mu (torch.Tensor): mean of the class
        """

        #-------------------#
        # This is a speedup hack to compute the examplars without using a dataloader
        # it works for a small batch size, if this does not work, use the commented code
        if k != 0:
            inp = self.P[the_class, :k].to(OPT.DEVICE)
            examplars_mu = self.phi(inp)[-1].mean(dim=0)
            examplars_mu = examplars_mu.view(-1).unsqueeze(0)
            self.P[the_class, :k].to('cpu')
        else:
            examplars_mu = torch.zeros((self.emb_size)).to(OPT.DEVICE).unsqueeze(0)
        #------- INSTEAD OF ------------#        
        #examplars_loader = DataLoader(self.P[the_class, :k], batch_size=OPT.BATCH_SIZE, shuffle=False)
        #examplars_mu = self.compute_mean_feat_with_dloader(examplars_loader)
        #-------------------#

        # Initialize the mean of the class
        mu = mu.unsqueeze(0)

        # Initialize the distances
        dist = torch.zeros((len(dataloader.dataset)))

        for i, x in enumerate(dataloader):
            x = x.to(OPT.DEVICE)
            x = x.to(torch.float32)
            #normalization = stats.DSET_NORMALIZATION[OPT.DATASET]
            #x = normalization(x)
            start_idx = i * OPT.BATCH_SIZE
            end_idx = start_idx + OPT.BATCH_SIZE
            
            # Compute the embeddings of the images
            embeddings = self.phi(x)
            embeddings = embeddings.view(embeddings.shape[0], -1)

            # Compute the distances as in the paper    
            if k == 0:
                dist[start_idx:end_idx] = torch.norm(mu - embeddings, dim=1)
            else:
                dist[start_idx:end_idx] = torch.norm(mu - 1/k * (embeddings + examplars_mu ), dim=1)

        return dist


    def update_class(self, the_images, the_class):
        """ implements algorithm 4 of the paper. 
        selects the most m representative images of a class 
        and updates the examplar set P

        Args:
            the_images (torch.Tensor): images of the class
            the_class (int): class index
        """

        if the_images.shape[0] < self.m:
            # This error happens when the number of images of a class is less than
            # the number of examplars per class
            raise ValueError(f"the_images.shape[0] = {the_images.shape[0]} < self.m = {self.m}")
        
        # First, we must modify the model to return only the features without the head
        self._remove_classifier_head()

        # Compute the mean of the images of the current class
        dataloader = DataLoader(the_images, batch_size=OPT.BATCH_SIZE, shuffle=False)
        mu = self.compute_mean_feat_with_dloader(dataloader)
        
        # Iteratively selects the most representative images up to m
        for k in range(self.m):
            # Initialize the distances
            dist = torch.zeros((the_images.shape[0]))

            # Compute the distances between the images and the mean
            dataloader = DataLoader(the_images, batch_size=OPT.BATCH_SIZE, shuffle=False)
            dist = self.compute_distances(dataloader, the_class, k, mu)
                
            # Find the image with the minimum distance
            idx = torch.argmin(dist, dim=0)

            # Add the image to the memory
            self.P[the_class, k] = the_images[idx] # p_k
            
            # Remove the image from the task
            the_images = torch.cat((the_images[:idx], the_images[idx+1:]))

        # Add the classifier head back to be able to train the model
        self._add_classifier_head()


    def _remove_classifier_head(self):
        """ Uninstall classifier head and save it temporalily """
        self.phi._temp_fc = self.phi.fc
        self.phi.fc = nn.Identity()

    def _add_classifier_head(self):
        """ Install classifier head """
        self.phi.fc = self.phi._temp_fc
        del self.phi._temp_fc
        

    def visualize_examplars(self, save_path='/tmp/examplars.png'):
        """ Visualize the examplars """
        # ts.set_image_mean([153.0076, 144.8722, 137.9779])
        # ts.set_image_std([54.9966, 56.9629, 60.5377])
        ts.save(self.P.view(-1, *self.P.shape[2:]), save_path, ncols=self.m)


class ICARL(Base):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=OPT.LR_CONT, weight_decay=OPT.WD_CONT)
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = "ICARL"
        self.replay_memory = ICARL_ReplayMemory(self.model, OPT.NUM_CLASSES, OPT.IMG_SHAPE, OPT.EMB_SIZE, K=OPT.MEMORY_SIZE)


    def compute_distillation_loss(self, old_model):
        """ Compute the output of the old model """
        
        # usage: kl_loss(F.log_softmax(input, dim=1), F.softmax(target, dim=1)
        # input should be probabilities in log space (to avoid underflow)
        # target should be probabilities
        kl_loss = nn.KLDivLoss(log_target=True)

        # Get the examplar set
        replay_buffer = self.replay_memory.get_examplars_dset()
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
            distill_loss += kl_loss(F.log_softmax(new_outputs, dim=1), F.softmax(old_outputs, dim=1))
            seen += x.shape[0]

        return distill_loss, seen


    def train(self, train_loader, val_loader, writer, tag, scheduler=False):

        self.model.to(OPT.DEVICE)
        
        # Previous frozen model
        old_model = copy.deepcopy(self.model)
        for p in old_model.parameters():
            p.requires_grad = False

        # We need to extract a tensor instead of a dataloader
        data, labels = transform_core50_task_trainloader_to_tensor(train_loader)

        # Update the examplar set
        with torch.no_grad():
            self.replay_memory.update(data, labels)
            self.replay_memory.visualize_examplars()

        
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


            # Compute the distillation loss on memory buffer
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
        if session == 'eval':
            print(f'        {tag}_{session:<6} - l_ce:{loss_ce:.5f}  a:{acc:.5f}')
        else:
            print(f'        {tag}_{session:<6} - l:{loss_dstll+loss_ce:.5f} = l_dstll:{loss_dstll:.5f} + l_ce:{loss_ce:.5f}  a:{acc:.5f}')

        if OPT.TENSORBOARD:
            writer.add_scalar(f'{tag}/loss_ce/{session}', loss_ce, epoch)
            writer.add_scalar(f'{tag}/loss_dstll/{session}', loss_dstll, epoch)
            writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)


    def get_csv_name(self):
        
        dset_task = f"{OPT.DATASET}_{OPT.NUM_TASKS}tasks"
        strategy_model = f"ICARL_mem{OPT.MEMORY_SIZE}_{OPT.MODEL.replace('_','')}"
        epochs =f"epochs{OPT.EPOCHS_CONT}.csv"

        fname = f"{dset_task}_{strategy_model}_{epochs}"
        return os.path.join(OPT.CSV_FOLDER, fname)
    
