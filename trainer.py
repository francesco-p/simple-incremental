import torch
from utils import check_output
import torch.nn.functional as F
from opt import OPT

class Trainer:
    """Trainer class for training a model on a dataset."""

    def __init__(self, model, device, num_classes, writer, seed, tag, checkpoint_path=OPT.chk_folder):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.writer = writer
        self.tag = tag
        self.checkpoint_path = checkpoint_path
        self.seed = seed

    def _train_epoch(self, train_loader, optimizer, loss_fn):
        """Train the model for one epoch."""
        cumul_loss_train = 0
        cumul_acc_train = 0
        seen = 0
        self.model.train()
        for x, y in train_loader:
            # Move to GPU
            x = x.to(self.device)
            y = y.to(self.device)

            # for p in self.model.parameters():
            #     print(p)
            #     break        
            #      
            # Forward data to model and compute loss
            y_hat = check_output(self.model(x))['y_hat']
            y_hat = y_hat.to(torch.float32)
            y_onehot = F.one_hot(y, num_classes=self.num_classes).to(torch.float32)
            loss_train = loss_fn(y_hat, y_onehot)

            # Backward
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Compute measures
            cumul_loss_train += loss_train.item()
            cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()
            seen += len(y)
        
        cumul_acc_train /= seen
        cumul_loss_train /= seen
        return cumul_loss_train, cumul_acc_train


    def eval(self, loader, loss_fn):
        """Evaluate the model on a dataset."""
        self.model.to(self.device)
        with torch.no_grad():
            cumul_loss_eval = 0
            cumul_acc_eval = 0
            seen = 0
            self.model.eval()
            for x, y in loader:
                # Move to GPU
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward data to model and compute loss
                y_hat = check_output(self.model(x))['y_hat']
                y_hat = y_hat.to(torch.float32)
                y_onehot = F.one_hot(y, num_classes=self.num_classes).to(torch.float32)
                loss_eval = loss_fn(y_hat, y_onehot)

                # Compute measures
                cumul_loss_eval += loss_eval.item()
                cumul_acc_eval += (y_hat.argmax(dim=1) == y).sum().item()
                seen += len(y)

            cumul_acc_eval /= seen
            cumul_loss_eval /= seen
            return cumul_loss_eval, cumul_acc_eval


    def train_eval(self, optimizer, loss_fn, epochs, train_loader, val_loader, eval_every=1):
        """Start training for a number of epochs."""
        self.model.to(self.device)

        for epoch in range(0, epochs):
            print(f'    EPOCH {epoch} ')

            tr_loss, tr_acc = self._train_epoch(train_loader, optimizer, loss_fn)
            self.log_metrics(tr_loss, tr_acc, epoch, 'train', self.writer)

            if (epoch == 0) or ((epoch % eval_every) == 0):
                val_loss, val_acc = self.eval(val_loader, loss_fn)
                self.log_metrics(val_loss, val_acc, epoch, 'eval', self.writer)
                torch.save(self.model.state_dict(), f'{self.checkpoint_path}/{self.tag}_epoch{epoch:04}_seed{self.seed}.pt')


    def log_metrics(self, loss, acc, epoch, session, writer):
        """ Prints metrics to screen and logs to tensorboard """
        print(f'        {session:<6} - loss:{loss:.5f} acc:{acc:.5f}')

        if writer is not None:
            writer.add_scalar(f'{self.tag}/loss/{session}', loss, epoch)
            writer.add_scalar(f'{self.tag}/acc/{session}', acc, epoch)
