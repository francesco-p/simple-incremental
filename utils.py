
import torch
import torch.nn.functional as F
from opt import OPT
import random
import numpy as np
import timm
import matplotlib.pyplot as plt
from models.resnet32 import resnet32

def plot(bottom_line, top_line, continual, approach_name):
    print(torch.tensor([x for y,x in continual]).mean(), torch.tensor([x for y,x in continual]).std())

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))
    ax.scatter(range(OPT.NUM_TASKS), [x for y,x in continual])
    ax.plot([x for y,x in continual], label=approach_name)
    ax.axhline(bottom_line, label='First Half', c='green', ls='-.')
    ax.axhline(top_line, label='Second Half', c='red', ls='-.')

    ax.set_ylim(bottom_line-0.01, top_line+0.01)
    ax.set_xticks(range(OPT.NUM_TASKS))
    ax.set_xlabel('tasks')
    ax.set_ylabel('accuracy')
    ax.legend()
    plt.show()
    plt.close()


def load_models(model_name, num_classes):

    model_fh =  get_model(model_name, num_classes, pretrained=False)
    model_fh.load_state_dict(torch.load(f'{OPT.CHK_FOLDER}/fh_0014.pt'))

    model_sh =  get_model(model_name, num_classes, pretrained=False)
    model_sh.load_state_dict(torch.load(f'{OPT.CHK_FOLDER}/sh_0014.pt'))

    return model_fh, model_sh


def get_model(model_name, num_classes, pretrained):

    if model_name == 'resnet18':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if model_name == 'dla46x_c':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if model_name == 'mobilenetv2_035':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet32':
        model = resnet32(num_classes=num_classes, pretrained=pretrained)
    else:
        NotImplementedError(f"Unknown model {model_name}")

    return model


def set_seeds(seed):
    """ Set reproducibility seeds """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def log_metrics(seen, loss, acc, epoch, session, writer, tag):
    """ Prints metrics to screen and logs to tensorboard """
    loss /= seen
    acc /= seen
    print(f'        {session:<6} - l:{loss:.5f}  a:{acc:.5f}')

    if OPT.LOG:
        writer.add_scalar(f'{tag}/loss/{session}', loss, epoch)
        writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)



def train_loop(optimizer, model, loss_fn, train_loader, val_loader, writer, tag, scheduler=False):
    model.to(OPT.DEVICE)
    if scheduler:
        #sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=OPT.EPOCHS, steps_per_epoch=len(train_loader))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(0, OPT.EPOCHS):
        print(f'    EPOCH {epoch} ')

        ##################
        #### Training ####
        cumul_loss_train = 0
        cumul_acc_train = 0
        seen = 0
        model.train()
        for x, y in train_loader:
            # Move to GPU
            x = x.to(OPT.DEVICE)
            y = y.to(OPT.DEVICE)

            # Forward data to model and compute loss
            y_hat = check_output(model(x))['y_hat']
            y_hat = y_hat.to(torch.float32)
            y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
            loss_train = loss_fn(y_hat, y_onehot)

            # Backward
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            #if scheduler:
            #    sched.step()

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
                    model.eval()
                    for x, y in val_loader:

                        # Move to GPU
                        x = x.to(OPT.DEVICE)
                        y = y.to(OPT.DEVICE)

                        # Forward to model
                        y_hat = check_output(model(x))['y_hat']
                        y_hat = y_hat.to(torch.float32)
                        y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
                        loss_val = loss_fn(y_hat, y_onehot)

                        # Compute measures
                        cumul_loss_val += loss_val.item()
                        cumul_acc_val += (y_hat.argmax(dim=1) == y).sum().item()
                        seen += len(y)

                    # Print measures
                    log_metrics(seen, cumul_loss_val, cumul_acc_val, epoch, 'val', writer, tag)

        if scheduler:
            sched.step(cumul_loss_val/seen)

        # Save the model
        if ((epoch % OPT.CHK_EVERY) == 0) and OPT.CHK:
            torch.save(model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}.pt')


def test(model, loss_fn, test_loader):
    model.to(OPT.DEVICE)

    with torch.no_grad():
        cumul_loss = 0
        cumul_acc = 0
        seen = 0
        model.eval()
        for x, y in test_loader:

            # Move to GPU
            x = x.to(OPT.DEVICE)
            y = y.to(OPT.DEVICE)

            # Forward to model
            y_hat = check_output(model(x))['y_hat']
            y_hat = y_hat.to(torch.float32)


            y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
            loss_val = loss_fn(y_hat, y_onehot)

            # Compute measures
            cumul_loss += loss_val.item()
            cumul_acc += (y_hat.argmax(dim=1) == y).sum().item()
            seen += len(y)

        # Prints metrics
        cumul_loss /= seen
        print(f'----------')
        print(f'        loss_test:{cumul_loss:.5f}')

        cumul_acc /= seen
        print(f'        acc_test:{cumul_acc:.5f}')

    return cumul_loss, cumul_acc


def check_output(out):
    """ Evil hack to check the output """
    out_dict = {}
    if type(out) == tuple:
        if out[0].shape == out[1].shape: #accomodates ojkd
            out_dict['bkb'] = out[0]
            out_dict['fr'] = out[1]
        else:
            out_dict['y_hat'], out_dict['fts'] = out[0], out[1]
    else:
        out_dict['y_hat'] = out
    return out_dict




