from torchvision import transforms as trs

DSET_CLASSES = {
        'CIFAR100':100,
        'CIFAR10':10,
        'SVHN':10
    }


DSET_TRANSF = {

    "CIFAR10": trs.Compose([trs.RandomCrop(32, padding=4, padding_mode='reflect'),
                trs.RandomHorizontalFlip(),
                trs.ToTensor(),
                trs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),

    "CIFAR100": trs.Compose([trs.RandomCrop(32, padding=4, padding_mode='reflect'),
                trs.RandomHorizontalFlip(),
                trs.ToTensor(),
                trs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),

    "SVHN": trs.Compose([trs.RandomHorizontalFlip(),
                trs.ToTensor()])
}
