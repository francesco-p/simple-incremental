from torchvision import transforms as trs

DSET_IMG_SHAPE = {

    "CIFAR10": (3, 32, 32),
    "CIFAR100": (3, 32, 32),
    "SVHN": (3, 32, 32),
    "Core50": (3, 128, 128)
}



DSET_CLASSES = {
        'CIFAR100':100,
        'CIFAR10':10,
        'SVHN':10,
        'Core50':50
    }


DSET_TRANSF = {

    "CIFAR10": trs.Compose([trs.RandomCrop(32, padding=4, padding_mode='reflect'),
                trs.RandomHorizontalFlip(),
                trs.ToTensor(),
                trs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),

    "CIFAR100": trs.Compose([trs.RandomCrop(32, padding=4, padding_mode='reflect'),
                trs.RandomHorizontalFlip(),
                trs.ToTensor(),
                trs.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
                ),
                #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    "SVHN": trs.Compose([trs.RandomHorizontalFlip(),
                trs.ToTensor()]),
    
    "Core50": trs.Compose([
        trs.ToPILImage(),
        trs.RandomHorizontalFlip(),
        trs.ToTensor(),
        trs.Normalize((153.0076, 144.8722, 137.9779), (54.9966, 56.9629, 60.5377))
    ])
}


DSET_NORMALIZATION = {
    "CIFAR10": trs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "CIFAR100": trs.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),

    # is it correct copilot suggested this?????
    "SVHN": trs.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
}
