import kfs_models.resnet as RN
import kfs_models.resnet_ap as RNAP
import kfs_models.convnet as CN
import kfs_models.densenet_cifar as DN
import kfs_models.elm as RP
from efficientnet_pytorch import EfficientNet
import timm
import torch

def get_model(args, model, channel, num_classes, im_size=(32, 32)):
    if model == 'ConvNet':
        model = CN.ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=128,
            net_depth=3,
            net_act='relu',         
            net_norm='instance',
            net_pooling='avgpooling',
            im_size=im_size
        )

    elif model == 'ConvNet4':
        model = CN.ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=128,
            net_depth=4,
            net_act='relu',         
            net_norm='instance',
            net_pooling='avgpooling',
            im_size=im_size
        )

    elif model == 'ConvNetBN':
        model = CN.ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=128,
            net_depth=3,
            net_act='relu',         
            net_norm='batchnorm',
            net_pooling='maxpooling',
            im_size=im_size
        )

    elif model == 'ResNet10':
        model = RN.ResNet(
            args.dataset,
            10,
            num_classes,
            norm_type='instance',
            size=im_size[0],
            nch=channel
        )

    elif model == 'ResNet10BN':
        model = RN.ResNet(
            args.dataset,
            10,
            num_classes,
            norm_type='batch',
            size=im_size[0],
            nch=channel
        )

    elif model == 'ResNet18BN':
        model = RN.ResNet(
            args.dataset,
            18,
            num_classes,
            norm_type='batch',
            size=im_size[0],
            nch=channel
        )

    elif model == 'ResNet10AP':
        model = RNAP.ResNetAP(
            args.dataset,
            10,
            num_classes,
            norm_type='instance',
            size=im_size[0],
            nch=channel
        )
    elif model == 'efficientnet':
        model = timm.create_model('efficientnet_b0', num_classes = num_classes, pretrained = False)
        embed = torch.nn.Sequential(*[l for l in list(model.children())[:6]])
    
    elif model == 'Trans':
        model = timm.create_model('vit_tiny_patch16_224', num_classes = num_classes, pretrained = False)
    
    elif model == 'RP':#random projection
        inp = channel*im_size[0]*im_size[1]
        hid = args.RP_hid
        model = RP.ELM(inp, hid, num_classes, args.no_init)


    # this is only used for CIFAR10, 100
    elif model == 'DenseNet':
        model = DN.densenet_cifar(num_classes)

    # this is only used for ImageNet
    elif model == 'Efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)

    else:
        raise Exception('unknown network architecture: {}'.format(model))

    return model
