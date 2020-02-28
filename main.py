import torch

import torchvision.transforms as transforms

import sys
import yaml
import pprint

import train
import test
import test_multiple


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################################################
# configuration loading
########################################################################

def get_config(config):
    """
    get the config and parse it to a dictionary
    @param config: the path to the configuration
    @return: the config dictionary
    """
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


configuration = get_config(sys.argv[1])
action = configuration['action']
print('the configuration used is:')
pprint.pprint(configuration, indent=4)


########################################################################
# image loaders and unloaders
########################################################################

# image size
imsize = configuration['imsize']

# loaders
loaders = {
    'std':      transforms.Compose(
                    [transforms.Resize(imsize),
                     transforms.RandomResizedCrop(256),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'no_norm':  transforms.Compose(
                    [transforms.Resize(imsize),
                     transforms.RandomResizedCrop(256),
                     transforms.ToTensor()])
}

# unloaders
unloaders = {
    'std':      transforms.Compose(
                    [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                     transforms.ToPILImage()]),
    'saving':   transforms.Compose(
                    [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]),
    'no_norm':  transforms.Compose(
                    [transforms.ToPILImage()]),
    'none':     None
}

configuration['loader'] = loaders[configuration['loader']]
configuration['unloader'] = unloaders[configuration['unloader']]

########################################################################
# main method
########################################################################

if __name__ == '__main__':
    if action == 'train':
        print('starting main training loop (torch lua model) with specified configuration')
        train.train(configuration)

    elif action == 'test':
        print('starting test loop for content and style images')
        test.test(configuration)

    elif action == 'test_multiple':
        print('starting test loop for content and style images with multiple result images')
        test_multiple.test(configuration)
