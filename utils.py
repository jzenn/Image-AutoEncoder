import torch
import torchvision.utils as utils

import datetime
import pytz
import os
import glob

from PIL import Image

# the date-time format
fmt = '%d_%m__%H_%M_%S'


def save_current_model(configuration, epoch, model, optimizer_state_dict,
                       content_loss, style_loss):
    """
    save the current model state dicts
    @param configuration: the config file
    @param epoch: the current epoch
    @param model: the current model
    @param optimizer_state_dict: the optimizer state_dict
    @param content_loss: the current content loss
    @param style_loss: the current style loss
    @return:
    """

    model_state_dict = model.state_dict()
    lambda_1 = configuration['lambda_1']
    lambda_2 = configuration['lambda_2']
    model_saving_path = configuration['model_saving_path']
    encoder_saving_path = configuration['encoder_saving_path']
    decoder_saving_path = configuration['decoder_saving_path']

    try:
        encoder_saving_dict = {
            'encoder': model.encoder.state_dict()
        }
    except:
        encoder_saving_dict = {
            'encoder': model.module.encoder.state_dict()
        }

    try:
        decoder_saving_dict = {
            'decoder': model.decoder.state_dict()
        }
    except:
        decoder_saving_dict = {
            'decoder': model.module.decoder.state_dict()
        }

    # save a checkpoint
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'content_loss': content_loss,
            'style_loss': style_loss
            },
        '{}/encoder_decoder_model_{}_{}__{}'.format(model_saving_path, lambda_1, lambda_2,
                             datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt)))

    # save encoder and decoder separately
    torch.save(encoder_saving_dict, '{}/encoder.pth'.format(encoder_saving_path))
    torch.save(decoder_saving_dict, '{}/decoder.pth'.format(decoder_saving_path))


def save_images(lambda_1, lambda_2, epoch, step, ground_truth_c, result_image_c, ground_truth_s, result_image_s,
                transformation, image_saving_path, batch_size, use_transformation=True):
    """
    saves a random sample of a batch of images
    @param lambda_1: lambda_1 of the loss
    @param lambda_2: lambda_2 of the loss
    @param epoch: the current epoch
    @param step: the current step
    @param ground_truth_c: the ground truth content image
    @param result_image_c: the result content image
    @param ground_truth_s: the ground truth style image
    @param result_image_s: the result style image
    @param transformation: a transformation to be applied
    @param image_saving_path: the path where the images are saved to
    @param batch_size: the batch size
    @param use_transformation: whether to use a transformation
    @return:
    """

    rand = int(torch.rand(1).item() * batch_size)

    if use_transformation:
        result_image_c = result_image_c[rand].cpu().clone().squeeze(0)
        ground_truth_c = transformation(ground_truth_c[rand].cpu().clone().squeeze(0))
        result_image_s = result_image_s[rand].cpu().clone().squeeze(0)
        ground_truth_s = transformation(ground_truth_s[rand].cpu().clone().squeeze(0))
    else:
        result_image_c = result_image_c[rand].cpu().clone().squeeze(0)
        ground_truth_c = ground_truth_c[rand].cpu().clone().squeeze(0)
        result_image_s = result_image_s[rand].cpu().clone().squeeze(0)
        ground_truth_s = ground_truth_s[rand].cpu().clone().squeeze(0)

    utils.save_image([ground_truth_s, result_image_s, ground_truth_c, result_image_c],
                     '{}/encoder_decoder_image_{}_{}__{}_{}__{}.jpeg'.format(
                         image_saving_path, lambda_1, lambda_2, epoch, step,
                         datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/Berlin')).strftime(fmt)),
                     normalize=True, scale_each=True)


def get_latest_model(configuration):
    """
    get the latest model in the encoder_model_path of the configuration
    @param configuration:
    @return:
    """
    model_path = configuration['encoder_model_path'] + '/encoder_*'
    latest_file = max(glob.iglob(model_path), key=os.path.getctime)
    print('the latest model is obviously {}'.format(latest_file))
    return latest_file


def imnorm(tensor, transformation):
    """
    normalizes an image recived from EncoderDecoder
    :param tensor: the image as tensor
    :param transformation: a transformation applied to the image before saving
    :return:
    """
    # clone the tensor to not change the original one
    image = tensor.cpu().clone()
    # remove the batch dimension
    image = image.squeeze(0)
    if transformation is not None:
        image = transformation(image)

    return image


def image_loader(image_name, transformation, add_fake_batch_dimension=True):
    """
    loads an image
    :param image_name: the path of the image
    :param transformation: the transformation done on the image
    :param add_fake_batch_dimension: should add a 4th batch dimension
    :return: the image on the current device
    """
    image = Image.open(image_name).convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    if add_fake_batch_dimension:
        image = transformation(image).unsqueeze(0)
    else:
        image = transformation(image)
    return image
