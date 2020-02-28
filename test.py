import torch
import torch.nn as nn
import torchvision.utils as u

import os

import net
import utils

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(configuration):
    """
    test loop that produces an output image, given an input image
    @param configuration: the config
    @return:
    """
    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']
    loader = configuration['loader']
    image_test_saving_path = configuration['image_saving_path']

    encoder_decoder_model = net.get_pretrained_encoder_decoder_model(configuration)
    encoder_decoder_model = nn.DataParallel(encoder_decoder_model)
    encoder_decoder_model.eval()

    number_content_images = len(os.listdir(content_images_path))
    number_style_images = len(os.listdir(style_images_path))
    content_image_files = ['{}/{}'.format(content_images_path, os.listdir(content_images_path)[i])
                           for i in range(number_content_images)]
    style_image_files = ['{}/{}'.format(style_images_path, os.listdir(style_images_path)[i])
                         for i in range(number_style_images)]

    for i in range(number_style_images):
        print("test_image {} at {}".format(i + 1, style_image_files[i]))

    for i in range(number_content_images):
        print("test_image {} at {}".format(i + 1, content_image_files[i]))

    test_image_files = content_image_files + style_image_files

    for i, file in enumerate(test_image_files):
        image = utils.image_loader(file, loader)

        with torch.no_grad():
            result_image = encoder_decoder_model(image)[0]

        u.save_image([utils.imnorm(image, None),
                          utils.imnorm(result_image, None)],
                         '{}/image_{}.jpeg'.format(image_test_saving_path, i), normalize=True, pad_value=1)
