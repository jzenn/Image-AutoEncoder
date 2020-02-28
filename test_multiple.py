import torch
import torchvision.utils as u

import os

import net
import utils

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(configuration):
    """
    test loop that produces several output images, given one input image
    @param configuration: the config
    @return:
    """
    number_of_models = len(configuration['encoder_model_path_list'])
    content_images_path = configuration['content_images_path']
    style_images_path = configuration['style_images_path']
    loader = configuration['loader']
    image_saving_path = configuration['image_saving_path']

    model_list = [0 for _ in range(number_of_models)]
    for i in range(len(model_list)):
        encoder_decoder_model = net.get_pretrained_encoder_decoder_model(configuration, use_list=True, list_index=i)
        model_list[i] = encoder_decoder_model

    number_content_images = len(os.listdir(content_images_path))
    number_style_images = len(os.listdir(style_images_path))
    content_image_files = ['{}/{}'.format(content_images_path, os.listdir(content_images_path)[i])
                           for i in range(number_content_images)]
    style_image_files = ['{}/{}'.format(style_images_path, os.listdir(style_images_path)[i])
                         for i in range(number_style_images)]

    all_image_files = style_image_files + content_image_files

    print('got {} images'.format(len(all_image_files)))

    for i in range(len(all_image_files)):
        print('at image {}'.format(i))
        with torch.no_grad():
            image = utils.image_loader(all_image_files[i], loader)

            result_images = [0 for _ in range(len(model_list))]

            for k in range(len(model_list)):
                output, _, _, _ = model_list[k](image)
                result_images[k] = utils.imnorm(output, None)

            # save all images in one row
            u.save_image([utils.imnorm(image, None)] + result_images, '{}/encoder_decoder_test_image_A_{}.jpeg'
                         .format(image_saving_path, i), normalize=True, scale_each=True, pad_value=1)

            # save all images in two rows
            u.save_image([utils.imnorm(image, None)] + result_images, '{}/encoder_decoder_test_image_B_{}.jpeg'
                         .format(image_saving_path, i), normalize=True, scale_each=True, pad_value=1, nrow=4)

            # save all images except the image with balancing 0:1 in one row
            u.save_image([utils.imnorm(image, None)] + result_images[1:], '{}/encoder_decoder_test_image_C_{}.jpeg'
                         .format(image_saving_path, i), normalize=True, scale_each=True, pad_value=1)
