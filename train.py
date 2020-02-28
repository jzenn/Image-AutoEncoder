import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import data_loader as dl
import utils
import net

# device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(configuration):
    """
    this is the main training loop
    :param configuration: the config
    :return:
    """
    epochs = configuration['epochs']
    print('going to train for {} epochs'.format(epochs))

    step_printing_interval = configuration['step_printing_interval']
    print('writing to console every {} steps'.format(step_printing_interval))

    epoch_saving_interval = configuration['epoch_saving_interval']
    print('saving the model every {} epochs'.format(epoch_saving_interval))

    image_saving_interval = configuration['image_saving_interval']
    print('saving the images every {} steps'.format(image_saving_interval))

    concat_dataloader = dl.get_concat_dataloader(configuration)
    print('got dataloader')

    lambda_1 = configuration['lambda_1']
    print('lambda_1 {}'.format(lambda_1))

    lambda_2 = configuration['lambda_2']
    print('lambda_2 {}'.format(lambda_2))

    encoder_decoder_model = net.get_encoder_decoder_model(configuration)
    print('got model')

    tensorboardX_path = configuration['tensorboardX_path']
    print('saving tensorboardX logs to {}'.format(tensorboardX_path))

    print('writing tensorboardX runs to {}'.format(tensorboardX_path))
    writer = SummaryWriter(logdir='{}/runs'.format(tensorboardX_path))

    # only optimize the decoder in the model (!)
    try:
        optimizer = optim.Adam(encoder_decoder_model.decoder.parameters(), lr=configuration['lr'])
    except:
        optimizer = optim.Adam(encoder_decoder_model.module.decoder.parameters(), lr=configuration['lr'])
    print('got optimizer')

    use_pretrained_model = configuration['use_pretrained_model']
    pretrained_model_path = configuration['pretrained_model_path']

    if use_pretrained_model:
        print('using pretrained model from {}'.format(pretrained_model_path))
        latest_model = utils.get_latest_model(configuration)
        checkpoint = torch.load(latest_model, map_location='cpu')
        encoder_decoder_model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded encoder_decoder_model state dict')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loaded optimizer state dict')

    epoch = 0

    print('make iterable from dataloader')
    data_loader = iter(concat_dataloader)
    print('data loader iterable')
    i = -1
    iteration = 0

    while epoch < epochs:
        epoch += 1
        print('epoch: {}'.format(epoch))

        while True:
            try:
                data = data_loader.__next__()
                i += 1
            except StopIteration:
                print('got to the end of the dataloader (StopIteration)')
                data_loader = iter(concat_dataloader)
                i = 0
                break
            except:
                print('something went wrong with the dataloader')
                continue

            # get the content_image batch
            content_image = data.get('coco').get('image')
            if content_image is None:
                print('something went wrong with the content_image')
                continue
            content_image = content_image.to(device)

            # get the style_image batch
            style_image = data.get('painter_by_numbers').get('image')
            if style_image is None:
                print('something went wrong with the content_image')
                continue
            style_image = style_image.to(device)

            iteration += 1

            # at first the content image
            # set all the gradients to zero
            optimizer.zero_grad()
            output_content, loss, content_feature_loss, per_pixel_loss = encoder_decoder_model(content_image)

            # backprop
            loss_c = loss.sum()
            loss_c.backward()
            optimizer.step()

            if iteration % 100 == 0:
                writer.add_scalar('data/training_loss_style', torch.sum(loss_s).item(), iteration)
                writer.add_scalar('data/training_feature_loss_style', torch.sum(content_feature_loss).item(), iteration)
                writer.add_scalar('data/training_per_pixel_loss_style', torch.sum(per_pixel_loss).item(), iteration)

            # the same for the style image
            # set all the gradients to zero
            optimizer.zero_grad()
            output_style, loss, content_feature_loss, per_pixel_loss = encoder_decoder_model(style_image)
            # backprop
            loss_s = loss.sum()
            loss_s.backward()
            optimizer.step()

            if iteration % 10 == 0:
                writer.add_scalar('data/training_loss_content', torch.sum(loss_c).item(), iteration)
                writer.add_scalar('data/training_feature_loss_content', torch.sum(content_feature_loss).item(), iteration)
                writer.add_scalar('data/training_per_pixel_loss_content', torch.sum(per_pixel_loss).item(), iteration)

            # print every step_printing_interval the loss
            if i % step_printing_interval == 0:
                print('epoch {}, step {}'.format(epoch, i))
                print('style_loss: {:4f}, content_loss: {:4f}'.format(torch.sum(loss_s).item(), torch.sum(loss_c).item()))

            # save every epoch_saving_interval the current model
            if i % image_saving_interval == 0 and epoch % epoch_saving_interval == 0:
                utils.save_current_model(configuration, epoch, encoder_decoder_model, optimizer.state_dict(),
                                         loss_c, loss_s)

            # save every image_saving_interval the processed images
            if i % image_saving_interval == 0:
                utils.save_images(lambda_1, lambda_2, epoch, i, content_image, output_content, style_image,
                                  output_style, configuration['unloader'], configuration['image_saving_path'],
                                  configuration['batch_size'], use_transformation=False)
