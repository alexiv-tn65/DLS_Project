import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from models.loss_functions import StyleLoss, ContentLoss
from models.image_preparation import Normalization


class StyleTransferNNet(nn.Module):
    def __init__(self, device):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_default_device(self.device)
        else:
            self.device: str = device

        # desired size of the output image
        self.imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
        # self.imsize = 512 if torch.cuda.is_available() else 256  # use small size if no GPU

        self.loader = transforms.Compose([
        transforms.Resize((self.imsize,self.imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

        # self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        # self.cnn = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(self.device).eval()
        self.cnn = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.to(self.device).eval()

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def get_style_model_and_losses(self, style_img, content_img):

        # desired depth layers to compute style/content losses :
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

        # normalization module
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

        # just in order to have an iterable access to or list of content/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def run_style_transfer(self, content_img, style_img, result_path, num_steps=100, 
        style_weight=1000000, content_weight=1):

        """Run the style transfer."""
        # print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)

        input_img = content_img

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        model.eval()
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        # print('Optimizing..')
        for _ in tqdm(range(num_steps)):

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        if result_path:
            save_image(input_img[0], result_path)

        return input_img

    def image_loader(self, image_path):
        image = Image.open(image_path)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

