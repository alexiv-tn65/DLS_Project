from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




def save_output_img(img_tensor, user_id):
    """
    This function saves style transfer resulted tensor
    to JPEG file.
    """
    PIL_transform = transforms.ToPILImage()
    img_tensor = img_tensor.cpu().squeeze(0)
    img = PIL_transform(img_tensor)
    output_img_name = user_id + '_output_img.jpg'
    img.save(output_img_name)


class VGG_Head(nn.Module):
    """
    This class reproduces head of VGG19 which contains
    only first 5 convolutions.
    """

    def __init__(self, cfg):
        super(VGG_Head, self).__init__()
        self.features = self.make_layers(cfg)

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


class NST_Assy():
    """
    Assembly of CNN network for image style transfer
    """

    def __init__(self, content_img, style_img, input_img, gamma, device):
        self.input_img = input_img
        self.gamma = max(gamma, 1e-5)
        self.model, self.style_loss, self.content_loss = self.assemble_model(
            content_img, style_img, device)
        self.optimizer = optim.LBFGS([self.input_img.requires_grad_()])

    def assemble_model(self, content_img, style_img, device):
        """
        Built VGG19 head and load pretrained weights.
        """
        cfg = [64, 64, 'M', 128, 128, 'M', 256]
        cnn = VGG_Head(cfg).features
        # load pretrained weights:
        cnn.load_state_dict(torch.load('VGG19_Head.pth'))
        cnn.to(device)

        # normalization module
        normalization = Normalization().to(device)

        # Lists of content/style losses:
        content_losses = []
        style_losses = []

        # for the TG Bot we are not planning to change model architecture
        # Therefore, we will hardcode locations of Style and Content loss
        # layers within NST model class:
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # start to assemble a new model of NST:
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # Redefine ReLU to make it out-of-place:
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

        # trim off the layers after the last content and style losses
        # In fact, this loop will break at the first iteration since we are
        # using modified CNN that represents only required layers from VGG19.
        # But this functionality will be usefull for possible modificaitons.
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def fit(self, num_steps=300):
        """Run the style transfer."""

        # Set weights for content and style losses:
        content_weight = 1
        style_weight = int(self.gamma * 1e5)

        run = [0]
        while run[0] <= num_steps:

            def closure():
                # values should be in range [0;1]
                self.input_img.data.clamp_(0, 1)

                self.optimizer.zero_grad()

                self.model(self.input_img)

                style_score = 0
                content_score = 0

                for sl in self.style_loss:
                    style_score += sl.loss
                for cl in self.content_loss:
                    content_score += cl.loss

                # apply weights to losses:
                style_score *= style_weight
                content_score *= content_weight

                # total loss:
                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                # uncomment if you'd like to see progress output
                # in the python terminal
                # if run[0] % 50 == 0:
                #     print('Run {} : Style Loss = {:4f} Content Loss = {:4f}'.format(
                #         run[0], style_score.item(), content_score.item()))

                return style_score + content_score

            self.optimizer.step(closure)

        # a last correction...
        self.input_img.data.clamp_(0, 1)

        return self.input_img


async def image_style_transfer(gamma, user_id):
    # set device for model run:
    device = get_device()

    # load content and style images
    content_img_name = user_id + '_content_img.jpg'
    style_img_name = user_id + '_style_img.jpg'
    content_img = img_loader(content_img_name, device)
    style_img = img_loader(style_img_name, device)

    # initialize input image:
    input_img = content_img.clone()

    # built nst cnn:
    nst = NST_Assy(content_img, style_img, input_img, gamma, device)

    # run style transfer:
    output_img = nst.fit()

    # save result:
    save_output_img(output_img, user_id)