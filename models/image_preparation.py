




def img_loader(img, device):
    """
    This function loads and pre-porcesses images for
    style transfer needs.
    """
    PIL_img = Image.open(img)
    max_dim = max(PIL_img.size)
    max_allow_dim = 256
    if max_dim > max_allow_dim:
        scale_ratio = max_allow_dim / max_dim
        loader = transforms.Compose([transforms.Resize(
            (int(PIL_img.size[1] * scale_ratio),
             int(PIL_img.size[0] * scale_ratio))),
            transforms.ToTensor()])
    else:
        loader = transforms.ToTensor()
    tensor_img = loader(PIL_img).unsqueeze(0).to(device)
    return tensor_img



class Normalization(nn.Module):
    """
    This class represent normalization layer for VGG head
    which has been trained on IMAGENET.
    """

    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
