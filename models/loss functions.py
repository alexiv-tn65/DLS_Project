import torch.nn as nn



class ContentLoss(nn.Module):
    """
    Content loss layer
    """

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        if input.shape == self.target.shape:
            self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """
    Style Loss layer
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input