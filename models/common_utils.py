import torch

def gram_matrix(input):
    """
    This function computes gramm matrix which
    is used to determine style loss.
    """
    batch_size, ch, h, w = input.shape
    features = input.view(batch_size * ch, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * ch * h * w)
