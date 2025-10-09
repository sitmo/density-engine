import torch

def get_best_device(verbose=False):
    if torch.cuda.is_available():
        if verbose:
            print('device is CUDA! :-)')
        return torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        if verbose: 
            print('device is MPS :-/')
        return torch.device('mps')
    else:
        if verbose:
            print('device is CPU :-|')
        return torch.device('cpu')
