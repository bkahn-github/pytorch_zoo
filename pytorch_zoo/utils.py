import os
import random
import pickle
import requests

import numpy as np

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def notify(obj, key):
    """Send a notification to your phone with IFTTT

    Setup a IFTTT webhook with https://medium.com/datadriveninvestor/monitor-progress-of-your-training-remotely-f9404d71b720
    
    Args:
        obj (Object): Object to send to IFTTT
        key ([type]): IFTTT webhook key
    """
    requests.post(f"https://maker.ifttt.com/trigger/notify/with/key/{key}", data=obj)


def seed_environment(seed):
    """Set random seeds for python, numpy, and pytorch to ensure reproducible research.
    
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def gpu_usage(device=device, digits=4):
    """Prints the amount of GPU memory currently allocated in GB.
    
    Args:
        device (torch.device, optional): The device you want to check.
                                        Defaults to device.
        digits (int, optional): The number of digits of precision.
                                Defaults to 4.
    """
    print(
        f"GPU Usage: {round((torch.cuda.memory_allocated(device=device) / 1e9), digits)} GB\n"
    )


def n_params(model):
    """Return the number of parameters in a pytorch model.
    
    Args:
        model (nn.Module): The model to analyze.
    
    Returns:
        int: The number of parameters in the model.
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def save_model(model, fold):
    """Save a trained pytorch model on a particular cross-validation fold to disk. 

    Implementation adapted from https://github.com/floydhub/save-and-resume.

    Args:
        model (nn.Module): The model to save.
        fold (int): The cross-validation fold the model was trained on.
    """
    filename = f"./checkpoint-{fold}.pt"
    torch.save(model, filename)


def load_model(model, fold):
    """Load a trained pytorch model saved to disk using `save_model`.
    
    Args:
        model (nn.Module): The model to save.
        fold (int): Which saved model fold to load.
    
    Returns:
        nn.Module: The same model that was passed in, but with the pretrained weights loaded.
    """
    model.load_state_dict(torch.load(f"./checkpoint-{fold}.pt"))

    return model


def save(obj, filename):
    """Save an object to disk.
    
    Args:
        obj (Object): The object to save.
        filename (String): The name of the file to save the object to.
    """
    with open(f"{filename}", "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path):
    """Load an object saved to disk with `save`.
    
    Args:
        path (String): The path to the saved object.
    
    Returns:
        Object: The loaded object.
    """
    with open(path, "rb") as handle:
        obj = pickle.load(handle)

    return obj

def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    """A masked softmax module to correctly implement attention in Pytorch.

    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    
    Args:
        vector (torch.tensor): The tensor to softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the softmax operation.
        dim (int, optional): The dimension to softmax over.
                            Defaults to -1.
        memory_efficient (bool, optional): Whether to use a less precise, but more memory efficient implementation of masked softmax.
                                            Defaults to False.
        mask_fill_value ([type], optional): The value to fill masked values with if `memory_efficient` is `True`.
                                            Defaults to -1e32.
    
    Returns:
        torch.tensor: The masked softmaxed output
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


# From: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L276-L307
def masked_log_softmax(vector, mask, dim=-1):
    """A masked log-softmax module to correctly implement attention in Pytorch.

    Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.

    Args:
        vector (torch.tensor): The tensor to log-softmax.
        mask (torch.tensor): The tensor to indicate which indices are to be masked and not included in the log-softmax operation.
        dim (int, optional): The dimension to log-softmax over.
                            Defaults to -1.
    
    Returns:
        torch.tensor: The masked log-softmaxed output

    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)
