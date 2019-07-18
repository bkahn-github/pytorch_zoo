<h1 align='center'>
    Pytorch Zoo
</h1>

<h4 align='center'>
    A collection of useful modules and utilities for kaggle not available in <a href="https://pytorch.org">Pytorch</a>
</h4>

<p align='center'>
    <a href="https://forthebadge.com">
        <img src="https://forthebadge.com/images/badges/made-with-python.svg" alt="forthebadge">
    </a>
    <a href="https://lgtm.com/projects/g/bkkaggle/pytorch_zoo/context:python">
        <img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/bkkaggle/pytorch_zoo.svg?logo=lgtm&logoWidth=18"/>
    </a>
    <a href="https://lgtm.com/projects/g/bkkaggle/pytorch_zoo/alerts/">
        <img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/bkkaggle/pytorch_zoo.svg?logo=lgtm&logoWidth=18"/>
    </a>
    <a href="https://github.com/prettier/prettier">
        <img src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square" alt="code style: prettier" />
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="http://makeapullrequest.com">
        <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
    </a>
    <a href="https://github.com/bkkaggle/pytorch_zoo/issues">
        <img alt="GitHub issues" src="https://img.shields.io/github/issues/bkkaggle/pytorch_zoo.svg?style=flat">
    </a>
    <a href="https://github.com/bkkaggle/pytorch_zoo/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/bkkaggle/pytorch_zoo.svg">
    </a>
    <a href="https://github.com/bkkaggle/pytorch_zoo/issues">
        <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/bkkaggle/pytorch_zoo.svg">
    </a>

</p>

<p align='center'>
    <a href='#overview'>Overview</a> •
    <a href='#installation'>Installation</a> •
    <a href='#documentation'>Documentation</a> •
    <a href='#contributing'>Contributing</a> •
    <a href='#authors'>Authors</a> •
    <a href='#license'>License</a>
</p>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [Installation](#installation)
-   [Documentation](#documentation)
    -   [Notifications](#notifications)
        -   [Sending yourself notifications when your models finish training](#sending-yourself-notifications-when-your-models-finish-training)
        -   [Viewing training progress with tensorboard in a kaggle kernel](#viewing-training-progress-with-tensorboard-in-a-kaggle-kernel)
    -   [Data](#data)
        -   [DynamicSampler(sampler, batch_size=32)](#dynamicsamplersampler-batch_size32)
        -   [trim_tensors(tensors)](#trim_tensorstensors)
    -   [Loss](#loss)
        -   [lovasz_hinge(logits, labels, per_image=True)](#lovasz_hingelogits-labels-per_imagetrue)
    -   [Metrics](#metrics)
        -   [iou(y_true_in, y_pred_in)](#iouy_true_in-y_pred_in)
    -   [Modules](#modules)
        -   [SqueezeAndExcitation(in_ch, r=16)](#squeezeandexcitationin_ch-r16)
        -   [ChannelSqueezeAndSpatialExcitation(in_ch)](#channelsqueezeandspatialexcitationin_ch)
        -   [ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(in_ch)](#concurrentspatialandchannelsqueezeandchannelexcitationin_ch)
        -   [GaussianNoise(0.1)](#gaussiannoise01)
    -   [Schedulers](#schedulers)
        -   [CyclicalMomentum(optimizer, base_momentum=0.8, max_momentum=0.9, step_size=2000, mode="triangular")](#cyclicalmomentumoptimizer-base_momentum08-max_momentum09-step_size2000-modetriangular)
    -   [Utils](#utils)
        -   [notify({'value1': 'Notification title', 'value2': 'Notification body'}, key)](#notifyvalue1-notification-title-value2-notification-body-key)
        -   [seed_environment(seed=42)](#seed_environmentseed42)
        -   [gpu_usage(device, digits=4)](#gpu_usagedevice-digits4)
        -   [n_params(model)](#n_paramsmodel)
        -   [save_model(model, fold=0)](#save_modelmodel-fold0)
        -   [load_model(model, fold=0)](#load_modelmodel-fold0)
        -   [save(obj, 'obj.pkl')](#saveobj-objpkl)
        -   [load('obj.pkl')](#loadobjpkl)
        -   [masked_softmax(logits, mask, dim=-1)](#masked_softmaxlogits-mask-dim-1)
        -   [masked_log_softmax(logits, mask, dim=-1)](#masked_log_softmaxlogits-mask-dim-1)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Installation

pytorch_zoo can be installed from pip

```
pip install pytorch_zoo
```

or if using a kaggle kernel without an internet connection, add the pytorch_zoo [dataset](https://www.kaggle.com/bkkaggle/pytorch-zoo) and run

```python
!pip install ../input/repository/bkkaggle-pytorch_zoo-0d7aa7b/
```

## Documentation

### Notifications

#### Sending yourself notifications when your models finish training

IFTTT allows you to easily do this. Follow https://medium.com/datadriveninvestor/monitor-progress-of-your-training-remotely-f9404d71b720 to setup an IFTTT webhook and get a secret key.

Once you have a key, you can send yourself a notification with:

```python
from pytorch_zoo.utils import notify

message = f'Validation loss: {val_loss}'
obj = {'value1': 'Training Finished', 'value2': message}

notify(obj, [YOUR_SECRET_KEY_HERE])
```

#### Viewing training progress with tensorboard in a kaggle kernel

Make sure tensorboard is installed in the kernel and run the following in a code cell near the beginning of your kernel:

```python
!mkdir logs
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -o ngrok-stable-linux-amd64.zip
LOG_DIR = './logs'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')

!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

temp = !curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

from pytorch_zoo.utils import notify

obj = {'value1': 'Tensorboard URL', 'value2': temp[0]}
notify(obj, [YOUR_SECRET_KEY_HERE])

!rm ngrok
!rm ngrok-stable-linux-amd64.zip
```

This will start tensorboard, set up a http tunnel, and send you a notification with a url where you can access tensorboard.

### Data

##### [DynamicSampler(sampler, batch_size=32)](./pytorch_zoo/data.py#L4)

A dynamic batch length data sampler. To be used with `trim_tensors`.

Implementation adapted from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779 and https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py

```python
train_dataset = data.TensorDataset(data)
sampler = data.RandomSampler(train_dataset)
sampler = DynamicSampler(sampler, batch_size=32, drop_last=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=len_sampler)

for epoch in range(10):
    for batch in train_loader:
        batch = trim_tensors(batch)
        train_batch(...)
```

_Arguments_:  
`sampler` (torch.utils.data.Sampler): Base sampler.  
`batch_size` (int): Size of minibatch.  
`drop_last` (bool): If `True`, the sampler will drop the last batch if its size would be less than `batch_size`.

##### [trim_tensors(tensors)](./pytorch_zoo/data.py#L48)

Trim padding off of a batch of tensors to the smallest possible length. To be used with `DynamicSampler`.

Implementation adapted from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779

```python
train_dataset = data.TensorDataset(data)
sampler = data.RandomSampler(train_dataset)
sampler = DynamicSampler(sampler, batch_size=32, drop_last=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=len_sampler)

for epoch in range(10):
    for batch in train_loader:
        batch = trim_tensors(batch)
        train_batch(...)
```

_Arguments_:  
`tensors` ([torch.tensor]): list of tensors to trim.

_Returns_:  
([torch.tensor]): list of trimmed tensors.

### Loss

##### [lovasz_hinge(logits, labels, per_image=True)](./pytorch_zoo/loss.py#L84)

The binary Lovasz Hinge loss for semantic segmentation.

Implementation adapted from https://github.com/bermanmaxim/LovaszSoftmax

```python
loss = lovasz_hinge(logits, labels)
```

_Arguments_:  
`logits` (torch.tensor): Logits at each pixel (between -\infty and +\infty).  
`labels` (torch.tensor): Binary ground truth masks (0 or 1).  
`per_image` (bool, optional): Compute the loss per image instead of per batch. Defaults to True.

_Shape_:

-   Input:
    -   logits: (batch, height, width)
    -   labels: (batch, height, width)
-   Output: (batch)

_Returns_:  
(torch.tensor): The lovasz hinge loss

### Metrics

##### [iou(y_true_in, y_pred_in)](./pytorch_zoo/metrics.py#L64)

Calculates the average IOU (intersection over union) score on thresholds from 0.5 to 0.95 with a step size of 0.05.

Implementation adapted from: https://www.kaggle.com/aglotero/another-iou-metric

```python
val_iou = iou(y_val, val_preds)
```

_Arguments_:  
`y_true_in` (numpy array): Ground truth labels.  
`y_pred_in` (numpy array): Predictions from model.

_Returns_:  
(int): Averaged IOU score over predictions.

### Modules

##### [SqueezeAndExcitation(in_ch, r=16)](./pytorch_zoo/modules.py#L6)

The channel-wise SE (Squeeze and Excitation) block from the [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) paper.

Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939 and https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

```python
# in __init__()
self.SE = SqueezeAndExcitation(in_ch, r=16)

# in forward()
x = self.SE(x)
```

_Arguments_:  
`in_ch` (int): The number of channels in the feature map of the input.  
`r` (int): The reduction ratio of the intermidiate channels. Default: 16.

_Shape_:

-   Input: (batch, channels, height, width)
-   Output: (batch, channels, height, width) (same shape as input)

##### [ChannelSqueezeAndSpatialExcitation(in_ch)](./pytorch_zoo/modules.py#L41)

The sSE (Channel Squeeze and Spatial Excitation) block from the [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

```python
# in __init__()
self.sSE = ChannelSqueezeAndSpatialExcitation(in_ch)

# in forward()
x = self.sSE(x)
```

_Arguments_:  
`in_ch` (int): The number of channels in the feature map of the input.

_Shape_:

-   Input: (batch, channels, height, width)
-   Output: (batch, channels, height, width) (same shape as input)

##### [ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(in_ch)](./pytorch_zoo/modules.py#L71)

The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation) block from the [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

```python
# in __init__()
self.scSE = ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(in_ch, r=16)

# in forward()
x = self.scSE(x)
```

_Arguments_:  
`in_ch` (int): The number of channels in the feature map of the input.  
`r` (int): The reduction ratio of the intermidiate channels. Default: 16.

_Shape_:

-   Input: (batch, channels, height, width)
-   Output: (batch, channels, height, width) (same shape as input)

##### [GaussianNoise(0.1)](./pytorch_zoo/modules.py#L104)

A gaussian noise module.

```python
# in __init__()
self.gaussian_noise = GaussianNoise(0.1)

# in forward()
if self.training:
    x = self.gaussian_noise(x)
```

_Arguments_:  
`stddev` (float): The standard deviation of the normal distribution. Default: 0.1.

_Shape_:

-   Input: (batch, \*)
-   Output: (batch, \*) (same shape as input)

### Schedulers

##### [CyclicalMomentum(optimizer, base_momentum=0.8, max_momentum=0.9, step_size=2000, mode="triangular")](./pytorch_zoo/schedulers.py#L7)

Pytorch's [cyclical learning rates](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py), but for momentum, which leads to better results when used with cyclic learning rates, as shown in [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820).

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.CyclicMomentum(optimizer)
data_loader = torch.utils.data.DataLoader(...)
for epoch in range(10):
    for batch in data_loader:
        scheduler.batch_step()
        train_batch(...)
```

_Arguments_:  
`optimizer` (Optimizer): Wrapped optimizer.  
`base_momentum` (float or list): Initial momentum which is the lower boundary in the cycle for each param groups. Default: 0.8  
`max_momentum` (float or list): Upper boundaries in the cycle for each parameter group. scaling function. Default: 0.9  
`step_size` (int): Number of training iterations per half cycle. Authors suggest setting step_size 2-8 x training iterations in epoch. Default: 2000  
`mode` (str): One of {triangular, triangular2, exp_range}. Default: 'triangular'  
`gamma` (float): Constant in 'exp_range' scaling function. Default: 1.0  
`scale_fn` (function): Custom scaling policy defined by a single argument lambda function. Mode paramater is ignored Default: None  
`scale_mode` (str): {'cycle', 'iterations'}. Defines whether scale_fn is evaluated on cycle number or cycle iterations (training iterations since start of cycle). Default: 'cycle'  
`last_batch_iteration` (int): The index of the last batch. Default: -1

### Utils

##### [notify({'value1': 'Notification title', 'value2': 'Notification body'}, key)](./pytorch_zoo/utils.py#L13)

Send a notification to your phone with IFTTT

Setup a IFTTT webhook with https://medium.com/datadriveninvestor/monitor-progress-of-your-training-remotely-f9404d71b720

```python
notify({'value1': 'Notification title', 'value2': 'Notification body'}, key=[YOUR_PRIVATE_KEY_HERE])
```

_Arguments_:  
`obj` (Object): Object to send to IFTTT  
`key` ([type]): IFTTT webhook key

##### [seed_environment(seed=42)](./pytorch_zoo/utils.py#L25)

Set random seeds for python, numpy, and pytorch to ensure reproducible research.

```python
seed_envirionment(42)
```

_Arguments_:  
`seed` (int): The random seed to set.

##### [gpu_usage(device, digits=4)](./pytorch_zoo/utils.py#L39)

Prints the amount of GPU memory currently allocated in GB.

```python
gpu_usage(device, digits=4)
```

_Arguments_:  
`device` (torch.device, optional): The device you want to check. Defaults to device.  
`digits` (int, optional): The number of digits of precision. Defaults to 4.

##### [n_params(model)](./pytorch_zoo/utils.py#L53)

Return the number of parameters in a pytorch model.

```python
print(n_params(model))
```

_Arguments_:  
`model` (nn.Module): The model to analyze.

_Returns_:  
(int): The number of parameters in the model.

##### [save_model(model, fold=0)](./pytorch_zoo/utils.py#L71)

Save a trained pytorch model on a particular cross-validation fold to disk.

Implementation adapted from https://github.com/floydhub/save-and-resume.

```python
save_model(model, fold=0)
```

_Arguments_:  
`model` (nn.Module): The model to save.  
`fold` (int): The cross-validation fold the model was trained on.

##### [load_model(model, fold=0)](./pytorch_zoo/utils.py#L84)

Load a trained pytorch model saved to disk using `save_model`.

```python
model = load_model(model, fold=0)
```

_Arguments_:
`model` (nn.Module): The model to save.  
`fold` (int): Which saved model fold to load.

_Returns_:  
(nn.Module): The same model that was passed in, but with the pretrained weights loaded.

##### [save(obj, 'obj.pkl')](./pytorch_zoo/utils.py#L99)

Save an object to disk.

```python
save(tokenizer, 'tokenizer.pkl')
```

_Arguments_:  
`obj` (Object): The object to save.  
`filename` (String): The name of the file to save the object to.

##### [load('obj.pkl')](./pytorch_zoo/utils.py#L110)

Load an object saved to disk with `save`.

```python
tokenizer = load('tokenizer.pkl')
```

_Arguments_:  
`path` (String): The path to the saved object.

_Returns_:  
(Object): The loaded object.

##### [masked_softmax(logits, mask, dim=-1)](./pytorch_zoo/utils.py#L124)

A masked softmax module to correctly implement attention in Pytorch.

Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

```python
out = masked_softmax(logits, mask, dim=-1)
```

_Arguments_:  
`vector` (torch.tensor): The tensor to softmax.  
`mask` (torch.tensor): The tensor to indicate which indices are to be masked and not included in the softmax operation.  
`dim` (int, optional): The dimension to softmax over. Defaults to -1.  
`memory_efficient` (bool, optional): Whether to use a less precise, but more memory efficient implementation of masked softmax. Defaults to False.  
`mask_fill_value` ([type], optional): The value to fill masked values with if `memory_efficient` is `True`. Defaults to -1e32.

_Returns_:  
(torch.tensor): The masked softmaxed output

##### [masked_log_softmax(logits, mask, dim=-1)](./pytorch_zoo/utils.py#L175)

A masked log-softmax module to correctly implement attention in Pytorch.

Implementation adapted from: https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

```python
out = masked_log_softmax(logits, mask, dim=-1)
```

_Arguments_:  
`vector` (torch.tensor): The tensor to log-softmax.  
`mask` (torch.tensor): The tensor to indicate which indices are to be masked and not included in the log-softmax operation.  
`dim` (int, optional): The dimension to log-softmax over. Defaults to -1.

_Returns_:  
(torch.tensor): The masked log-softmaxed output

## Contributing

This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features or modules, feel free to open an issue or a pull request. Feel free to use the library or code from it in your own projects, and if you feel that some code used in this project hasn't been properly accredited, please open an issue.

## Authors

-   _Bilal Khan_ - _Initial work_

## License

This project is licensed under the MIT License - see the [license](LICENSE) file for details

## Acknowledgements

This project contains code adapted from:

-   https://github.com/bermanmaxim/LovaszSoftmax
-   https://www.kaggle.com/aglotero/another-iou-metric
-   https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
-   https://github.com/bckenstler/CLR
-   https://github.com/floydhub/save-and-resume
-   https://github.com/allenai/allennlp
-   https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779

This README is based on:

-   https://github.com/mxbi/mlcrate
-   https://github.com/athityakumar/colorls
-   https://github.com/amitmerchant1990/electron-markdownify/blob/master/README.md

```

```
