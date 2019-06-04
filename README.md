# Pytorch Zoo

[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of useful modules and utilities for kaggle not available in Pytorch

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [Overview](#overview)
-   [Installation](#installation)
-   [Documentation](#documentation)
    -   [Notifications](#notifications)
        -   [Sending yourself notifications when your models finish training](#sending-yourself-notifications-when-your-models-finish-training)
        -   [Viewing training progress with tensorboard in a kaggle kernel](#viewing-training-progress-with-tensorboard-in-a-kaggle-kernel)
    -   [Loss](#loss)
    -   [Metrics](#metrics)
    -   [Modules](#modules)
    -   [Schedulers](#schedulers)
    -   [Utils](#utils)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Overview

## Installation

## Documentation

### Notifications

#### Sending yourself notifications when your models finish training

IFTTT allows you to easily do this. Follow https://medium.com/datadriveninvestor/monitor-progress-of-your-training-remotely-f9404d71b720 to setup a webhook and get a secret key.

Then you can send yourself a notification with:

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

This will start tensorboard, set up a http tunnel on tensorboard's port, and send you a notification with a url where you can access tensorboard.

### Loss

[lovasz_hinge(logits, labels, per_image=True)](./pytorch_zoo/loss.py#L84)

The binary Lovasz Hinge loss for semantic segmentation.

**Arguments**:  
`logits` (torch.tensor): Logits at each pixel (between -\infty and +\infty).  
`labels` (torch.tensor): Binary ground truth masks (0 or 1).  
`per_image` (bool, optional): Compute the loss per image instead of per batch. Defaults to True.

**Shape**:

-   Input:
    -   logits: (batch, height, width)
    -   labels: (batch, height, width)
-   Output: (batch)

**Returns**:  
torch.tensor: The lovasz hinge loss

### Metrics

### Modules

### Schedulers

### Utils

## Contributing

This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features or modules, feel free to open an issue or a pull request

## Authors

-   **Bilal Khan** - _Initial work_

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
