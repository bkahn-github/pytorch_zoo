import torch
from torch.utils import data

class DynamicSampler(data.BatchSampler):
    """A dynamic batch length data sampler. To be used with `trim_tensors`.

    Implementation adapted from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779 and https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py

    Args:
        sampler (torch.utils.data.Sampler): Base sampler.
        batch_size (int): Size of minibatch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``.
    """
    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64) 
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, "produced an inccorect number of batches. expected %i, but yielded %i" %(len(self), yielded)

def trim_tensors(tensors):
    """Trim padding off of a batch of tensors to the smallest possible length. To be used with `DynamicSampler`.

    Implementation adapted from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779

    Args:
        tensors ([torch.tensor]): list of tensors to trim.

    Returns:
        ([torch.tensor]): list of trimmed tensors. 
    """

    max_len = torch.max(torch.sum( (tensors[0] != 0  ), 1))
    if max_len > 2: 
        tensors = [tsr[:, :max_len] for tsr in tensors]
    return tensors