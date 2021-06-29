import torch
from torch import autograd
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax()

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)
        # NLLLoss(x, class) = -weights[class] * x[class]
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
    