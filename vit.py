from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor, nn


class ViT(nn.Module):
    def __init__(self, level):
        super(ViT, self).__init__()
        self.level = level
        if 'dinov2' in level:
            self.model = torch.hub.load('facebookresearch/dinov2', level)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', level)
        self.embed_dim = self.model.embed_dim

    def forward(self, x):
        with torch.no_grad():
            if 'dinov2' in self.level:
                x = self.model.prepare_tokens_with_masks(x)
            else:
                x = self.model.prepare_tokens(x)
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
            if 'dinov2' in self.level and 'reg' in self.level:
                x = x[:, 1 + self.model.num_register_tokens:, :]
            else:
                x = x[:, 1:, :]
        return x
