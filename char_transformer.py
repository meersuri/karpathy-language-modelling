import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

class Transformer:
    def __init__(self, dataset, block_size=8, batch_size=4, seed=1994):
        self.ds = dataset
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(seed)


if __name__ == '__main__':
    data_path = 'shakespeare.txt'
    ds = Dataset(data_path)
    t = Transformer(ds)
    print(t.ds.encode('hello world'))
    print(t.ds.decode(t.ds.encode('hello world')))
    print(t.ds.train.shape, t.ds.train.dtype)
    print(t.ds.train[:8 + 1])
    x, y = t.ds.get_batch(t.ds.train, 8, 4)
    print(x)
    print(y)
    hparams = HyperParams(lr=1e-3)
    lm = BigramLM(ds, len(ds.vocab), hparams)
    lm.to(lm.device)
    x, y = x.to(lm.device), y.to(lm.device)
    logits, loss = lm.forward(x, y)
    print(logits)
    print(loss)
    inp = torch.tensor([[0]]).to(lm.device)
    print(ds.decode(lm.generate(inp)[0].tolist()))
    opt = torch.optim.AdamW(lm.parameters(), lr=hparams.lr)
    lm.train(opt, steps=10000)
    print(ds.decode(lm.generate(inp, max_tokens=500)[0].tolist()))
