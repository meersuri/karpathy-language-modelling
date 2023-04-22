from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from dataset import Dataset

class HyperParams:
    def __init__(self, lr=1e-3):
        self.lr = lr

class BaseLM(torch.nn.Module):
    def __init__(self, dataset, hparams):
        super().__init__()
        self.ds = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate(self, batch, block_size, max_tokens=100):
        out = torch.clone(batch)
        for i in range(max_tokens):
            batch = batch[:, -block_size:]
            logits, loss = self(batch)
            probs = torch.softmax(logits[:, -1, :], dim=1)
            samples = torch.multinomial(probs, num_samples=1)
            batch = torch.cat([batch, samples], dim=1)
            out = torch.cat([out, samples], dim=1)
        return out

    def train(self, opt, steps=1000, block_size=8, batch_size=32):
        for i in tqdm(range(steps)):
            x, y = self.ds.get_batch(ds.train, block_size, batch_size)
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            logits, loss = self.forward(x, y)
            loss.backward()
            opt.step()

    def calc_loss(self, ds, block_size, batch_size, split='train'):
        with torch.no_grad():
            if split == 'train':
                data = ds.train
            else:
                data = ds.val
            n = 100
            loss = 0
            for i in range(n):
                x, y = ds.get_batch(data, block_size, batch_size)
                x, y = x.to(self.device), y.to(self.device)
                _, batch_loss = self(x, targets=y)
                loss += batch_loss
            loss /= n
            return loss


class BigramLM(BaseLM):
    def __init__(self, dataset, block_size, embed_dim, attn_heads, key_dim, value_dim, hparams):
        super().__init__(dataset, hparams)
        vocab_size = len(dataset.vocab)
        self.token_lut = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos_lut = torch.nn.Embedding(block_size, embed_dim)
        self.attn = MultiHeadAttention(attn_heads, embed_dim, key_dim, value_dim)
        self.ffwd = FeedForward(embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        token = self.token_lut(x)
        pos = self.pos_lut(torch.arange(x.shape[1], device=self.device))
        x = token + pos
        logits = self.out_proj(self.ffwd(self.attn(x)))
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss



class BigramLM2(BaseLM):
    def __init__(self, dataset, hparams):
        super().__init__(dataset, hparams)
        vocab_size = len(dataset.vocab)
        self.linear1 = torch.nn.Linear(1, vocab_size)

    def forward(self, x, targets=None):
        x = torch.unsqueeze(x, -1).float()
        logits = self.linear1(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss


class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, masked=True, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.masked = masked
        self.key_dim = key_dim
        self.query_proj = torch.nn.Linear(input_dim, key_dim)
        self.key_proj = torch.nn.Linear(input_dim, key_dim)
        self.value_proj = torch.nn.Linear(input_dim, value_dim)
        self.out_proj = torch.nn.Linear(value_dim, output_dim)

    def forward(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = (1/np.sqrt(self.key_dim))*torch.matmul(q, k.transpose(-2, -1))
        if self.masked:
            scores = torch.tril(scores)
            scores[scores==0.0] = -torch.inf
        wts = torch.softmax(scores, dim=2)
        y = torch.matmul(wts, v)
        y = self.out_proj(y)
        return y


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, head_count, input_dim, key_dim, value_dim, masked=True):
        super().__init__()
        head_size = input_dim//head_count
        self.sa = torch.nn.ModuleList([
            SelfAttention(input_dim, key_dim, value_dim, output_dim=head_size) for i in range(head_count - 1)
        ])
        head_size += (input_dim - head_count*head_size)
        self.sa.append(SelfAttention(input_dim, key_dim, value_dim, output_dim=head_size))

    def forward(self, x):
        sa_outs = [sa(x) for sa in self.sa]
        out = torch.cat([x for x in sa_outs], dim=-1)
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return F.relu(self.proj(x))


if __name__ == '__main__':
    data_path = 'shakespeare.txt'
    ds = Dataset(data_path)
    hparams = HyperParams(lr=1e-3)
    block_size = 8 # number of tokens in the sequence
    embed_dim = 70 # token idx -> vector of embed_dim
    batch_size = 32
    attn_heads = 8
    key_dim, value_dim = 8, 8
    lm = BigramLM(ds, block_size, embed_dim, attn_heads, key_dim, value_dim, hparams)
    lm.to(lm.device)
    x, y = ds.get_batch(ds.train, block_size, batch_size)
    x, y = x.to(lm.device), y.to(lm.device)
    logits, _ = lm.forward(x, y)
    loss = lm.calc_loss(ds, block_size, batch_size, 'val')
    print(f'init val loss: {loss}')
    inp = torch.tensor([[0]]).to(lm.device)
    print(ds.decode(lm.generate(inp, block_size)[0].tolist()))
    opt = torch.optim.AdamW(lm.parameters(), lr=hparams.lr)
    steps = 1000
    lm.train(opt, steps=steps)
    print(ds.decode(lm.generate(inp, block_size, max_tokens=1000)[0].tolist()))
    loss = lm.calc_loss(ds, block_size, batch_size, 'val')
    print(f'val loss: {loss}')
    time_str = datetime.strftime(datetime.now(), '%Y%d%m%H%M%S')
    torch.save( {
        'steps': steps,
        'model_state_dict': lm.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'val_loss': loss,
    }, f'model_{time_str}.ckpt')
