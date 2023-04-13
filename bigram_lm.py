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

    def generate(self, batch, max_tokens=100):
        for i in range(max_tokens):
            logits, loss = self(batch)
            probs = torch.softmax(logits[:, -1, :], dim=1)
            samples = torch.multinomial(probs, num_samples=1)
            batch = torch.cat([batch, samples], dim=1)
        return batch

    def train(self, opt, steps=1000, block_size=8, batch_size=32):
        for i in tqdm(range(steps)):
            x, y = self.ds.get_batch(ds.train, block_size, batch_size)
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            logits, loss = self.forward(x, y)
            loss.backward()
            opt.step()
        print(loss)


class BigramLM(BaseLM):
    def __init__(self, dataset, hparams):
        super().__init__(dataset, hparams)
        vocab_size = len(dataset.vocab)
        self.token_lut = torch.nn.Embedding(vocab_size, vocab_size)
        self.attn = MultiHeadAttention(2, vocab_size, 20, 30)

    def forward(self, x, targets=None):
        logits = self.attn(self.token_lut(x))
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


if __name__ == '__main__':
    data_path = 'shakespeare.txt'
    ds = Dataset(data_path)
    hparams = HyperParams(lr=1e-3)
    lm = BigramLM(ds, hparams)
    lm.to(lm.device)
    x, y = ds.get_batch(ds.train, 8, 1)
    x, y = x.to(lm.device), y.to(lm.device)
    logits, loss = lm.forward(x, y)
    print(loss)
    inp = torch.tensor([[0]]).to(lm.device)
    print(ds.decode(lm.generate(inp)[0].tolist()))
    opt = torch.optim.AdamW(lm.parameters(), lr=hparams.lr)
    lm.train(opt, steps=10000)
    print(ds.decode(lm.generate(inp, max_tokens=500)[0].tolist()))
