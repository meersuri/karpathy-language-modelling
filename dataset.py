import torch

class Dataset:
    def __init__(self, data_path, train_val_split = 0.8):
        self.train_val_split = train_val_split
        self.raw_data = self.load_data(data_path)
        self.vocab = sorted(list(set(self.raw_data)))
        self.token2idx, self.idx2token = self.build_token_mappings(self.vocab)
        self.train, self.val = self.split_train_val(self.raw_data)

    def load_data(self, path):
        with open(path) as f:
            raw_data = f.read()
        return raw_data

    def build_token_mappings(self, vocab):
        token2idx = {c: i for i, c in enumerate(vocab)}
        idx2token = {i: c for i, c in enumerate(vocab)}
        return token2idx, idx2token

    def split_train_val(self, raw_data):
        n = len(raw_data)
        train = self.encode(raw_data[:int(self.train_val_split*n)])
        val = self.encode(raw_data[int(self.train_val_split*n):])
        return torch.tensor(train), torch.tensor(val)

    def encode(self, s):
        return [self.token2idx[c] for c in s]

    def decode(self, ints):
        return ''.join([self.idx2token[i] for i in ints])

    def get_batch(self, data, block_size, batch_size):
        starts = torch.randint(0, len(data) - block_size - 1, [batch_size])
        x = torch.stack([data[idx: idx + block_size] for idx in starts])
        y = torch.stack([data[idx + 1: idx + 1 + block_size] for idx in starts]) 
        return x, y


if __name__ == '__main__':
    data_path = 'shakespeare.txt'
    ds = Dataset(data_path)
    print(ds.encode('hello world'))
    print(ds.decode(t.ds.encode('hello world')))
    print(ds.train.shape, t.ds.train.dtype)
    print(ds.train[:8 + 1])
    x, y = t.ds.get_batch(t.ds.train, 8, 4)
    print(x)
    print(y)
