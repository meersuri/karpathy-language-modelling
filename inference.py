import sys
import textwrap

import torch

from bigram_lm import BigramLM
from dataset import Dataset

def repl_loop(model, ds, max_tokens=100):
    running = True
    try:
        while running:
            print('>>> ', end='')
            prime_text = input()
            if any(c for c in prime_text if c not in ds.vocab):
                print('Out of vocab character enter, try again')
                continue
            tokens = torch.tensor(ds.encode(prime_text)).long()
            tokens = tokens[None, :]
            tokens.to(model.device)
            gen = model.generate(tokens, 256, max_tokens=max_tokens)[0].tolist()
            gen = ds.decode(gen)
            print(textwrap.indent(gen, prefix='\t'))
    except (KeyboardInterrupt, EOFError) as e:
        return

@torch.no_grad()
def eval_loss(model, ds, block_size):
    print(model.calc_loss(ds.val, block_size))

def main():
    ds = Dataset('shakespeare.txt')
    vocab_size = len(ds.vocab)
    block_size = 256
    embed_dim = 384
    layers = 6
    attn_heads = 6

    ckpt = torch.load(sys.argv[1])
    model = BigramLM(
            vocab_size=vocab_size,
            layers=layers,
            block_size=block_size,
            embed_dim=embed_dim,
            attn_heads=attn_heads,
            )
    model.load_state_dict(ckpt['model_state_dict'])
    model.device = 'cpu'
    model.eval()
    model.to(model.device)
    repl_loop(model, ds, 300)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Expected 1 argument - path to saved model/checkpoint')
        sys.exit(1)
    main()
