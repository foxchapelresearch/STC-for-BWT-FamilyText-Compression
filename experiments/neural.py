"""Online (prequential) byte-level LSTM compressor, nncp-style.

The file is split into B contiguous streams coded in parallel. Each step predicts the next CHUNK bytes of
every stream with the current weights (that prediction is the code length a decoder could reproduce),
then takes one Adam step on those same bytes. Nothing is pre-trained or transmitted.

usage: python3 -I neural.py <file> <maxbytes> <out_prefix>
"""
import sys, math, time
import numpy as np
import torch, torch.nn as nn

path, maxbytes, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
data = np.frombuffer(open(path, 'rb').read()[:maxbytes], dtype=np.uint8)
B, CHUNK, EMB, HID, LAYERS = 32, 32, 64, 512, 2
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(0)

seg = len(data) // B
streams = torch.tensor(data[:seg * B].reshape(B, seg).astype(np.int64))
digit = (streams >= 48) & (streams <= 57)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, EMB)
        self.rnn = nn.LSTM(EMB, HID, LAYERS, batch_first=True)
        self.out = nn.Linear(HID, 256)
    def forward(self, x, h):
        y, h = self.rnn(self.emb(x), h)
        return self.out(y), h

m = Model().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
h = None
total_bits = 0.0; digit_bits = 0.0
prev = torch.zeros(B, 1, dtype=torch.long)  # a fixed "start" symbol (byte 0) before each stream
t0 = time.time()
steps = (seg + CHUNK - 1) // CHUNK
for s in range(steps):
    lo, hi = s * CHUNK, min(seg, (s + 1) * CHUNK)
    tgt = streams[:, lo:hi]
    inp = torch.cat([prev, tgt[:, :-1]], dim=1)
    prev = tgt[:, -1:]
    logits, h = m(inp.to(dev), h)
    nll = nn.functional.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1).to(dev), reduction='none')
    loss = nll.mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    h = tuple(t.detach() for t in h)
    bits = nll.detach().cpu() / math.log(2)
    total_bits += bits.sum().item()
    digit_bits += bits[digit[:, lo:hi].reshape(-1)].sum().item()
    if s % 500 == 0:
        print(f"step {s}/{steps}  {total_bits / 8 / ((s + 1) * CHUNK * B) * 8:.4f} bpc so far  {time.time() - t0:.0f}s", flush=True)
n = seg * B
print(f"bytes={n} total={total_bits / 8:.0f} bytes ({total_bits / n:.4f} bpc)  digits={digit_bits / 8:.0f}  time={time.time() - t0:.0f}s")
np.save(out + '_emb_in.npy', m.emb.weight.detach().cpu().numpy())
np.save(out + '_emb_out.npy', m.out.weight.detach().cpu().numpy())
