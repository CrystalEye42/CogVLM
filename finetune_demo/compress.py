import torch
import os
import sys

ckpt = sys.argv[1]
new_loc = ckpt
if len(sys.argv) > 1:
    new_loc = os.path.join(os.path.dirname(ckpt), sys.argv[2])

data = torch.load(ckpt, map_location='cpu')
for k in list(data.keys()):
    print(k)
    if k != 'module':
        del data[k]
torch.save(data, new_loc)

