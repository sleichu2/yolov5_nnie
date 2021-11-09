import torch
import struct
from utils.torch_utils import select_device

# Initialize
device = select_device('cpu')
model = torch.load('runs/exp8/sku11e84v5l.pt', map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

f = open('weights/sku11e84v5l.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
