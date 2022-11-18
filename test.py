import torch
import torch.nn as nn


x = torch.rand(2, 5)

bkb = nn.Linear(5,3)
bkb_fc = nn.Linear(3,10)

fr = nn.Linear(3,3)
fr_fc = nn.Linear(3,10)





_x1 = bkb(x)

_x2 = _x1.clone()
_x1 = _x1.detach()

pred1 = bkb_fc(_x1)
loss1 = pred1 - torch.zeros(pred1.shape)

pred2 = fr_fc(fr(_x2))
loss2 = pred2 - torch.zeros(pred2.shape)

loss = (loss1 + loss2).sum()
loss.backward()

print('hi')