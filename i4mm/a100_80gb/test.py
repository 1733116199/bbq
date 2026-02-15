import torch
import i4mm
import numpy as np

x = torch.randint(-8, 8, (640, 640)).int().cuda()
w = torch.randint(-8, 8, (640, 640)).int().cuda()

def encode(input: torch.Tensor):
    temp = input.clone()
    temp = temp.reshape((np.prod(input.shape[:-1]).item(), input.shape[-1]))
    neg = temp < 0
    temp[neg] = 16 + temp[neg]
    shift = torch.Tensor([0, 4]).int().cuda()
    res = torch.sum(
        torch.reshape(temp, (temp.shape[0], temp.shape[1] // 2, 2)) << shift[None, None, :],
        dim=-1
    )
    res = torch.reshape(res, (*input.shape[:-1], input.shape[-1] // 2))
    return res.to(torch.uint8)

exp = torch.matmul(x.cpu(), w.cpu().T).cuda()
act = i4mm.run(encode(x), encode(w), None)
assert torch.all(act == exp)
print("PASSED")