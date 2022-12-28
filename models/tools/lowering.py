import sys
import math
import torch
import numpy as np

from models.tools.layer_info import ConvLayerInfo
from models.tools.progressbar import progressbar


def weight_lowering(weight: torch.Tensor, layer_info: ConvLayerInfo):
    Co, Ci, FW, FH = layer_info.weight_shape()
    return torch.reshape(weight.detach(), shape=(Co, Ci*FW*FH))

def ifm_lowering(ifm: torch.Tensor, layer_info: ConvLayerInfo, verbose: bool=False):
    N, Ci, W, H = ifm.shape
    FW, FH, P, S = layer_info.FW, layer_info.FH, layer_info.P, layer_info.S

    if P > 0:
        ifm = torch.nn.functional.pad(ifm, (P, P, P, P), value=0)  # add zero padding manually

    # variables for verbose
    OW, OH = layer_info.OW, layer_info.OH
    iter_cnt = 0
    total_cnt = N*OW*OH
    if verbose:
        sys.stdout.write(f"\rlowering {progressbar(status=iter_cnt, total=total_cnt, scale=50)} {iter_cnt/total_cnt*100:.0f}%")

    lowered_ifm = []
    for n in range(N):
        for rp in range(0, H - FH + (2 * P) + 1, S):
            for cp in range(0, W - FW + (2 * P) + 1, S):
                lowered_ifm.append(list(ifm[n, :, rp:rp + FH, cp:cp + FW].flatten()))

                if verbose:
                    iter_cnt += 1
                    sys.stdout.write(f"\rlowering {progressbar(status=iter_cnt, total=total_cnt, scale=50)} {iter_cnt/total_cnt*100:.0f}%")

    if verbose:
        sys.stdout.write('\r')

    lowered_ifm = torch.tensor(lowered_ifm)

    return lowered_ifm


if __name__ == '__main__':
    N = 1
    C = 3
    W = 10
    H = 10
    FW = 3
    FH = 3
    S = 2
    P = 0

    t = torch.tensor(torch.arange(0, N*C*W*H, 1)).reshape(N, C, H, W)
    lt = ifm_lowering(t, ConvLayerInfo(N=N, Ci=C, W=W, H=H, Co=3, FW=FW, FH=FH, S=S, P=P), verbose=True)
    for v in lt:
        print(' '.join(map(lambda x: f"{x:4d}", list(v.detach().cpu().numpy()))))