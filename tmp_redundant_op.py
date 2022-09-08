import os
import math
import numpy as np

W = 28
H = 28
FW = 5
FH = 5
P = 0
S = 3
OW = math.floor((W+2*P-FW)/S) + 1
OH = math.floor((H+2*P-FH)/S) + 1

if_map = np.arange(0, W*H, 1).reshape((W, H))


# Lowering IFM
lowered_if_map = []
for cp in range(0, W-FW+1, S):
    for rp in range(0, H-FH+1, S):
        lowered_if_map.append(list(if_map[cp:cp+FW, rp:rp+FH].flatten()))

lowered_if_map = np.array(lowered_if_map)


# Exception Test
result = {
    'pattern matched': 0,
    'stride exception': 0,
    'out of index exception': 0,
    'unknown exception': 0,
}
unknown_logs = []

for i1 in range(0, FW*FH, 1):
    for i2 in range(i1+1, FW*FH, 1):
        d = i2 - i1
        dv = math.floor(i2/FW) - math.floor(i1/FW)
        dh = (i2 % FW) - (i1 % FW)
        dr = (OW * dv + dh) // S

        for lidx, line in enumerate(lowered_if_map):
            # stride exception
            if dv % S != 0 or dh % S != 0:
                result['stride exception'] += 1
                continue

            # out of index exception
            oh = lidx % OW
            if oh - (dh // S) < 0 or oh - (dh // S) >= OW or lidx - dr < 0:
                result['out of index exception'] += 1
                continue

            # unknown exception
            if line[i1] != lowered_if_map[lidx-dr][i2]:
                unknown_logs.append(f"lidx: {lidx:3d}  oh: {oh:2d}  (d, dh, dv, dr): {d, dh, dv, dr}  (e[i1], e[i2]): {line[i1], lowered_if_map[lidx-dr][i2]}")
                result['unknown exception'] += 1
                continue

            result['pattern matched'] += 1

with open(os.path.join(os.curdir, 'tmp_redundant_op_ulog.txt'), 'wt') as file:
    file.write('\n'.join(unknown_logs))

with open(os.path.join(os.curdir, 'tmp_redundant_op_lifm.txt'), 'wt') as file:
    file.write('\n'.join([f"idx: {lidx:3d} -> " + '  '.join(map(lambda x: f"{x:3d}", line)) for lidx, line in enumerate(lowered_if_map)]))

print(result)
