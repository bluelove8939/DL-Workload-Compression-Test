import math
import numpy as np

W = 28
H = 28
FW = 3
FH = 3
P = 0
S = 1
OW = math.floor((W+2*P-FW)/S) + 1
OH = math.floor((H+2*P-FH)/S) + 1

if_map = np.arange(0, W*H, 1).reshape((W, H))


# Lowering IFM
lowered_if_map = []
for rp in range(0, H-FH+1, S):
    for cp in range(0, W-FW+1, S):
        lowered_if_map.append(list(if_map[cp:cp+FW, rp:rp+FH].flatten()))

lowered_if_map = np.array(lowered_if_map)


# Exception Test
details = np.zeros_like(lowered_if_map)
result = {
    'pattern matched': 0,
    'stride exception': 0,
    'out of index exception': 0,
    'unknown exception': 0,
}

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
                details[lidx][i1] = 1
                continue

            # out of index exception
            oh = lidx % OW
            if oh - dh < 0 or oh - dh >= OW or lidx - dr < 0:
                result['out of index exception'] += 1
                details[lidx][i1] = 2
                continue

            # unknown exception
            if line[i1] != lowered_if_map[lidx-dr][i2]:
                result['unknown exception'] += 1
                details[lidx][i1] = 3
                # input(f"{i1, i2, lidx}")
                continue

            result['pattern matched'] += 1
            details[lidx][i1] = 0


print(result)

for line in details:
    print('\t'.join(map(str, line)))
