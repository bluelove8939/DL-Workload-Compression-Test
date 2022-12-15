import os
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

from compression.algorithms import zeroval_compression, bitplane_compression, csc_compression, zrle_compression, ebp_compression

algo_methods = {
    'ZVC': zeroval_compression,
    'ZRLE': bitplane_compression,
    'CSC': csc_compression,
}

sparsity = np.arange(0, 1, 0.05)
height = 64
width = 64
arrsize = height * width
iternum = 1000
results = {}

for sp in tqdm.tqdm(sparsity, ncols=50):
    nz = int(np.ceil(arrsize * sp))
    avg_ratio = {}

    mat = np.ceil(np.random.normal(size=arrsize) * 128).astype(np.dtype('int8'))
    idxvec = random.sample(range(len(mat)), nz)
    mat[idxvec] = 0
    mat = np.reshape(mat, newshape=(height, width))

    for _ in range(iternum):
        for arr in mat:
            # arr = np.ceil(np.random.normal(size=arrsize) * 128).astype(np.dtype('int8'))

            for aname, cmethod in algo_methods.items():
                if aname not in avg_ratio.keys():
                    avg_ratio[aname] = 0
                avg_ratio[aname] += arrsize * 8 / len(cmethod(arr, wordwidth=8))

    for aname, cmethod in algo_methods.items():
        if aname not in results.keys():
            results[aname] = []

        results[aname].append(avg_ratio[aname] / (iternum * height))

filename = 'accelerator_algorithm_test.csv'
dirname = os.path.join(os.curdir, 'logs')

with open(os.path.join(dirname, filename), 'wt') as file:
    lines = [f'sparsity,{",".join(sorted(algo_methods.keys()))}']
    for idx, sp in enumerate(sparsity):
        line = [f'{sp*100:.0f}']
        for aname in sorted(algo_methods.keys()):
            line.append(f'{results[aname][idx]}')
        lines.append(','.join(line))

    file.write('\n'.join(lines))