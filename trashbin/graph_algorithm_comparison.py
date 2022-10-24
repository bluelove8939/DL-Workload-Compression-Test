import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from compression.algorithms import bdizv_compression, bitplane_compression, csc_compression, zrle_compression, ebp_compression, zeroval_compression

algo_methods = {
    'ZVC': zeroval_compression,
    'CSC': csc_compression,
    'ZRLE': zrle_compression,
}

sparsity = np.arange(0, 1, 0.05)
arrsize = 512
iternum = 2000
average = {}
variance = {}

for sp in tqdm.tqdm(sparsity, ncols=50):
    nz = int(np.ceil(arrsize * sp))
    ratio = {}

    for _ in range(iternum):
        arr = np.random.normal(size=arrsize).astype(np.dtype('float32'))
        idxvec = []

        for _ in range(nz):
            idx = np.random.randint(0, arrsize)
            while idx in idxvec:
                idx = np.random.randint(0, arrsize)
            idxvec.append(idx)

        arr[idxvec] = 0

        for aname, cmethod in algo_methods.items():
            if aname not in ratio.keys():
                ratio[aname] = []
            ratio[aname].append(arrsize * 32 / len(cmethod(arr, wordwidth=32)))

    for aname, cmethod in algo_methods.items():
        if aname not in average.keys():
            average[aname] = []
            variance[aname] = []

        average[aname].append(np.average(np.array(ratio[aname])))
        variance[aname].append(np.var(np.array(ratio[aname])))


filename = 'algorithm_ratio_averages.csv'
dirname = os.path.join(os.curdir, '..', 'logs')

with open(os.path.join(dirname, filename), 'wt') as file:
    lines = [f'sparsity,{",".join(sorted(algo_methods.keys()))}']
    for idx, sp in enumerate(sparsity):
        line = [f'{sp*100:.0f}']
        for aname in sorted(algo_methods.keys()):
            line.append(f'{average[aname][idx]}')
        lines.append(','.join(line))

    file.write('\n'.join(lines))


filename = 'algorithm_ratio_variances.csv'
dirname = os.path.join(os.curdir, '..', 'logs')

with open(os.path.join(dirname, filename), 'wt') as file:
    lines = [f'sparsity,{",".join(sorted(algo_methods.keys()))}']
    for idx, sp in enumerate(sparsity):
        line = [f'{sp*100:.0f}']
        for aname in sorted(algo_methods.keys()):
            line.append(f'{variance[aname][idx]}')
        lines.append(','.join(line))

    file.write('\n'.join(lines))