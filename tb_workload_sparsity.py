import os
import numpy as np


if __name__ == '__main__':
    lines = []

    dirname = os.path.join(os.curdir, 'extractions_activations')
    for modelname in os.listdir(dirname):
        if 'output' not in modelname:
            continue

        print(f"testing {modelname}")

        for filename in os.listdir(os.path.join(dirname, modelname)):
            if 'comparison_result' in filename or 'filelist' in filename:
                continue

            filepath = os.path.join(dirname, modelname, filename)

            with open(filepath, 'rb') as file:
                content = file.read()
                arr = np.frombuffer(content, dtype=np.dtype('float')).flatten()
                arrsize = arr.shape[0]
                nonzerocnt = np.count_nonzero(arr)
                zerocnt = arrsize - nonzerocnt

                lines.append(','.join(list(map(str, [modelname, filename, arrsize, zerocnt]))))
                print(f"filename: {filename:30s}  arrsize: {arrsize}  zerocnt: {zerocnt}")

        print()

    logdirname = os.path.join(os.curdir, 'logs')
    logfilename = 'sparsity_test'

    os.makedirs(logdirname, exist_ok=True)

    if logfilename in os.listdir(logdirname):
        lfidx = 2
        while f"{logfilename}{lfidx}" in os.listdir(logdirname):
            lfidx += 1
        logfilename = f"{logfilename}{lfidx}"

    with open(os.path.join(logdirname, logfilename + '.csv'), 'wt') as file:
        file.write('\n'.join(lines))