import os
import numpy as np


lines = []

if __name__ == '__main__':
    dirname = os.path.join(os.curdir, 'extractions')
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
                print(f"filename: {filename}  arrsize: {arrsize}  zerocnt: {zerocnt}")

        print()

    logdirname = os.path.join(os.curdir, 'logs')
    logfilename = 'sparsity_test.csv'

    with open(os.path.join(logdirname, logfilename), 'wt') as file:
        file.write('\n'.join(lines))