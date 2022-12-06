import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Sparsity Test Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, '../../extractions_quant_wfile'),
                    help='Directory of model extraction files', dest='extdir')
parser.add_argument('-dt', '--dtype', default='int8', type=str, help='Dtype of numpy array', dest='dtypename')
comp_args, _ = parser.parse_known_args()


dirname = comp_args.extdir
dtypename = comp_args.dtypename

print("Sparsity Test Config")
print(f"- dirname: {dirname}")
print(f"- dtype: {dtypename}\n")


if __name__ == '__main__':
    lines = []

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
                arr = np.frombuffer(content, dtype=np.dtype(dtypename)).flatten()
                arrsize = arr.shape[0]
                nonzerocnt = np.count_nonzero(arr)
                zerocnt = arrsize - nonzerocnt

                lines.append(','.join(list(map(str, [modelname, filename, arrsize, zerocnt]))))
                print(f"filename: {filename:30s}  arrsize: {arrsize}  zerocnt: {zerocnt}")

        print()

    logdirname = os.path.join(os.curdir, '../../logs')
    logfilename = 'sparsity_test'

    os.makedirs(logdirname, exist_ok=True)

    if f"{logfilename}.csv" in os.listdir(logdirname):
        lfidx = 2
        while f"{logfilename}{lfidx}" in os.listdir(logdirname):
            lfidx += 1
        logfilename = f"{logfilename}{lfidx}"

    with open(os.path.join(logdirname, logfilename + '.csv'), 'wt') as file:
        file.write('\n'.join(lines))