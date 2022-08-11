import os
import math
import numpy as np


def shannon_entropy(arr: np.ndarray, bsize: int, base: float) -> float:
    occurance = {}
    total_cnt = 0
    zerocnt = arr.flatten().shape[0] - np.count_nonzero(arr.flatten())
    arr = arr.tobytes()

    for i in range(0, len(arr), bsize):
        case = arr[i:i+bsize]
        if case not in occurance.keys():
            occurance[case] = 1
        else:
            occurance[case] += 1
        total_cnt += 1

    entropy = 0
    maxval = -1
    maxval_key = None

    for key, val in occurance.items():
        entropy += -1 * (val / total_cnt) * math.log(val / total_cnt, bsize)
        if val > maxval:
            maxval_key = key

    print(sorted(np.unique(np.array(list(occurance.values()))), reverse=True)[:5], zerocnt, maxval_key)

    return entropy


lines = []

if __name__ == '__main__':
    dirname = os.path.join(os.curdir, 'extractions_quant_wfile')
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
                arr = np.frombuffer(content, dtype=np.dtype('int8')).flatten()
                entropy = shannon_entropy(arr, bsize=8, base=2)

                lines.append(','.join(list(map(str, [modelname, filename, entropy]))))
                print(f"filename: {filename:30s}  entropy: {entropy}")

        print()

    logdirname = os.path.join(os.curdir, 'logs')
    logfilename = 'entropy_test.csv'

    with open(os.path.join(logdirname, logfilename), 'wt') as file:
        file.write('\n'.join(lines))