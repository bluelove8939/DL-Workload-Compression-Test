import os
import math
import numpy as np
from scipy.stats import entropy as ent_method


def shannon_entropy(content: bytes, bsize: int, base: float) -> float:
    occurance = {}
    for i in range(0, len(content), bsize):
        case = content[i:i+bsize]
        if case not in occurance.keys():
            occurance[case] = 1
        else:
            occurance[case] += 1

    total_cnt = math.floor(len(content) / bsize)
    probs = np.array(list(occurance.values())) / total_cnt
    entropy = np.sum(-1 * probs * np.array(list(map(lambda x: math.log(x, base), probs))))

    return float(entropy)


def modelfilter(modelname):
    return 'output' in modelname and (
        'resnet18' in modelname.lower() or
        'resnet34' in modelname.lower() or
        'alexnet'  in modelname.lower() or
        'vgg16'    in modelname.lower()
    )


if __name__ == '__main__':
    bsize = 32  # 8words for float32
    base = 2    # entropy in 'bits'

    lines = []

    dirname = os.path.join(os.curdir, 'extractions_activations')
    for modelname in list(filter(modelfilter, os.listdir(dirname))):
        if 'output' not in modelname:
            continue

        print(f"testing {modelname}")

        for filename in os.listdir(os.path.join(dirname, modelname)):
            if 'comparison_result' in filename or 'filelist' in filename:
                continue

            filepath = os.path.join(dirname, modelname, filename)

            with open(filepath, 'rb') as file:
                content = file.read()
                entropy = shannon_entropy(content, bsize=bsize, base=base)

                lines.append(','.join(list(map(str, [modelname, filename, entropy]))))
                print(f"filename: {filename:30s}  entropy: {entropy}")

        print()

    logdirname = os.path.join(os.curdir, 'logs')
    logfilename = 'entropy_test'

    os.makedirs(logdirname, exist_ok=True)

    if logfilename in os.listdir(logdirname):
        lfidx = 2
        while f"{logfilename}{lfidx}.csv" in os.listdir(logdirname):
            lfidx += 1
        logfilename = f"{logfilename}{lfidx}"

    with open(os.path.join(logdirname, logfilename + '.csv'), 'wt') as file:
        file.write('\n'.join(lines))