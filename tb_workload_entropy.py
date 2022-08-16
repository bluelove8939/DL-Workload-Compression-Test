import os
import math
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Entropy Test Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, 'extractions_quant_activations'),
                    help='Directory of model extraction files', dest='extdir')
parser.add_argument('-bs', '--bsize', default=4, type=int, help='Size of data block (Bytes)', dest='bsize')
parser.add_argument('-ba', '--base', default=2, type=int, help='Base of Shannon\'s entropy', dest='base')
comp_args, _ = parser.parse_known_args()


dirname = comp_args.extdir
bsize = comp_args.bsize
base = comp_args.base

print("Entropy Test Config")
print(f"- dirname: {dirname}")
print(f"- block size: {bsize}")
print(f"- base: {base}\n")


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
    # return 'output' in modelname and (
    #     'resnet18' in modelname.lower() or
    #     'resnet34' in modelname.lower() or
    #     'alexnet'  in modelname.lower() or
    #     'vgg16'    in modelname.lower()
    # )
    return True


if __name__ == '__main__':
    lines = []

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

    if f"{logfilename}.csv" in os.listdir(logdirname):
        lfidx = 2
        while f"{logfilename}{lfidx}.csv" in os.listdir(logdirname):
            lfidx += 1
        logfilename = f"{logfilename}{lfidx}"

    with open(os.path.join(logdirname, logfilename + '.csv'), 'wt') as file:
        file.write('\n'.join(lines))