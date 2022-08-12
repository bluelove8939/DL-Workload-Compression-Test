import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Result Analyzing Configs')
parser.add_argument('-fp', '--filepath', default=os.path.join(os.curdir, 'logs', 'compression_test_result8.csv'),
                    help='Path to result csv file', dest='filepath')
comp_args, _ = parser.parse_known_args()


filepath = comp_args.filepath
categories = []
results = {}
algo_names = ['BPC', 'ZRLE', 'ZVC']
# algo_names = ['BPC', 'ZRLE']

for name in algo_names:
    results[name] = []
results['total size'] = []

with open(filepath, 'rt') as file:
    content = list(map(lambda x: x.split(','), file.readlines()[1:]))
    content = sorted(content, key=lambda x: x[0])

    for model_name, param_name, file_size, *comp_ratios in content:
        model_name = model_name.split('_')[0]
        if model_name not in categories:
            categories.append(model_name)
            for nidx, name in enumerate(algo_names):
                results[name].append(int(file_size) / float(comp_ratios[nidx]))
            results['total size'].append(int(file_size))
        else:
            for nidx, name in enumerate(algo_names):
                results[name][categories.index(model_name)] += int(file_size) / float(comp_ratios[nidx])
            results['total size'][categories.index(model_name)] += int(file_size)

    for name in algo_names:
        results[name] = np.array(results['total size']) / np.array(results[name])
    del results['total size']

width_max = 0.8
width = width_max / len(results.keys())

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(results.items()):
    xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
    plt.bar(xval, val, width=width, label=key)
    for i, j in zip(xval, val):
        plt.annotate(f"{j:.2f}", xy=(i, j+0.06), ha='center')
plt.xticks(x_axis, categories, rotation=0, ha='center')
plt.ylim([0.0, 3.5])

plt.title("Compression Algorithm test on DL activations (FP32)")
plt.legend()
plt.tight_layout()
plt.show()