import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Result Analyzing Configs')
parser.add_argument('-fp', '--filepath', default=os.path.join(os.curdir, 'logs', 'compression_test_result.csv'),
                    help='Path to result csv file', dest='filepath')
comp_args, _ = parser.parse_known_args()


filepath = comp_args.filepath
categories = []
results = {
    'BPC': [],
    # 'BDI': [],
    'total size': [],
}

with open(filepath, 'rt') as file:
    content = list(map(lambda x: x.split(','), file.readlines()[1:]))
    content = sorted(content, key=lambda x: x[0])
    # for model_name, param_name, file_size, bpc_comp_ratio, bdi_comp_ratio in content:
    for model_name, param_name, file_size, bpc_comp_ratio in content:
        model_name = model_name.split('_')[0]
        if model_name not in categories:
            categories.append(model_name)
            results['BPC'].append(int(file_size) / float(bpc_comp_ratio))
            # results['BDI'].append(int(file_size) / float(bdi_comp_ratio))
            results['total size'].append(int(file_size))
        else:
            results['BPC'][categories.index(model_name)] += int(file_size) / float(bpc_comp_ratio)
            # results['BDI'][categories.index(model_name)] += int(file_size) / float(bdi_comp_ratio)
            results['total size'][categories.index(model_name)] += int(file_size)

    results['BPC'] = np.array(results['total size']) / np.array(results['BPC'])
    # results['BDI'] = np.array(results['total size']) / np.array(results['BDI'])
    del results['total size']

width_max = 0.8
width = width_max / len(results.keys())

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(results.items()):
    xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
    plt.bar(xval, val, width=width, label=key)
    for i, j in zip(xval, val):
        plt.annotate(f"{j:.2f}", xy=(i, j+0.2), ha='center')
plt.xticks(x_axis, categories, rotation=0, ha='center')
plt.ylim([0, 3])

plt.title("Compression Algorithm test on DL activations (INT8)")
plt.legend()
plt.tight_layout()
plt.show()