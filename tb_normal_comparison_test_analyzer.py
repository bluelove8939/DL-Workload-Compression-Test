import os
import numpy as np
import matplotlib.pyplot as plt


categories = []
results = {}


for model_name in os.listdir(os.path.join(os.curdir, 'extractions')):
    result_path = os.path.join(os.curdir, 'extractions', model_name, 'comparison_results.csv')
    with open(result_path, 'rt') as file:
        content = file.readlines()

    algo_indexes = list(map(lambda x: x.strip(), content[0].split(',')))[1:]
    avg_cratio = np.array([0] * len(algo_indexes), dtype=float)
    size_sum = 0

    for line in content[1:]:
        lineparsed = line.split(',')
        avg_cratio += np.array(list(map(lambda x: float(x.strip()), lineparsed[1:]))) * os.path.getsize(lineparsed[0])
        size_sum += os.path.getsize(lineparsed[0])

    avg_cratio /= size_sum

    categories.append(model_name.split('_')[0])

    for algo_name in algo_indexes:
        if algo_name not in results.keys():
            results[algo_name] = []

    for key, val in zip(algo_indexes, avg_cratio):
        results[key].append(val)

width_max = 0.8
width = width_max / len(results.keys())

x_axis = np.arange(len(categories))
for idx, (key, val) in enumerate(results.items()):
    xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
    plt.bar(xval, val, width=width, label=key)
    for i, j in zip(xval, val):
        plt.annotate(f"{j:.2f}", xy=(i, j+0.2), ha='center')
plt.xticks(x_axis, categories, rotation=0, ha='center')
plt.ylim([0, 6])

plt.title("Compression algorithm comparison on DL weight (FP32)")
plt.legend()
plt.tight_layout()
plt.show()