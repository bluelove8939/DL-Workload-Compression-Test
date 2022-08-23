import os
import numpy as np
import matplotlib.pyplot as plt


filepath = os.path.join(os.curdir, 'logs', 'sparsity_ratio_test_result.csv')
chunksize = 16
layerfilter = 'ReLU'
outputfilter = 'output0'

values_per_graph = 5  # values per one axis
starting_point = 0   # start layer index

results = {}
categories = list(range(0, chunksize+1))

fig, axes = plt.subplots(2, 1)
axes = axes.flatten()

with open(filepath, 'rt') as file:
    content = list(map(lambda x: x.split(','), file.readlines()))
    for outputinfo, *ratios in content:
        layer, lidx, outputname = outputinfo.split('_')

        if layer != layerfilter or outputfilter != outputname:
            continue
        if int(lidx) < starting_point:
            continue

        gidx = (int(lidx) - starting_point) // values_per_graph
        cidx = (int(lidx) - starting_point) % values_per_graph

        if gidx >= len(axes):
            continue

        ratios = np.array(list(map(float, ratios)))
        ratios /= np.sum(ratios)

        width_max = 0.8
        width = width_max / values_per_graph

        x_axis = np.arange(len(categories))
        xval = x_axis + ((cidx - (values_per_graph / 2) + 0.5) * width)

        axes[gidx].bar(xval, ratios, width=width, label=layer+lidx)
        axes[gidx].set_xticks(x_axis, categories, rotation=0, ha='center')
        axes[gidx].set_ylim([0, 0.6])
        axes[gidx].set_xlabel('non zeros per line')
        axes[gidx].set_ylabel('ratio')
        axes[gidx].legend(prop={'size': 10})

plt.suptitle("Sparsity Ratio of Output Activation Data (AlexNet, Fp32, 64B)")
plt.tight_layout()
plt.show()