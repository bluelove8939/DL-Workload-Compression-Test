import os
import numpy as np
import matplotlib.pyplot as plt


filepath = os.path.join(os.curdir, '../logs', 'sparsity_ratio_test_result3.csv')
chunksize = 16
layerfilter = 'ConvReLU2d'
outputfilter = 'output0'

values_per_graph = 5  # values per one axis
starting_point = 0   # start layer index

results = {}
categories = list(range(0, chunksize+1))

fig, axes = plt.subplots(3, 4, constrained_layout=False)
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
        axes[gidx].set_xticks(x_axis, categories, rotation=0, ha='center', size=6)
        axes[gidx].set_ylim([0, 1.])
        axes[gidx].set_xlabel('non zeros per line')
        axes[gidx].set_ylabel('ratio')
        axes[gidx].legend(prop={'size': 6})

# plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
plt.suptitle("Sparsity Ratio of Output Activation Data (InceptionV3, INT8, 16B)")
plt.tight_layout()
plt.show()