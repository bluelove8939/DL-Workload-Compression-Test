import numpy as np
import matplotlib.pyplot as plt

xmin, xmax = -4, 4
ymin, ymax = -2, 4

def relu(x: np.ndarray):
    y = x.copy()
    y[x < 0] = 0
    return y

x = np.arange(xmin, xmax, 0.1)
y = relu(x)

# print(x)
# print(y)

ax = plt.axes()
ax.figure.set_size_inches(4.4, 3)

ax.plot(x, y, "black", linewidth=2)
ax.text(1.5, 3.5, '$y=ReLU(x)$', fontsize=11)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

ax.set_ylim([ymin, ymax])
ax.set_xlim([xmin, xmax])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

xticks = list(range(xmin+1, 0, 1)) + list(range(1, xmax, 1))
yticks = list(range(ymin+1, 0, 1)) + list(range(1, ymax, 1))

ax.set_xticks(xticks, xticks, ha='center', font='serif', fontsize=10)
ax.set_yticks(yticks, yticks, ha='center', font='serif', fontsize=10)

ax.set_xlabel('$x$', fontname='serif', fontsize=10)
ax.set_ylabel('$y$', rotation=0, fontname='serif', fontsize=10)
ax.xaxis.set_label_coords(1.00, 0.30)
ax.yaxis.set_label_coords(0.47, 0.97)

plt.tight_layout()
plt.savefig("G:\내 드라이브\국과연_보고서\Fig_relu_graph.pdf", dpi=200, bbox_inches='tight', pad_inches=0)
# plt.show()