import os
import numpy as np
import matplotlib.pyplot as plt


for nz in range(0, 17):
    pass


if __name__ == '__main__':
    categories = list(range(0, 17))
    results = {}
    variances = {}
    algo_names = ['BPC', 'BDI', 'ZRLE', 'ZVC']

    for name in algo_names:
        variances[name] = []
        results[name] = []

    for nz in range(0, 17):
        ratr = {}

        for name in algo_names:
            ratr[name] = []


        with open(os.path.join(os.curdir, 'logs', f"ratio_test_result_nz_{nz}.csv"), 'rt') as file:
            content = file.readlines()
            for line in content:
                if len(line.strip()) == 0:
                    continue

                parsed = line.split(',')
                name = parsed[0].strip()
                ratio = float(parsed[1].strip())

                if name not in algo_names:
                    continue

                # print(line)

                ratr[name].append(ratio)

        for name in algo_names:
            ratr[name] = np.array(ratr[name])
            results[name].append(np.sum(ratr[name]) / len(ratr[name]))
            variances[name].append(np.var(ratr[name]))

    print(results)
    print(variances)


    plt.title("Compression Ratio with the Number of Non-Zero Words (FP32, 64B Line)")

    width_max = 0.8
    width = width_max / len(results.keys())

    x_axis = np.arange(len(categories))
    for idx, (key, val) in enumerate(results.items()):
        # xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
        plt.plot(x_axis, val, marker='o', label=key)
        # for i, j in zip(xval, val):
        #     plt.annotate(f"{j:.2f}", xy=(i, j + 0.05), ha='center')
    plt.xticks(x_axis, categories, rotation=0, ha='center')
    # plt.ylim([0.0, 8.0])
    plt.yscale('log')


    plt.legend()
    plt.tight_layout()
    plt.show()