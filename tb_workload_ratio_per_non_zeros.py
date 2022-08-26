import os
import numpy as np
import matplotlib.pyplot as plt


dtypename = 'fp32'
chunksize = 64
wordwidth = 32
quant = False


if __name__ == '__main__':
    categories = list(range(0, (chunksize // (wordwidth // 8)) + 1))
    results = {}
    variances = {}
    algo_names = ['BPC', 'BDI', 'ZRLE', 'ZVC']

    for name in algo_names:
        variances[name] = []
        results[name] = []

    for nz in categories:
        ratr = {}

        for name in algo_names:
            ratr[name] = []

        with open(os.path.join(os.curdir, 'logs', f'ratio_test_result{"_quant" if quant else ""}_{dtypename}_cs{chunksize}', f"ratio_test_result_nz_{nz}.csv"), 'rt') as file:
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


    width_max = 0.8
    width = width_max / len(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    # fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})

    x_axis = np.arange(len(categories))
    for idx, (key, val) in enumerate(results.items()):
        ax1.plot(x_axis, val, marker='o', label=key)
        # ax1.plot(x_axis, val, label=key)
        # if idx != 1: continue
        # for i, j in zip(x_axis, val):
        #     ax1.annotate(f"{j:.2f}", xy=(i, j + 0.02), ha='center', size=7)
    ax1.set_xticks(x_axis, categories, rotation=0, ha='center')
    ax1.set_yscale('log')
    ax1.set_xlabel('number of non-zero words per cache line')
    ax1.set_ylabel('compression ratio [log]')
    ax1.legend()

    for idx, (key, val) in enumerate(variances.items()):
        xval = x_axis + ((idx - (len(variances.keys()) / 2) + 0.5) * width)
        ax2.bar(xval, val, width=width, label=key)
        # for i, j in zip(xval, val):
        #     ax2.annotate(f"{j:.2f}", xy=(i, j + 0.02), ha='center', size=7)
    ax2.set_xticks(x_axis, categories, rotation=0, ha='center')
    # ax2.set_yscale('log')
    # ax2.set_ylim([0, 1])
    ax2.set_xlabel('number of non-zero words per cache line')
    ax2.set_ylabel('variance')
    ax2.legend()

    plt.suptitle(f"Compression Ratio with the Number of Non-Zero Words per Cache Line ({dtypename.upper()}, {chunksize}B Line)")
    plt.tight_layout()
    plt.show()