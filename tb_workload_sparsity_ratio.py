import os
import numpy as np
import argparse

from models.tools.progressbar import progressbar
from compression.custom_streams import FileStream


parser = argparse.ArgumentParser(description='Testbench Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, 'extractions_quant_activations', 'GoogLeNet_Imagenet_output'),
                    help='Directory of model extraction files', dest='dirname')
parser.add_argument('-cs', '--chunksize', default=16, type=int,
                    help='Size of a chunk (Bytes)', dest='chunksize')
parser.add_argument('-wd', '--wordwidth', default=8, type=int,
                    help='Bitwidth of a word (bits)', dest='wordwidth')
parser.add_argument('-dt', '--dtype', default='int8', type=str,
                    help='Dtype of numpy array', dest='dtypename')
parser.add_argument('-vs', '--verbose_step', default=1000, type=int,
                    help='Step of verbose (print log for every Nth step for integer value N)', dest='vstep')
parser.add_argument('-ld', '--logdirname', default=os.path.join(os.curdir, 'logs'), type=str,
                    help='Directory of output log files', dest='logdirname')
parser.add_argument('-lf', '--logfilename', default='sparsity_ratio_test_result.csv', type=str,
                    help='Name of logfile', dest='logfilename')
comp_args, _ = parser.parse_known_args()


dirname = comp_args.dirname
chunksize = comp_args.chunksize
wordwidth = comp_args.wordwidth
dtypename = comp_args.dtypename
vstep = comp_args.vstep
logdirname = comp_args.logdirname
logfilename = comp_args.logfilename

os.makedirs(logdirname, exist_ok=True)

cnt, tmp = 2, logfilename
while tmp in os.listdir(logdirname):
    name, extn = logfilename.split('.')
    tmp = '.'.join([name + str(cnt),  extn])
    cnt += 1
logfilename = tmp

print("Sparsity Ratio Test Config")
print(f"- dirname: {dirname}")
print(f"- chunk size: {chunksize}Byte")
print(f"- dtype: {dtypename}")
print(f"- word width: {wordwidth}bit")
print(f"- verbose step: {vstep}")
print(f"- logfile path: {os.path.join(logdirname, logfilename)}\n")


results = {}
stream = FileStream()

filenum = len(os.listdir(dirname))

def print_verbose():
    print(f"\r{progressbar(status=stream.cursor, total=stream.fullsize(), scale=50)} "
          f"{int(stream.cursor / stream.fullsize() * 100):3d}% [{stream.cursor}/{stream.fullsize()}]  "
          f"filename: {filename}({stream.fullsize()}Bytes) [{fidx + 1}/{filenum}]", end='')

for fidx, filename in enumerate(os.listdir(dirname)):
    if 'comparison_result' in filename or 'filelist' in filename:
        continue

    filepath = os.path.join(dirname, filename)
    stream.load_filepath(filepath=filepath, dtype=np.dtype(dtypename))
    stream.reset()

    results[filename] = np.zeros((chunksize // wordwidth * 8) + 1)


    cnt_iter = 0
    print_verbose()

    while True:
        arr = stream.fetch(chunksize)
        if arr is None: break
        results[filename][np.count_nonzero(arr)] += 1

        cnt_iter += 1

        if cnt_iter % vstep == 0:
            print_verbose()

    print_verbose()


with open(os.path.join(logdirname, logfilename), 'wt') as logfile:
    logfile.write('\n'.join([f"{key},{','.join(list(map(str, val)))}" for key, val in results.items()]))