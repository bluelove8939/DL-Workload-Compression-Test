import os
import numpy as np
import argparse

from models.tools.progressbar import progressbar
from compression.modules import BitPlaneCompressor, ZeroRLECompressor, ZeroValueCompressor, EBPCompressor, ZlibCompressor, BDICompressor, EBDICompressor
from compression.custom_streams import FileStream
from compression.binary_array import array2binary


parser = argparse.ArgumentParser(description='Extraction Configs')
parser.add_argument('-fp', '--filepath',
                    default=os.path.join(os.curdir, 'extractions_activations', 'AlexNet_Imagenet_output', 'Conv2d_0_output0'),
                    type=str, help='Path to output activation data dump file', dest='filepath')
parser.add_argument('-cs', '--chunksize', default=64, type=int,
                    help='Size of a chunk (Bytes)', dest='chunksize')
parser.add_argument('-wd', '--wordwidth', default=8, type=int,
                    help='Bitwidth of a word (bits)', dest='wordwidth')
parser.add_argument('-dt', '--dtype', default='int8', type=str,
                    help='Dtype of numpy array', dest='dtypename')
parser.add_argument('-ta', '--test-algorithm', default='BPC', dest='taname',
                    help='Algorithm to test compression ratio with bitpattern (BDI, BPC, EBDI, EBPC, ZRLE, ZVC)',
                    type=str)
parser.add_argument('-ca', '--compare-algorithm', default='ZRLE', dest='caname',
                    help='Algorithm to compare compression ratio (BDI, BPC, EBDI, EBPC, ZRLE, ZVC)', type=str)
parser.add_argument('-th', '--thres', default=0, type=float,
                    help='Thresholds of compression ratio', dest='thres')
parser.add_argument('-nz', '--non-zeros', default=-1, dest='nonzero_num',
                    help='Number of non-zero words in a cache line', type=int)
parser.add_argument('-pr', '--file-proportion', default=20, type=int,
                    help='File proportion (compress only N bytes if the proportion is N percent)', dest='fsprop')
parser.add_argument('-ld', '--logdirname', default=os.path.join(os.curdir, 'logs', 'ratio_test_result_int8'), type=str,
                    help='Directory of output log files', dest='logdirname')
parser.add_argument('-lf', '--logfilename', default='bitpattern_test_result.csv', type=str,
                    help='Name of logfile', dest='logfilename')
comp_args, _ = parser.parse_known_args()


filepath = comp_args.filepath
chunksize = comp_args.chunksize
wordwidth = comp_args.wordwidth
test_algorithm = comp_args.taname
compare_algorithm = comp_args.caname
dtypename = comp_args.dtypename
thres = comp_args.thres
fsprop = comp_args.fsprop
nonzero_num = comp_args.nonzero_num
logdirname = comp_args.logdirname
logfilename = comp_args.logfilename

os.makedirs(logdirname, exist_ok=True)

cnt, tmp = 2, logfilename
while tmp in os.listdir(logdirname):
    name, extn = logfilename.split('.')
    tmp = '.'.join([name + str(cnt),  extn])
    cnt += 1
logfilename = tmp

print("Bitpattern Compression Test Config")
print(f"- file path: {filepath}")
print(f"- chunk size: {chunksize}Byte")
print(f"- dtype: {dtypename}")
print(f"- word width: {wordwidth}bit")
print(f"- test algorithm: {test_algorithm}")
print(f"- compare algorithm: {compare_algorithm}")
print(f"- threshold: {thres}")
print(f"- file proportion: {fsprop}%")
print(f"- logfile path: {os.path.join(logdirname, logfilename)}\n")

results = []
compressors = {
    'BPC':  BitPlaneCompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    'BDI':  BDICompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    'ZRLE': ZeroRLECompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    'ZVC':  ZeroValueCompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    # 'EBPC': EBPCompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    # 'EBDI': EBDICompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    # 'Zlib': ZlibCompressor(bandwidth=-1, wordbitwidth=wordwidth),
}


stream = FileStream()
stream.load_filepath(filepath=filepath, dtype=np.dtype(dtypename))
stream.proportion = fsprop

print(f"{progressbar(status=0, total=stream.fullsize(), scale=50)} {0:3d}%", end='')

while True:
    comp_results = {}

    test_ratio = 0
    arr = stream.fetch(size=chunksize)

    if arr is None:
        break

    original = array2binary(arr, wordwidth)
    comp_results['RAW'] = (1, '[' + ', '.join(list(map(lambda x: str(x), arr))) + ']')
    comp_results['ORIG'] = (1, original)

    for algo_name, algo_compressor in compressors.items():
        compressed = algo_compressor.compress(arr)
        comp_results[algo_name] = (len(original) / len(compressed), compressed)

    if abs(comp_results[test_algorithm][0] - comp_results[compare_algorithm][0]) > thres and np.count_nonzero(arr) == nonzero_num:
        results.append('\n'.join([f"{key:5s},{ratio:.6f},{res}" for key, (ratio, res) in comp_results.items()]))

    print(f"\r{progressbar(status=stream.cursor, total=stream.fullsize(), scale=50)} {int(stream.cursor / stream.fullsize() * 100):3d}%", end='')

print('\n\n')

with open(os.path.join(logdirname, logfilename), 'wt') as logfile:
    logfile.write('\n\n'.join(results))
