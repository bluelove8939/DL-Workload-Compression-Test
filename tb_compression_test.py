import os
import numpy as np
import argparse

from compression import BitPlaneCompressor, array2binary, bitplane_compression
from custom_streams import FileStream


parser = argparse.ArgumentParser(description='Extraction Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, 'extractions'),
                    help='Directory of model extraction files', dest='dirname')
parser.add_argument('-cs', '--chunksize', default=128, help='Size of a chunk (Bytes)', dest='chunksize')
parser.add_argument('-wd', '--wordwidth', default=32, help='Bitwidth of a word (Bits)', dest='wordwidth')
parser.add_argument('-mi', '--maxiter', default=10, help='Number of maximum iteration', dest='maxiter')
parser.add_argument('-ld', '--logdirname', default=os.path.join(os.curdir, 'logs'),
                    help='Directory of output log files', dest='logdirname')
parser.add_argument('-lf', '--logfilename', default='compression_test_result.csv',
                    help='Name of logfile', dest='logfilename')
comp_args, _ = parser.parse_known_args()


dirname = comp_args.dirname
chunksize = comp_args.chunksize
wordwidth = comp_args.wordwidth
maxiter = comp_args.maxiter
logdirname = comp_args.logdirname
logfilename = comp_args.logfilename

print("Compression Test Config")
print(f"- dirname: {dirname}")
print(f"- chunksize: {chunksize}")
print(f"- wordwidth: {wordwidth}")
print(f"- maxiter: {'undefined' if maxiter is None else maxiter}")
print(f"- logfilepath: {os.path.join(logdirname, logfilename)}\n")

results = {}


# Read each files and compress with given algorithms
for modelname in os.listdir(dirname):
    for filename in os.listdir(os.path.join(dirname, modelname)):
        if filename.__contains__('comparison_result') or filename.__contains__('filelist'):
            continue

        file_fullpath = os.path.join(dirname, modelname, filename)

        with open(file_fullpath, 'rb') as file:
            filesize = os.path.getsize(file_fullpath)
            original_total_size = 0
            compressed_total_size = 0

            print(f"reading {file_fullpath} ({filesize}Bytes)")

            for offset in range(0, filesize, chunksize):
                content = file.read(chunksize)
                arr = np.frombuffer(content, dtype=np.dtype('<f'))

                original = array2binary(arr, wordwidth)
                compressed = bitplane_compression(arr, wordwidth)

                original_total_size += len(original)
                compressed_total_size += len(compressed)

                print(f"\r[OFFSET {offset}Bytes] compression ratio: {len(original) / len(compressed):.6f} "
                      f"({original_total_size / compressed_total_size:.6f})", end='')

                if maxiter is not None and offset > maxiter * chunksize:
                    break

            comp_ratio = original_total_size / compressed_total_size
            results[file_fullpath] = ','.join(list(map(str, [modelname, filename, filesize, comp_ratio])))

        # stream = FileStream(filepath=file_fullpath, dtype=np.dtype('<f'))
        # compressor = BitPlaneCompressor(stream=stream, bandwidth=chunksize, wordbitwidth=wordwidth)
        # comp_ratio = compressor.calc_compression_ratio()
        # results[file_fullpath] = ','.join(list(map(str, [modelname, filename, stream.fullsize(), comp_ratio])))

        print(f"\ntotal compression ratio: {comp_ratio:.6f}\n")


# Save compression test results
categories = ['Model Name', 'Param Name', 'File Size(Bytes)', 'Comp Ratio(BPC)']
os.makedirs(logdirname, exist_ok=True)
with open(os.path.join(logdirname, logfilename), 'wt') as file:
    file.write('\n'.join([','.join(categories)] + list(results.values())))