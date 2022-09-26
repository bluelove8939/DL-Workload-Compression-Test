import os
import numpy as np
import argparse

from compression.modules import BitPlaneCompressor, ZeroRLECompressor, ZeroValueCompressor, EBPCompressor, ZlibCompressor, BDICompressor, EBDICompressor
from compression.custom_streams import FileStream
from compression.file_quant import FileQuantizer


parser = argparse.ArgumentParser(description='Extraction Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, '../extractions_activations'), type=str,
                    help='Directory of model extraction files', dest='dirname')
parser.add_argument('-cs', '--chunksize', default=128, type=int,
                    help='Size of a chunk (Bytes)', dest='chunksize')
parser.add_argument('-wd', '--wordwidth', default=32, type=int,
                    help='Bitwidth of a word (bits)', dest='wordwidth')
parser.add_argument('-mi', '--maxiter', default=-1, type=int,
                    help='Number of maximum iteration (-1 for no limitation)', dest='maxiter')
parser.add_argument('-dt', '--dtype', default='float32', type=str,
                    help='Dtype of numpy array', dest='dtypename')
parser.add_argument('-qdt', '--quant-dtype', default='none', type=str,
                    help='Dtype for quantization', dest='qdtypename')
parser.add_argument('-vs', '--verbose_step', default=1, type=int,
                    help='Step of verbose (print log for every Nth step for integer value N)', dest='vstep')
parser.add_argument('-pr', '--file-proportion', default=100, type=int,
                    help='File proportion (compress only N bytes if the proportion is N percent)', dest='fsprop')
parser.add_argument('-ld', '--logdirname', default=os.path.join(os.curdir, '../logs'), type=str,
                    help='Directory of output log files', dest='logdirname')
parser.add_argument('-lf', '--logfilename', default='compression_test_result.csv', type=str,
                    help='Name of logfile', dest='logfilename')
comp_args, _ = parser.parse_known_args()


dirname = comp_args.dirname
chunksize = comp_args.chunksize
wordwidth = comp_args.wordwidth
maxiter = comp_args.maxiter
dtypename = comp_args.dtypename
qdtypename = comp_args.qdtypename
vstep = comp_args.vstep
fsprop = comp_args.fsprop
logdirname = comp_args.logdirname
logfilename = comp_args.logfilename

os.makedirs(logdirname, exist_ok=True)

cnt, tmp = 2, logfilename
while tmp in os.listdir(logdirname):
    name, extn = logfilename.split('.')
    tmp = '.'.join([name + str(cnt),  extn])
    cnt += 1
logfilename = tmp

print("Compression Test Config")
print(f"- dirname: {dirname}")
print(f"- chunksize: {chunksize}Byte")
print(f"- wordwidth: {wordwidth}bit")
print(f"- dtype: {dtypename}")
print(f"- quantization: {qdtypename}")
print(f"- maxiter: {maxiter if maxiter != -1 else 'N/A'}")
print(f"- verbose step: {vstep}")
print(f"- file proportion: {fsprop}%")
print(f"- logfilepath: {os.path.join(logdirname, logfilename)}\n")

results = {}
compressors = {
    'BPC':  BitPlaneCompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    'BDI': BDICompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    'ZRLE': ZeroRLECompressor(bandwidth=64, wordbitwidth=wordwidth),
    'ZVC':  ZeroValueCompressor(bandwidth=64, wordbitwidth=wordwidth),
    'EBPC': EBPCompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    'EBDI': EBDICompressor(bandwidth=chunksize, wordbitwidth=wordwidth),
    # 'Zlib': ZlibCompressor(bandwidth=-1, wordbitwidth=wordwidth),
}


# Read each files and compress with given algorithms
for modelname in os.listdir(dirname):
    if 'output' not in modelname:
        continue

    # if 'ResNet' in modelname:
    #     continue
    #
    # if 'InceptionV3' in modelname:
    #     continue

    for filename in os.listdir(os.path.join(dirname, modelname)):
        if 'comparison_result' in filename or 'filelist' in filename:
            continue

        file_fullpath = os.path.join(dirname, modelname, filename)
        stream = FileStream()

        if qdtypename != 'none':
            qfile_fullpath = os.path.join(os.curdir, '../extractions_quant_wfile', modelname, filename)
            quantizer = FileQuantizer(orig_dtype=np.dtype(dtypename), quant_dtype=np.dtype(qdtypename))
            quantizer.quantize(filepath=file_fullpath, output_filepath=qfile_fullpath)
            stream.load_filepath(filepath=qfile_fullpath, dtype=np.dtype(qdtypename))
        else:
            stream.load_filepath(filepath=file_fullpath, dtype=np.dtype(dtypename))

        stream.proportion = fsprop
        comp_ratios = {}

        print(f"compression ratio test with {stream}({stream.filesize()}Bytes)")
        for algo_name, algo_compressor in sorted(compressors.items(), key=lambda x: x[0]):
            print(f"compressing with {algo_name}...")
            algo_compressor.instream = stream
            comp_ratios[algo_name] = algo_compressor.calc_compression_ratio(maxiter=maxiter, verbose=1, verbose_step=vstep)
            print()

        results[file_fullpath] = ','.join(list(map(str, [
            modelname, filename, stream.fullsize(), *[f"{val}" for key, val in sorted(comp_ratios.items(), key=lambda x: x[0])]
        ])))

        print(f"total compression ratio: {' '.join([f'{val:.6f}({key})' for key, val in sorted(comp_ratios.items(), key=lambda x: x[0])])}\n")


# Save compression test results
categories = ['Model Name', 'Param Name', 'File Size(Bytes)', *list(map(lambda x: f"Comp Ratio({x})", sorted(compressors.keys())))]
os.makedirs(logdirname, exist_ok=True)
with open(os.path.join(logdirname, logfilename), 'wt') as file:
    file.write('\n'.join([','.join(categories)] + list(results.values())))