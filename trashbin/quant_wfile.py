import os
import numpy as np
import argparse

from compression.custom_streams import FileStream
from compression.file_quant import FileQuantizer


parser = argparse.ArgumentParser(description='Quantization with File Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, '../extractions_activations'), type=str,
                    help='Directory of model extraction files', dest='dirname')
parser.add_argument('-dt', '--dtype', default='float32', type=str, help='Dtype of numpy array', dest='dtypename')
parser.add_argument('-qdt', '--quant-dtype', default='int8', type=str, help='Dtype for quantization', dest='qdtypename')
comp_args, _ = parser.parse_known_args()


dirname = comp_args.dirname
dtypename = comp_args.dtypename
qdtypename = comp_args.qdtypename

print("Quantization with File Config")
print(f"- dirname: {dirname}")
print(f"- dtype: {dtypename}")
print(f"- quantization: {qdtypename}")


# Read each files and compress with given algorithms
for modelname in os.listdir(dirname):
    if 'output' not in modelname:
        continue

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

        print(f"compression ratio test with {stream}({stream.fullsize()}Bytes)")
