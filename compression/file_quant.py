import os
import numpy as np


class FileQuantizer(object):
    def __init__(self, orig_dtype: np.dtype, quant_dtype: np.dtype, normalize=True):
        self.orig_dtype = orig_dtype    # original dtype of the given file e.g. float32
        self.quant_dtype = quant_dtype  # quantized dtype of the given file e.g. float16, int8...
        self.normalize = normalize      # normalize given array

    def quantize(self, filepath: str, output_filepath: str) -> None:
        with open(filepath, 'rb') as file:
            content = file.read()
            arr = np.frombuffer(content, dtype=self.orig_dtype)

        quant_arr = None
        if 'float' in self.quant_dtype.name:
            quant_arr = arr.clone()
        elif 'uint' in self.quant_dtype.name:
            bitwidth = int(self.quant_dtype.name[4:])
            if self.normalize:
                arr = arr / arr.max().item()
            quant_arr = arr * (2**(bitwidth-1)-1)
        elif 'int' in self.quant_dtype.name:
            bitwidth = int(self.quant_dtype.name[3:])
            if self.normalize:
                arr = arr / np.abs(arr).max().item()
            quant_arr = arr * (2 ** (bitwidth - 1) - 1)

        quant_arr = quant_arr.astype(self.quant_dtype)
        os.makedirs(os.path.split(output_filepath)[0], exist_ok=True)

        with open(output_filepath, 'wb') as file:
            file.write(quant_arr.tobytes())


if __name__ == '__main__':
    dirname = os.path.join(os.curdir, '../extractions', 'ResNet50_Imagenet')
    filepath = os.path.join(dirname, 'ResNet50_Imagenet_conv1_weight')

    quant_dirname = os.path.join(os.curdir, 'extractions_quant_wfiles', 'ResNet50_Imagenet')
    quant_filepath = os.path.join(quant_dirname, 'ResNet50_Imagenet_conv1_weight')

    os.makedirs(dirname, exist_ok=True)
    os.makedirs(quant_dirname, exist_ok=True)

    quantizer = FileQuantizer(orig_dtype=np.dtype('float32'), quant_dtype=np.dtype('int8'))
    quantizer.quantize(filepath=filepath, output_filepath=quant_filepath)

    from compression import array2binary, print_binary
    from compression.custom_streams import FileStream

    stream = FileStream(filepath=filepath, dtype=np.dtype('float32'))
    arr = stream.fetch(32)
    print_binary(array2binary(arr, 32), 32)

    qstream = FileStream(filepath=quant_filepath, dtype=np.dtype('int8'))
    qarr = qstream.fetch(8)
    print_binary(array2binary(qarr, 8), 8)
