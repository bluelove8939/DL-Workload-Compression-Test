import numpy as np


def quantize(x, quant, safetyFactor=1.0, normalize=False):
    if quant[:5] == 'float':
        numBit = int(quant[5:])
        if x is not None:
            x = x.clone()  # perform no quantization; comes with dtype conversion
        if numBit == 32:
            dtype = np.float32
        elif numBit == 16:
            dtype = np.float16
        else:
            assert False

    elif quant[:5] == 'fixed':
        numBit = int(quant[5:])
        if numBit > 16:
            assert (numBit <= 32)
            dtype = np.int32
        elif numBit > 8:
            dtype = np.int16
        else:
            dtype = np.int8

        if x is not None:
            if normalize:
                x = x.div(x.abs().max().item() / safetyFactor)  # now op in [-1.0, 1.0]
            x = x.mul(2 ** (numBit - 1) - 1)  # now quantized to use full range

    elif quant[:6] == 'ufixed':
        numBit = int(quant[6:])
        if numBit > 16:
            assert (numBit <= 32)
            dtype = np.uint32
        elif numBit > 8:
            dtype = np.uint16
        else:
            dtype = np.uint8
        assert (x.ge(0).all().item())
        if x is not None:
            if normalize:
                x = x.div(x.max().item() / safetyFactor)  # now op in [0, 1.0]
            x = x.mul(2 ** numBit - 1)  # now quantized to use full range

    else:
        assert False

    return x, numBit, dtype


getWW = lambda qm: quantize(None, quant=qm)[1]