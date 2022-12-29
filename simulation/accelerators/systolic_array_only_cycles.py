def systolic_array_cycles(arr_shape, wgt_shape, act_shape):
    sh, sw = arr_shape
    wh, ww = wgt_shape
    ah, aw = act_shape

    unit_cycle = 3*sh + sw - 2
    wgt_mult = (wh // sh) * (ww // sw)
    act_mult = (ah // sh) * (aw // sw)

    return unit_cycle * wgt_mult * act_mult


if __name__ == '__main__':
    arr_shape = (8, 8)
    wgt_shape = (32, 32)
    act_shape = (32, 32)

    print(systolic_array_cycles(arr_shape, wgt_shape, act_shape))