import math


def systolic_array_cycles_ws(arr_shape, wgt_shape, act_shape):
    sh, sw = arr_shape
    wh, ww = wgt_shape
    ah, aw = act_shape

    # sta_input_cycle = sh * (ww // sw) * (wh // sh)
    # str_input_cycle = (sw * (ah // sh) + sh - 1) * (wh // sh) * (aw // sw)

    sta_tile_num = (ww // sw) * (wh // sh)
    sta_input_cycle = sh * sta_tile_num
    str_input_cycle = (aw + sh - 1) * sta_tile_num


    # if wh == aw:
    #     total_mult_num = (ww // sw) * (ah // sh) * (wh // sh)
    # elif ww == ah:
    #     total_mult_num = (wh // sh) * (aw // sw) * (ww // sw)
    # else:
    #     total_mult_num = (ww // sw) * (ah // sh) * (wh // sh)
    #
    # return unit_cycle * total_mult_num

    return sta_input_cycle + str_input_cycle


def compressed_accelerator_cycles_os(wgt_shape, act_shape, pe_num, mult_num, chunk_size):
    wh, ww = wgt_shape
    ah, aw = act_shape

    cnum_per_vec = ww // chunk_size
    vec_cycle = cnum_per_vec * math.ceil(chunk_size / mult_num)
    fold_factor = math.ceil(aw / pe_num)

    return vec_cycle * fold_factor * wh + pe_num


if __name__ == '__main__':
    arr_shape = (8, 8)
    wgt_shape = (256, 256)
    act_shape = (256, 256)

    print(systolic_array_cycles_ws(arr_shape, wgt_shape, act_shape))
    print(compressed_accelerator_cycles_os(wgt_shape, act_shape, pe_num=32, mult_num=2, chunk_size=4))