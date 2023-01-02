import math


def systolic_array_cycles_ws(arr_shape, wgt_shape, act_shape):
    sh, sw = arr_shape
    wh, ww = wgt_shape
    ah, aw = act_shape

    sta_tile_num = (ww // sw) * (wh // sh)
    sta_input_cycle = sh * sta_tile_num
    str_input_cycle = (aw + sh - 1) * sta_tile_num

    return sta_input_cycle + str_input_cycle


def compressed_accelerator_cycles_os(wgt_shape, act_shape, pe_num, mult_num, chunk_size):
    wh, ww = wgt_shape
    ah, aw = act_shape

    cnum_per_vec = ww // chunk_size
    chunk_cycle = math.ceil(chunk_size / mult_num)
    fold_factor = math.ceil(aw / pe_num)

    return cnum_per_vec * chunk_cycle * fold_factor * wh + pe_num


if __name__ == '__main__':
    arr_shape = (8, 8)
    wgt_shape = (64, 4096)
    act_shape = (4096, 128)

    sa_cycle = systolic_array_cycles_ws(arr_shape, wgt_shape, act_shape)
    ca_cycle = compressed_accelerator_cycles_os(wgt_shape, act_shape, pe_num=32, mult_num=2, chunk_size=4)

    print(sa_cycle)
    print(ca_cycle)
    print(sa_cycle / ca_cycle)