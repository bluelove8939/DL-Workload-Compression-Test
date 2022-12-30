def systolic_array_cycles_ws(arr_shape, wgt_shape, act_shape):
    sh, sw = arr_shape
    wh, ww = wgt_shape
    ah, aw = act_shape

    # unit_cycle = 3*sh + sw - 2
    sta_input_cycle = sh * (ww // sw) * (wh // sh)
    str_input_cycle = (sw * (ah // sh) + sh - 1) * (wh // sh) * (aw // sw)

    # if wh == aw:
    #     total_mult_num = (ww // sw) * (ah // sh) * (wh // sh)
    # elif ww == ah:
    #     total_mult_num = (wh // sh) * (aw // sw) * (ww // sw)
    # else:
    #     total_mult_num = (ww // sw) * (ah // sh) * (wh // sh)
    #
    # return unit_cycle * total_mult_num

    return sta_input_cycle + str_input_cycle


if __name__ == '__main__':
    arr_shape = (8, 8)
    wgt_shape = (32, 32)
    act_shape = (32, 32)

    print(systolic_array_cycles_ws(arr_shape, wgt_shape, act_shape))