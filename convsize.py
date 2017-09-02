# test_convsize

import numpy as np


def get_conv_outputsize(input_len, n_layers, 
                        filter_size, stride, padding=0):
    # padding: same
    # stride: 2
    # 
    if n_layers > 1:
        return get_conv_outputsize(
            get_conv_outputsize(input_len, 1, filter_size, stride, padding), 
            n_layers-1, filter_size, stride, padding)
    else:
        if padding == 'valid':
            outconv_len = input_len - filter_size + 1
        elif padding == 'same':
            outconv_len = input_len
        elif isinstance(padding, int):
            outconv_len = input_len + 2*padding - filter_size + 1
        else:
            raise ValueError('Invalid pad: {0}'.format(pad))
        output_len = np.ceil(outconv_len / stride)
        return int(output_len)

# WORKS. BUT SHOULD NOT BE NEEDED IF USING FIXED BATCH SIZE..