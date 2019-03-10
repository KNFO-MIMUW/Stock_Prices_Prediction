import numpy as np
import math
import pywt


def denoise(data, lvl=2, wavelet='Haar'):
    data = np.atleast_2d(data)
    (row_num, col_num) = np.shape(data)
    denoised_rows = []

    for i in range(0, row_num):
        coeffs = pywt.wavedec(data[i, ], wavelet, level=lvl)
        finest = coeffs[-1]

        sigma = np.median(np.abs(finest))/0.6745
        threshold = sigma*math.sqrt(2*math.log(col_num))

        for j in range(1, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold)

        denoised_row = pywt.waverec(coeffs, wavelet)
        denoised_rows.append(denoised_row)

    return np.vstack(denoised_rows)
