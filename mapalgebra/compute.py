import operator as op
import os
import sys

import numpy as np
import pymannkendall as mk
from loguru import logger

BANDS = 36


def numberic(trend: str) -> int:
    if trend == 'no trend':
        return 1
    elif trend == 'increasing':
        return 2
    elif trend == 'decreasing':
        return 3


def con(slope, p) -> int:
    if p <= 0.01 and slope <= 0:
        return 1
    elif 0.01 < p and p <= 0.05 and slope <= 0:
        return 2
    elif 0.05 < p and slope <= 0:
        return 3
    elif p <= 0.01 and slope > 0:
        return 6
    elif 0.01 < p and p <= 0.05 and slope > 0:
        return 5
    elif 0.05 < p and slope > 0:
        return 4
    else:
        return 0


# def compute2(data):
#     shape = data.shape
#     result = np.zeros(shape[1:])
#     with np.nditer(result, flags=['multi_index'], op_flags=['writeonly']) as it:
#         for x in it:
#             # print(it.multi_index)
#             vec = data[(slice(None), *it.multi_index)]
#             trend = mk.original_test(vec)
#             x[...] = con(trend.slope, trend.p)
#     return result


def getCompute(test, pbar, slice_fn):
    print('test:', test)

    def compute(data):
        shape = data.shape
        # pbar and pbar.update(shape[1] * shape[2])
        result = np.zeros([BANDS, *shape[1:]])
        for idx in np.ndindex(shape[1:]):
            vec = data[(slice(None), *idx)]
            vec = vec[slice_fn]
            logger.debug(vec)
        return result
    return compute


def compute(data, test, agg=None):
    shape = data.shape
    result = None
    for idx in np.ndindex(shape[1:]):
        pixel = data[(slice(None), *idx)]
        # pixel[np.isnan(pixel)] = 0
        pixel = np.nan_to_num(pixel)
        if agg:
            pixel = pixel.reshape(-1, 12)
            pixel = np.array([sum(row[agg[0]:agg[1]]) for row in pixel])

        # logger.debug('{}{}', idx, pixel)
        try:
            # func = op.attrgetter(test)(mk)
            # # print(func)
            # # sys.exit()
            # # func = mk.seasonal_test
            # trend = func(pixel)

            if result is None:
                result = np.zeros([pixel.shape[0], *shape[1:]])

            for i in range(pixel.shape[0]):
                result[(i, *idx)] = pixel[i]  # con(trend.slope, trend.p)
        # except RuntimeWarning as e:
        #     print(e)
        #     tqdm.write('idx =', idx, 'test =', test)
        #     result[(0, *idx)] = 0
        #     result[(1, *idx)] = 0
        #     sys.exit()
        except Exception as e:
            logger.error("{}{}{}", e, idx, pixel)
    return result
