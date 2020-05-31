import operator as op
import os
import sys

import fiona
import numpy as np
import pymannkendall as mk
from loguru import logger
from shapely.geometry import Point, shape

BANDS = 1


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
            # try:
            #     trend = op.attrgetter(test)(mk)(vec)
            #     pbar.update(1)
            #     result[(0, *idx)] = con(trend.slope, trend.p)
            #     result[(1, *idx)] = numberic(trend.trend)
            #     result[(2, *idx)] = trend.p
            #     result[(3, *idx)] = trend.h
            # except RuntimeWarning as e:
            #     # print('idx =', idx, 'test =', test, sep=' ')
            #     result[(0, *idx)] = 0
            #     result[(1, *idx)] = 0
            #     result[(2, *idx)] = 0
            #     result[(3, *idx)] = 0
        return result
    return compute


selector = [
    #   1     2     3     4     5     6     7     8     9     10    11    12
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2000
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2001
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2002
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2003
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2004
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2005
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2006
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2007
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2008
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2009
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2010
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2011
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2012
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2013
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2014
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2015
    False, False, False, False, True, True, True, True, True, False, False, False,  # 2016
]


def compute(data, test, agg=None):
    shape = data.shape
    result = np.zeros([BANDS, *shape[1:]])
    for idx in np.ndindex(shape[1:]):
        pixel = data[(slice(None), *idx)]
        # pixel[np.isnan(pixel)] = 0
        pixel = np.nan_to_num(pixel)
        if agg:
            pixel = pixel.reshape(-1, 12)
            pixel = np.array([sum(row[agg[0]:agg[1]]) for row in pixel])
        # logger.debug('{}{}', idx, pixel)
        try:
            trend = op.attrgetter(test)(mk)(pixel)
            # trend1 = op.attrgetter(test)(mk)(pixel[120:240])
            # trend2 = op.attrgetter(test)(mk)(pixel[240:360])
            # trend3 = op.attrgetter(test)(mk)(pixel[360:480])
            # logger.log(idx, pixel)
            result[(0, *idx)] = con(trend.slope, trend.p)
            # result[(1, *idx)] = con(trend1.slope, trend1.p)
            # result[(2, *idx)] = con(trend2.slope, trend2.p)
            # result[(3, *idx)] = con(trend3.slope, trend3.p)
            # result[(1, *idx)] = numberic(trend.trend)
            # result[(0, *idx)] = pixel[0]
            # result[(1, *idx)] = pixel[1]
        # except RuntimeWarning as e:
        #     print(e)
        #     tqdm.write('idx =', idx, 'test =', test)
        #     result[(0, *idx)] = 0
        #     result[(1, *idx)] = 0
        #     sys.exit()
        except Exception as e:
            logger.error("{}{}{}", e, idx, pixel)
    return result
