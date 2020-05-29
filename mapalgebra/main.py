import pymannkendall as mk
import numpy as np
import rasterio
import concurrent.futures
import multiprocessing
import rasterio
from tqdm import tqdm
import operator as op
import itertools
import warnings
import os.path
from loguru import logger
import sys

logger.remove()
logger.add('main.log', format="{message}")

num_cores = multiprocessing.cpu_count()

BANDS = 2

np.seterr(all='warn')
warnings.filterwarnings('error')

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


def getCompute(test, pbar):
    print('test:', test)

    def compute(data):
        shape = data.shape
        # pbar and pbar.update(shape[1] * shape[2])
        print('shape', shape)
        result = np.zeros([BANDS, *shape[1:]])
        for idx in np.ndindex(shape[1:]):
            vec = data[(slice(None), *idx)]
            try:
                trend = op.attrgetter(test)(mk)(vec)
                pbar.update(1)
                result[(0, *idx)] = con(trend.slope, trend.p)
                result[(1, *idx)] = numberic(trend.trend)
                result[(2, *idx)] = trend.p
                result[(3, *idx)] = trend.h
            except RuntimeWarning as e:
                # print('idx =', idx, 'test =', test, sep=' ')
                result[(0, *idx)] = 0
                result[(1, *idx)] = 0
                result[(2, *idx)] = 0
                result[(3, *idx)] = 0
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


def compute(data, test):
    shape = data.shape
    result = np.zeros([BANDS, *shape[1:]])
    for idx in np.ndindex(shape[1:]):
        pixel = data[(slice(None), *idx)]
        # pixel[np.isnan(pixel)] = 0
        pixel = np.nan_to_num(pixel)
        # logger.debug("{}{}", idx, pixel)
        # pixel = pixel[selector]
        # pixel = pixel.reshape(-1, 12)
        # pixel = np.array([sum(row[4:9]) for row in pixel]) 
        try:
            trend = op.attrgetter(test)(mk)(pixel)
            # logger.log(idx, pixel)
            result[(0, *idx)] = con(trend.slope, trend.p)
            result[(1, *idx)] = numberic(trend.trend)
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


def multiply(obj):
    return obj.width * obj.height

wh = lambda obj: obj.width * obj.height

def main(infile, test='original_test', num_workers=4):
    outfile = os.path.basename(infile).replace('.tif', '_{}.tif'.format(test))
    with rasterio.Env():
        with rasterio.open(infile) as src:
            profile = src.profile
            profile.update(blockxsize=32, blockysize=32,
                           tiled=True, count=BANDS, dtype=rasterio.float64, nodata=0)

            pbar = tqdm(total=multiply(src), desc='{:.13}'.format(test))
            # compute = getCompute(test, pbar)
            # print('compute', compute)
            # random = np.arange(24).reshape(2,3,4)
            # print(compute(random))
            with rasterio.open(outfile, 'w', **profile) as dst:
                windows = [window for ij, window in dst.block_windows()]
                data_gen = (src.read(window=window) for window in windows)
                test_gen = itertools.repeat(test)
                # for window, data in zip(windows, data_gen):
                #     result = compute(data)
                #     dst.write(result.astype(rasterio.float64), window=window)
                # selector_gen = itertools.repeat(selector)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for window, result in zip(windows, executor.map(compute, data_gen, test_gen)):
                        pbar.update(multiply(window))
                        # print('window', window)
                        dst.write(result.astype(rasterio.float64), window=window)


if __name__ == "__main__":
    tests = [
        'yue_wang_modification_test',
    ]
    for test in tests:
        main('/Volumes/Samsung/Google Drive/NDVIMAX IMAGES/GIMMS_MAXNDVI_01_10.tif', test=test)
        main('/Volumes/Samsung/Google Drive/NDVIMAX IMAGES/GIMMS_MAXNDVI_81_90.tif', test=test)
        main('/Volumes/Samsung/Google Drive/NDVIMAX IMAGES/GIMMS_MAXNDVI_91_00.tif', test=test)
        main('/Volumes/Samsung/Google Drive/NDVIMAX IMAGES/MODIS_MAXNDVI_01_10.tif', test=test)
        main('/Volumes/Samsung/Google Drive/NDVIMAX IMAGES/MODIS_MAXNDVI_11_16.tif', test=test)
        
