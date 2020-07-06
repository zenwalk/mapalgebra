
# 根据180个字段的数据，生成ndvi与4种气象要素之间的
# 相关系数和p值

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk
import sys
from loguru import logger
from compute import con
from pylab import mpl
import riomucho
import rasterio
import numpy

np.seterr(divide='ignore', invalid='ignore')

file_list = [
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/81_16_NDVI_weather_36years_180bands.tif',
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/01_16_MODIS_NDVI_wea_8km_16years_80bands.tif',
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/81_00_GIMMS_NDVI_wea_8km_20years_100bands.tif',
]

colormap = {
    1: (252, 3, 1, 255),
    2: (247, 132, 2, 255),
    3: (255, 253, 166, 255),
    4: (223, 253, 118, 255),
    5: (47, 246, 3, 255),
    6: (15, 87, 2, 255),
}


def get_classify_pearsonr(x, y):
    try:
        r, p = stats.pearsonr(x, y)
    except Exception as e:
        return 0
    else:

        if p < 0.01:
            out = 1 if r < 0 else 6
        elif 0.01 <= p < 0.05:
            out = 2 if r < 0 else 5
        else:
            out = 3 if r < 0 else 4

        return out


def basic_run(datum, window, ij, g_args):
    print(window)
    data = datum[0]
    out = np.zeros((8, *data.shape[1:]))

    for i, j in np.ndindex(data.shape[1:]):
        pixel = data[:, i, j]
        pixel = pixel.reshape(5, -1)
        x = pixel[0]

        x0 = x[:20]
        x1 = x[20:]
        for b in range(1, 5):
            y = pixel[b]
            y0, y1 = y[:20], y[20:]

            out[(b-1)*2, i, j] = get_classify_pearsonr(x0, y0) if x0.mean() >= 0.1 else 0
            out[(b-1)*2+1, i, j] = get_classify_pearsonr(x1, y1) if x1.mean() >= 0.1 else 0

            # try:
            #     r0, p_value0 = stats.pearsonr(x0, y0)
            #     out[b*4-4+0, i, j] = r0
            #     out[b*4-4+1, i, j] = p_value0
            # except:
            #     out[b*4-4+0, i, j] = -9999
            #     out[b*4-4+1, i, j] = -9999

            # try:
            #     r1, p_value1 = stats.pearsonr(x1, y1)
            #     out[b*4-4+2, i, j] = r1
            #     out[b*4-4+3, i, j] = p_value1
            # except:
            #     out[b*4-4+2, i, j] = -9999
            #     out[b*4-4+3, i, j] = -9999

    return out.astype(np.int8)


# get windows from an input
with rasterio.open(file_list[0]) as src:
    # grabbing the windows as an example. Default behavior is identical.
    windows = [[window, ij] for ij, window in src.block_windows()]
    options = src.meta
    # since we are only writing to 2 bands
    options.update(count=8, nodata=0, dtype=np.int8)

global_args = {
    'divide': 2
}

processes = 4

# run it
with riomucho.RioMucho(file_list, 'ndvi_pearsonr.tif', basic_run,
                       windows=windows,
                       global_args=global_args,
                       options=options) as rm:

    rm.run(processes)

with rasterio.open('ndvi_pearsonr.tif', mode='r+') as src:
    for i in range(1, 9):
        src.write_colormap(i, colormap)

sys.exit()
# ------

read = {
    'increasing': 3,
    'decreasing': 1,
    'no trend': 2,
}


def basic_run2(datum, window, ij, g_args):
    print(window)

    data = datum[0]
    out = np.zeros((3, *data.shape[1:]))

    for i, j in np.ndindex(data.shape[1:]):
        pixel = data[:, i, j]
        pixel = pixel.reshape(5, -1)
        x = pixel[0]
        x0 = x[:20]
        x1 = x[20:]

        try:
            trend = mk.yue_wang_modification_test(x)
            print(trend)
            out[0] = read[trend.trend]
        except Exception as e:
            out[0] = 0

        continue

        try:
            trend = mk.yue_wang_modification_test(x0)
            out[1] = read[trend.trend]
        except Exception as e:
            out[1] = 0

        try:
            trend = mk.yue_wang_modification_test(x1)
            out[2] = read[trend.trend]
        except Exception as e:
            out[2] = 0

    return out.astype(np.int8)


# get windows from an input
with rasterio.open(file_list[0]) as src:
    # grabbing the windows as an example. Default behavior is identical.
    windows = [[window, ij] for ij, window in src.block_windows()]
    options = src.meta
    # since we are only writing to 2 bands
    options.update(count=3, nodata=0, dtype=np.int8)

global_args = {
    'divide': 2
}

processes = 4

# run it
with riomucho.RioMucho(file_list, 'ndvi_zones.tif', basic_run2,
                       windows=windows,
                       global_args=global_args,
                       options=options) as rm:

    rm.run(processes)
