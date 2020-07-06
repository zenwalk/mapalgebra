
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

np.seterr(all='raise', over='ignore')

float32_min = np.finfo(np.float32).min

file_list = [
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/180test.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/81_16_NDVI_weather_36years_180bands.tif',
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/01_16_MODIS_NDVI_wea_8km_16years_80bands.tif',
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/81_00_GIMMS_NDVI_wea_8km_20years_100bands.tif',
]

read = {
    'increasing': 3,
    'decreasing': 1,
    'no trend': 2,
}


def basic_run2(datum, window, ij, g_args):
    print(window)

    data = datum[0]
    out = np.zeros((12, *data.shape[1:]))
    out.fill(float32_min)

    for i, j in np.ndindex(data.shape[1:]):
        pixel = data[:, i, j]
        pixel = pixel.reshape(5, -1)
        x = pixel[0]

        if np.nanmean(x) < 0.1:
            continue

        # if np.nanmean(x) < 0.1:
        #     continue
        x0 = x[:20]
        x1 = x[20:]

        try:
            trend = mk.yue_wang_modification_test(x0)

            out[0, i, j] = trend.slope
            out[1, i, j] = trend.p
            out[2, i, j] = con(trend.slope, trend.p)
        except Exception as e:
            pass

        try:
            trend = mk.yue_wang_modification_test(x1)
            out[3, i, j] = trend.slope
            out[4, i, j] = trend.p
            out[5, i, j] = con(trend.slope, trend.p)
        except Exception as e:
            pass

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(x0)), x0)
            out[6, i, j] = slope
            out[7, i, j] = p_value
            out[8, i, j] = con(slope, p_value)
            # print(out[8, i, j])
        except Exception as e:
            print(e)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(x1)), x1)
            out[9, i, j] = slope
            out[10, i, j] = p_value
            out[11, i, j] = con(slope, p_value)
        except Exception as e:
            print(e)
    return out.astype(np.float32)


# get windows from an input
with rasterio.open(file_list[0]) as src:
    # grabbing the windows as an example. Default behavior is identical.
    windows = [[window, ij] for ij, window in src.block_windows()]
    options = src.meta
    # since we are only writing to 2 bands
    options.update(count=12, nodata=float32_min, dtype=np.float32)

global_args = {
    'divide': 2
}

print(options)
processes = 4

# run it
# with riomucho.RioMucho(file_list, 'ndvi_slope_p_class.tif', basic_run2,
#                        windows=windows,
#                        global_args=global_args,
#                        options=options) as rm:

#     rm.run(processes)

with rasterio.open(file_list[0]) as src:
    with rasterio.open('ndvi_slope_p_class.tif', 'w', **options) as dst:
        data = src.read()
        out = basic_run2([data], None, None, None)

        dst.write(out)
