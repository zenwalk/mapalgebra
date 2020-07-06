import riomucho
import rasterio
import numpy as np
import sys

from loguru import logger

from scipy import stats


def basic_run(datum, window, ij, g_args):
    print(window)
    # do something
    data = datum[0]
    # print(data.shape)
    out = np.zeros((8, *data.shape[1:]))
    for idx in np.ndindex(data.shape[1:]):
        # print(idx)
        pixel = data[(slice(None), *idx)]
        # # pixel[np.isnan(pixel)] = 0
        pixel = np.nan_to_num(pixel)
        pixel = pixel.reshape(5, -1)
        ndvi = pixel[0]
        for i, item in enumerate(pixel[1:]):
            try:
                r, p = stats.pearsonr(ndvi, item)
            except Exception as e:
                logger.info(e)
            out[(i*2, *idx)] = np.nan_to_num(r)
            out[(i*2+1, *idx)] = np.nan_to_num(p)

    return out #.astype(np.float32)


def main(fn):
    # get windows from an input
    with rasterio.open(fn) as src:
        # grabbing the windows as an example. Default behavior is identical.
        windows = [[window, ij] for ij, window in src.block_windows()]
        options = src.meta
        # since we are only writing to 2 bands
        options.update(count=8, dtype=rasterio.float64)

    global_args = {
        'divide': 2
    }

    processes = 4

    # run it
    with riomucho.RioMucho([fn], fn.replace('.tif', '.pearsonr.tif'), basic_run,
                           windows=windows,
                           global_args=global_args,
                           options=options) as rm:

        rm.run(processes)


file_list = [
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images2/NDVimages_wea_com_01_10_gimms.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images2/NDVimages_wea_com_01_10_modis.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images2/NDVimages_wea_com_11_16_modis.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images2/NDVimages_wea_com_81_90_gimms.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images2/NDVimages_wea_com_91_00_gimms.tif',
]

for fname in file_list:
    main(fname)
