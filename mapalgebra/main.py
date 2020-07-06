import concurrent.futures
import itertools
import multiprocessing
import operator as op
import os.path
import sys
import warnings

import numpy as np
import pymannkendall as mk
import rasterio
from loguru import logger
from tqdm import tqdm

from compute import compute

logger.remove()
logger.add('main.log', format="{message}")

num_cores = multiprocessing.cpu_count()

BANDS = 36

np.seterr(all='warn')
warnings.filterwarnings('error')


def multiply(obj):
    return obj.width * obj.height


def wh(obj): return obj.width * obj.height


def get_bands_of_month(m, band_count, begin_year, end_year) -> list:
    assert(m == None or 1 <= m and m <= 12)
    if m == None:
        if begin_year == None and end_year == None:
            return None
        else:
            return list(range((begin_year-1981)*12+1, (end_year-1981+1)*12+1))
    else:
        band_list = list(range(m, band_count+1, 12))
        if begin_year == None and end_year == None:
            return band_list
        else:
            return list(map(lambda x: x[1], list(filter(lambda x: begin_year <= x[0] <= end_year, zip(range(1981, 2016 + 1), band_list)))))


def main(infile, begin_year, end_year, month=None, agg=None, test='original_test', num_workers=4):
    outfile = os.path.basename(infile).replace('.tif', '_M{}_{}_{}_Y{}_Y{}.tif'.format(month or '', agg or '', test, begin_year, end_year))
    # print(outfile)
    with rasterio.Env():
        with rasterio.open(infile) as src:
            profile = src.profile

            BANDS = end_year - begin_year + 1 # profile['count'] / 12

            profile.update(blockxsize=32, blockysize=32,
                           tiled=True, count=BANDS, dtype=rasterio.uint8, nodata=0)

            pbar = tqdm(total=multiply(src), desc='{:.50}'.format(test))
            # print('compute', compute)
            # random = np.arange(24).reshape(2,3,4)
            # print(compute(random))

            selected_bands = get_bands_of_month(
                month, src.count, begin_year, end_year)

            print(selected_bands)

            with rasterio.open(outfile, 'w', **profile) as dst:
                windows = [window for ij, window in dst.block_windows()]
                data_gen = (src.read(window=window, indexes=selected_bands) for window in windows)
                # data_gen = (src.read(window=window) for window in windows)
                test_gen = itertools.repeat(test)
                agg_gen = itertools.repeat(agg)
                # for window, data in zip(windows, data_gen):
                #     result = compute(data)
                #     dst.write(result.astype(rasterio.float64), window=window)
                # selector_gen = itertools.repeat(selector)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for window, result in zip(windows, executor.map(compute, data_gen, test_gen, agg_gen)):
                        pbar.update(multiply(window))
                        pbar.write('window {}'.format(window))
                        dst.write(result.astype(rasterio.uint8), window=window)
                # dst.write_colormap(1, {
                #     6: (56, 168, 0, 255),
                #     5: (121, 201, 0, 255),
                #     4: (206, 237, 0, 255),
                #     3: (255, 204, 0, 255),
                #     2: (255, 102, 0, 255),
                #     1: (255, 0, 0, 255),
                #     0: (255, 255, 255, 255)
                # })


if __name__ == "__main__":
    tests = [
        'yue_wang_modification_test',
        # 'original_test',
        # 'seasonal_test '
    ]

    year_range_list = [
        # [1981, 2016],
        [1981, 1990],
        [1991, 2000],
        [2001, 2010],
        [2011, 2016]
    ]

    file_list = [
        # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/NDVImax_images/GIMMS_MAXNDVI_01_10_Pro8km.tif',
        # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/NDVImax_images/GIMMS_MAXNDVI_81_90_Pro8km.tif',
        # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/NDVImax_images/GIMMS_MAXNDVI_91_00_Pro8km.tif',
        # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/NDVImax_images/MODIS_MAXNDVI_01_10_Pro8km.tif',
        # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/NDVImax_images/MODIS_MAXNDVI_11_16_pro8km.tif',
        '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/weather_inter/V12001.tif',
        '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/weather_inter/V13003.tif',
        '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/weather_inter/V13004.tif',
        '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/weather_inter/V13011.tif',
    ]

    for f in file_list:
        for year_range in year_range_list:
            #main(infile=fn, agg=None, begin_year=year_range[0], end_year=year_range[1], test=test, num_workers=num_cores)
            main(infile=f, agg=[4, 9], begin_year=year_range[0], end_year=year_range[1], test='merge_5-9', num_workers=num_cores)



    # for fn in file_list:
    #     for test in tests:
    #         for year_range in year_range_list:
    #             main(infile=fn, agg=None, begin_year=year_range[0], end_year=year_range[1], test=test, num_workers=num_cores)
               # main(infile=fn, agg=[4, 9], begin_year=year_range[0], end_year=year_range[1], test=test, num_workers=num_cores)
                # for month in range(1, 13):
                #     main(infile=fn, month=month,
                #          begin_year=year_range[0], end_year=year_range[1], test=test, num_workers=num_cores)
