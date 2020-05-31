
import os
from glob import glob

import numpy as np
import rasterio

color_map = {
    6: [56, 168, 0],
    5: [121, 201, 0],
    4: [206, 237, 0],
    3: [255, 204, 0],
    2: [255, 102, 0],
    1: [255, 0, 0],
    0: [255, 255, 255]
}


def main(infile, outfile):
    with rasterio.Env():
        with rasterio.open(infile) as src:
            profile = src.profile
            profile.update(count=3, dtype=rasterio.uint8,
                           nodata=0, driver='PNG')

            with rasterio.open(outfile, 'w', **profile) as dst:
                data = src.read(indexes=1)
                result = np.zeros([3, *data.shape])
                for idx in np.ndindex(data.shape):
                    pixel = data[idx]
                    color = color_map.get(pixel, [0, 0, 0])
                    result[(0, *idx)] = color[0]
                    result[(1, *idx)] = color[1]
                    result[(2, *idx)] = color[2]
                    # pixel[np.isnan(pixel)] = 0
                    # pixel = np.nan_to_num(pixel)
                dst.write(result.astype(rasterio.uint8))


if __name__ == "__main__":
    file_list = glob('./**/*.tif')
    for infile in file_list:
        outfile = infile.replace('.tif', '.png')
        main(infile, outfile)
