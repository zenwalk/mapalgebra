
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk
import sys
from loguru import logger
from compute import con
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

file_list = [
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/81_16_NDVI_weather_36years_180bands.tif',
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/01_16_MODIS_NDVI_wea_8km_16years_80bands.tif',
    # '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images4/81_00_GIMMS_NDVI_wea_8km_20years_100bands.tif',
]

float32_min = np.finfo(np.float32).min

names = [
    'ndvi',
    'v12001',
    'v13003',
    'v13004',
    'v13011'
]

layer_names = [
    'Average temperature (0.1℃)',  # °C
    'Average relative humidity (1%)',
    'Average water vapour pressure (0.1hPa)',
    'Average precipitation (0.1mm)',
]

abbr_years = {
    '81': 1981,
    '00': 2000,
    '01': 2001,
    '16': 2016,
}


def draw_fig(y, x, ax, postion_x, position_y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(slope, intercept, r_value, p_value, std_err)
    ax.plot(x, y, 'o', label='original data')
    # print(layer_names[idx])
    r2, p2 = stats.pearsonr(x, y)
    # ax.set_title("r = {:.2}, p = {:.2}".format(r2, p2))
    ax.plot(x, intercept + slope*x, 'r', label='fitted line')
    # ax.set_title('R = {:f}, p = {:f}'.format(r_value, p_value))
    # ax.text(postion_x, position_y,
    #         "slope = {:.2}, {}".format(slope, print_p(p_value)),
    #         fontsize=26,
    #         verticalalignment='center',
    #         transform=ax.transAxes)
    ax.text(postion_x, position_y,
            "r = {:.2}, p = {:.2}".format(r2, p2),
            fontsize=26,
            verticalalignment='center',
            transform=ax.transAxes)


def print_p(p):
    if p > 0.05:
        return 'p > 0.05'
    elif p <= 0.05 and 0.01 < p:
        return 'p < 0.05'
    else:
        return 'p < 0.01'


def draw_hist(data, ndvi, zone, idx1, idx2):
    res = []
    with np.nditer([data, ndvi], ['external_loop'], order='F') as it:
        for a, b in it:
            if b[0] == zone[0] or b[0] == zone[1]:
                pixel = a.reshape(5, -1)
                try:
                    r, p_value = stats.pearsonr(pixel[idx1], pixel[idx2])
                    res.append((r, p_value))
                except:
                    pass
    return res


def main(fn):

    import re
    [(begin_year, end_year)] = re.findall('(\d\d)_(\d\d)', fn)

    begin_year = abbr_years[begin_year]
    end_year = abbr_years[end_year]

    with rasterio.open(fn) as src:
        profile = src.profile
        data = src.read()
        print(fn)
        print(data.shape)
        year_count = int(data.shape[0] / len(names))

        print("year_count", year_count)

        # for idx, name in enumerate(names):

        #     trend = np.zeros(data.shape[1:])

        #     with np.nditer(trend, flags=['multi_index'], op_flags=['writeonly']) as it:
        #         for x in it:
        #             # print the progress for debug
        #             if it.multi_index[0] % 20 == 0 and it.multi_index[1] == 0:
        #                 print(it.multi_index)

        #             try:
        #                 vec = data[(slice(None), *it.multi_index)]
        #                 vec = vec[idx*year_count:(idx+1)*year_count]
        #                 if vec.mean() < 0.1:
        #                     x[...] = 0
        #                 else:
        #                     temp = mk.yue_wang_modification_test(vec)
        #                     x[...] = con(temp.slope, temp.p)
        #                 # slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(vec)), vec)
        #                 # x[...] = con(slope, p_value)
        #             except Exception as e:
        #                 # logger.error(e)
        #                 x[...] = 0

        #     profile.update(count=1, dtype=rasterio.uint8, nodata=0)

        #     with rasterio.open(fn.replace('.tif', '_trend_of_{}.tif'.format(name)), 'w', **profile) as dst:
        #         dst.write(trend.astype(np.uint8), 1)
        #         dst.write_colormap(1, {
        #             6: (56, 168, 0, 255),
        #             5: (121, 201, 0, 255),
        #             4: (206, 237, 0, 255),
        #             3: (255, 204, 0, 255),
        #             2: (255, 102, 0, 255),
        #             1: (255, 0, 0, 255),
        #             0: (255, 255, 255, 255)
        #         })
        #     break  # 只是计算ndvi的趋势

        # np.save('cache.npy', trend)

        # trend = np.load('cache.npy')
        ndvi = np.load('cache.npy')

        result12 = []
        result56 = []

        print('data[year_count:]', data[year_count:].shape)

        # fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(45, 30))
        # for i in range(5):
        #     for j in range(5):
        #         if i == j:
        #             continue
        #         ax = axs[i, j]
        #         hist_data = draw_hist(data, ndvi, zone=(1, 2), idx1=i, idx2=j)
        #         ax.hist([h[0] for h in hist_data])
        #         # ax.hist([h[1] for h in hist_data])

        # fig.savefig('demo.png')
        # sys.exit()

        for layer in data[year_count:]:

            tmp = np.concatenate((layer[ndvi == 1], layer[ndvi == 2]))
            tmp = tmp[tmp > float32_min]
            result12.append(np.mean(tmp))

            tmp = np.concatenate((layer[ndvi == 5], layer[ndvi == 6]))
            tmp = tmp[tmp > float32_min]
            result56.append(np.mean(tmp))

        result12 = np.array(result12).reshape(4, -1)
        result56 = np.array(result56).reshape(4, -1)

        print(result12.shape)
        print('result12', result12)

        print(result56.shape)
        print('result56', result56)

        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(60, 40))
        # fig.tight_layout(pad=3.0)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # plt.setp(
        #     axs,
        #     xticks=list(range(begin_year, end_year+1, 3)),
        #     # xticklabels=range(begin_year, end_year+1)
        # )

        it = np.nditer([data[:year_count], None], op_flags=[['readonly'], ['readwrite', 'allocate']])

        with it:
            for i, out in it:
                if i <= 0.1:
                    out[...] = np.nan
                elif i >= 1:
                    out[...] = 1
                else:
                    out[...] = i

            result_ndvi = it.operands[1]

        print(ndvi.shape)

        # result = np.zero(8)
        # for i, j in np.ndindex(data.shape[1:]):
        #     if ndvi[i, j] == 1 or ndvi[i, j] == 2:
        #         pixel = data[i, j]


        # print(result_ndvi)
        result_ndvi = np.nanmean(result_ndvi, axis=(1, 2))

        print('result_ndvi', result_ndvi)

        for idx in range(len(result12)):
            y = result12[idx]
            # x = range(len(y))
            # x = range(begin_year, end_year+1)
            x = result_ndvi
            y0 = y[:20]
            y1 = y[20:]
            x0 = x[:20]
            x1 = x[20:]

            draw_fig(x0, y0, axs[0, idx], 0.05, 0.9)
            axs[0, idx].set_xlabel(layer_names[idx], fontsize=26)
            axs[0, idx].set_ylabel('average NDVI', fontsize=26)

            draw_fig(x1, y1, axs[1, idx], 0.05, 0.9)
            axs[1, idx].set_xlabel(layer_names[idx], fontsize=26)
            axs[1, idx].set_ylabel('average NDVI', fontsize=26)
            # if p_value <= 0.05:
            #     axs[0, idx].spines['bottom'].set_color('red')
            #     axs[0, idx].spines['top'].set_color('red')
            #     axs[0, idx].spines['left'].set_color('red')
            #     axs[0, idx].spines['right'].set_color('red')

        for idx in range(len(result56)):
            y = result56[idx]
            # x = range(begin_year, end_year+1)
            x = result_ndvi
            y0 = y[:20]
            y1 = y[20:]
            x0 = x[:20]
            x1 = x[20:]

            draw_fig(x0, y0, axs[2, idx], 0.05, 0.9)
            axs[2, idx].set_xlabel(layer_names[idx], fontsize=26)
            axs[2, idx].set_ylabel('average NDVI', fontsize=26)

            draw_fig(x1, y1, axs[3, idx], 0.05, 0.9)
            axs[3, idx].set_xlabel(layer_names[idx], fontsize=26)
            axs[3, idx].set_ylabel('average NDVI', fontsize=26)

            # if p_value <= 0.05:
            #     axs[1, idx].spines['bottom'].set_color('red')
            #     axs[1, idx].spines['top'].set_color('red')
            #     axs[1, idx].spines['left'].set_color('red')
            #     axs[1, idx].spines['right'].set_color('red')

        fig.savefig(fn.replace('.tif', '_personr.png'))


for f in file_list:
    main(f)
