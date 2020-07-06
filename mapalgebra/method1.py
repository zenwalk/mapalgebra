
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


file_list = [
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images1/NDVi_weather_com_01_10_gimms.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images1/NDVi_weather_com_01_10_modis.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images1/NDVi_weather_com_11_16_modis.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images1/NDVi_weather_com_81_90_gimms.tif',
    '/Volumes/Samsung/onedrive/deskmini/mk/Relationship_analysis/composite_images1/NDVi_weather_com_91_00_gimms.tif',
]

float32_min = np.finfo(np.float32).min

abbr = {
    '01': 2001,
    '10': 2010,
    '11': 2011,
    '16': 2016,
    '81': 1981,
    '90': 1990,
    '91': 1991,
    '00': 2000,
}


def main(fn):

    import re
    [(begin_year, end_year)] = re.findall('(\d\d)_(\d\d)', fn)
    begin_year = abbr[begin_year]
    end_year = abbr[end_year]

    with rasterio.open(fn) as src:
        print(fn)
        data = src.read()
        print(data.shape)
        ndvi = data[0]

        result12 = []
        result56 = []

        for layer in data[1:]:

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

        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 15))

        print('begin_year', begin_year)
        print('end_year', end_year)

        plt.setp(
            axs,
            xticks=range(end_year-begin_year+1),
            xticklabels=range(begin_year, end_year+1)
        )

        for idx in range(len(result12)):
            y = result12[idx]
            x = range(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            print(slope, intercept, r_value, p_value, std_err)
            axs[0, idx].plot(x, y, 'o', label='original data')
            axs[0, idx].plot(x, intercept + slope*x, 'r', label='fitted line')
            axs[0, idx].set_title('R = {:f}, p = {:f}'.format(r_value, p_value))
            if p_value <= 0.05:
                axs[0, idx].spines['bottom'].set_color('red')
                axs[0, idx].spines['top'].set_color('red')
                axs[0, idx].spines['left'].set_color('red')
                axs[0, idx].spines['right'].set_color('red')

        for idx in range(len(result56)):
            y = result56[idx]
            x = range(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            print(slope, intercept, r_value, p_value, std_err)
            axs[1, idx].plot(x, y, 'o', label='original data')
            axs[1, idx].plot(x, intercept + slope*x, 'r', label='fitted line')
            axs[1, idx].set_title('R = {:f}, p = {:f}'.format(r_value, p_value))
            if p_value <= 0.05:
                axs[1, idx].spines['bottom'].set_color('red')
                axs[1, idx].spines['top'].set_color('red')
                axs[1, idx].spines['left'].set_color('red')
                axs[1, idx].spines['right'].set_color('red')

        fig.savefig(fn.replace('.tif', '.png'))


for f in file_list:
    main(f)
