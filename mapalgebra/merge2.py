import os
import sys


begin_year = 1981
end_year = 2017

for y in range(begin_year, end_year):
    print('rio calc "(take (read 1) 1)"', end=' ')
    for m in range(5, 10):
        fn = "{}_{:02}.tif".format(y, m)
        # print('fn', fn)
        # print('d', d)
        print(fn, end=' ')
    print("{}_agg59.tif".format(y))
