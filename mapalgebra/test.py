import functools
import operator
from time import sleep

import numpy as np
from tqdm import tqdm

res = np.empty((500, 200))


def mul(x):
    return functools.reduce(operator.mul, x)


with np.nditer(res, flags=['multi_index', 'c_index'], op_flags=['writeonly']) as it:
    # with tqdm(total=10000*10000) as pbar:
    total = mul(res.shape)
    step_size = total // 50
    pbar = tqdm(total=total)
    for x in it:
        # if it.multi_index[1] == 0:
        #     pbar.update(10000)
        if it.index % step_size == 0:
            pbar.update(step_size)
        sleep(0.0001)
        x[...] = np.random.rand()
    pbar.close()
