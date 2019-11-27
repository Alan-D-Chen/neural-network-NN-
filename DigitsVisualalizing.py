# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 上午 12:36
# @Author  : Alan D. Chen
# @FileName: DigitsVisualalizing.py
# @Software: PyCharm

import pylab as pl
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

pl.gray()
for i in range(digits.data.shape[0]):
    pl.matshow(digits.images[i])

pl.show()