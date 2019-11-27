#!/usr/bin/python
# -*- coding:utf-8 -*-

# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split


digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
#######  3====>[0,0,0,1,0,0,0,0,0,0]    ###########
print ("Let's start fitting")
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print("y的真实值---->y的预测值:\n", confusion_matrix(y_test, predictions))
print("结果为：\n", classification_report(y_test, predictions))

