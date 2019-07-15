'''
《Python深度学习》4.4.1
make DNN fitting y=x**2
'''

import numpy as  np
import pandas as pd
import random
import neuralpy

# Generate 50 random values, y = x**2
# random.seed
random.seed(2019)
sample_size = 50
sample = pd.Series(random.sample(
         range(-10000, 10000), sample_size))

x = sample/10000
y = x**2

# print(x.head(10))
# print(y.head(10))
# View information about the data: count mean std min 25% 50% 75% max
# print(x.describe())


count = 0
# dataSet is a list
dataSet = [([x.ix[count]]), ([y.ix[count]])]
count = 1

while (count<sample_size):
    print("Working on data item: ", count)
    dataSet = (dataSet + [([x.ix[count, 0]])], [([y.ix[count, 0]])])
    count = count + 1

fit = neuralpy.Network(1, 3, 7, 1)
epochs = 100
learning_rate = 1
print("fitting model right now")
fit.train(dataSet, epochs, learning_rate)

count = 0
pred = []

while (count<sample_size):
    out = fit.forward(x[count])
    print("Obs: ", count + 1,
          "y = ", round(y[count], 4),
          "prediction = ",
          round(pd.Series(out), 4))
    pred.append(out)
    count = count + 1
