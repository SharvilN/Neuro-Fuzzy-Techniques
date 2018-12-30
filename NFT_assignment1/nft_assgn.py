from pandas import read_csv
import random
import numpy
import math


def sigmoid(x, alpha):
    return 1/(1+numpy.exp(-x))

data = read_csv('dataset.csv', sep='\t')
data = data.as_matrix()

for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] == 'yes':
            data[i][j] = 1
        elif data[i][j] == 'no':
            data[i][j] = 0
        else:
            data[i][j] = data[i][j][:2] + '.' + data[i][j][-1]
            data[i][j] = float(data[i][j])

print(data)

w = [[2*numpy.random.uniform()-1 for x in range(3)] for y in range(6)]
w = numpy.matrix(w)

v = [[2*numpy.random.uniform()-1 for i in range(2)] for j in range(3)]
v = numpy.matrix(v)

print(w)
print(v)

threshold = 0.1
learning_rate = 0.5

train_size = int(len(data)*0.2)
training_data = data[0:train_size]
test_data = data[train_size+1:]

error = 1

while error > threshold:

    error = 0
    for row in training_data:

        inp = numpy.transpose(row[:6])
        inp = numpy.matrix(inp)
        zin = inp * w

        print(zin)
        z = zin

        for i in range(zin.shape[1]):
            z[0, i] = sigmoid(zin[0, i], learning_rate)

        print(z)

        yin = z*v
        print(yin)

        y = yin
        for i in range(yin.shape[1]):
            y[0, i] = sigmoid(yin[0, i], learning_rate)

        error += 0.5*(y-row[6:])*(y-row[6:])

        for j in range(v.shape[0]):
            v[j, 0] -= learning_rate*(y[0, 0] - row[-2])*(sigmoid(yin[0, 0], learning_rate))*(1 - sigmoid(yin[0, 0], learning_rate))*z[0, j]
            v[j, 1] -= learning_rate*(y[0, 1] - row[-1])*(sigmoid(yin[0, 1], learning_rate))*(1 - sigmoid(yin[0, 1], learning_rate))*z[0, j]

        for i in range(w.shape[0]):






