from pandas import read_csv
import random
import numpy
import math


def sigmoid(x, alpha):
    return 1/(1+math.exp(-alpha*x))

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

# print(data)

w = [[random.random() for x in range(1)] for y in range(6)]
w = numpy.matrix(w)


v = [[random.random() for i in range(2)] for j in range(1)]
v = numpy.matrix(v)


# print(w)
# print(v)

threshold = 0.1
learning_rate = 1

train_size = int(len(data)*0.2)
training_data = data[0:train_size]
test_data = data[train_size+1:]

for row in training_data:

    inp = numpy.transpose(row[:6])
    inp = numpy.matrix(inp)
    zin = inp*w

    for i in range(len(zin)):
        for j in range(len(zin[0])):
            zin[i, j] = sigmoid(zin[i, j], learning_rate)

    # print(zin)

    yin = zin*v

    for i in range(len(yin)):
        for j in range(len(yin[0])):
            yin[i, j] = sigmoid(yin[i, j], learning_rate)

    print(yin)

    if (yin[0, 0] > threshold and row[-2] == 0) or (yin[0, 0] < threshold and row[-2] == 1):
        updated = 1
        for i in range(len(v)):
            v[i, 0] += learning_rate*(row[-2] - yin[0, 0])*zin[0, i]

        for i in range(len(w)):
            for j in range(len(zin)):
                w[i, j] += learning_rate*(zin[0, j] - )

    if (yin[0, 1] > threshold and row[-1] == 0) or (yin[0, 1] < threshold and row[-1] == 1):
        updated = 1
        for i in range(len(v)):
            v[i, 1] += learning_rate*(row[-2] - yin[0, 0])*zin[0, i]






