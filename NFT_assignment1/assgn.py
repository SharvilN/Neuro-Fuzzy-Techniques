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

w = [[random.random() for x in range(2)] for y in range(6)]
w = numpy.matrix(w)

# v = [[random.random() for i in range(2)] for j in range(1)]
# v = numpy.matrix(v)

threshold = 0.1
learning_rate = 1

train_size = int(len(data)*0.2)
training_data = data[0:train_size]
test_data = data[train_size+1:]

for row in training_data:

    inp = numpy.transpose(row[:6])
    inp = numpy.matrix(inp)
    yin = inp*w

    for i in range(2):
        yin[0, i] = sigmoid(yin[0, i], learning_rate)

    print(" y : ", yin)

    # print("inp shape", inp.shape)
    # print("w shape", w.shape)
    # print("yin shape", yin.shape)

    # for i in range(len(yin)):
    #     for j in range(len(yin[0])):
    #         yin[i, j] = sigmoid(yin[i, j], learning_rate)

    # print(yin)
    updated = 1
    while updated:

        updated = 0

        if (yin[0, 0] > threshold and row[-2] == 0) or (yin[0, 0] < threshold and row[-2] == 1):
            updated = 1
            for i in range(len(w)):
                w[i, 0] += learning_rate*(row[-2] - yin[0, 0])*inp[0, i]

        if (yin[0, 1] > threshold and row[-1] == 0) or (yin[0, 1] < threshold and row[-1] == 1):
            updated = 1
            for i in range(len(w)):
                w[i, 1] += learning_rate*(row[-1] - yin[0, 1])*inp[0, i]

        yin = inp * w
        for i in range(2):
            yin[0, i] = sigmoid(yin[0, i], learning_rate)

print(w)

for row in test_data:

    inp = numpy.transpose(row[:6])
    inp = numpy.matrix(inp)
    yin = inp * w

    for j in range(2):
        yin[0, j] = sigmoid(yin[0, j], learning_rate)

    print("Predicted output : %s, Expected Output : %s", yin, row[-2:])






