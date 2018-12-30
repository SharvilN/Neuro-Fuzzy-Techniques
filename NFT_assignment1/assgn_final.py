from pandas import read_csv
import random
import numpy
import math

[][]
def sigmoid(x, alpha):
    return 1/(1+numpy.exp(alpha*(-x)))

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
            if data[i][j] < 38:
                data[i][j] = 0
            else:
                data[i][j] = 1

print(data)

# 6 4 2

w = [[2*numpy.random.uniform()-1 for i in range(4)] for j in range(6)]
v = [[2*numpy.random.uniform()-1 for i in range(2)] for j in range(4)]

zin = [0 for i in range(4)]
yin = [0 for i in range(2)]

z = [0 for i in range(4)]
y = [0 for i in range(2)]

print("w : ", w)
print("v : ", v)

threshold = 0.1
learning_rate = 1

bz = [2*numpy.random.uniform()-1 for i in range(4)]
by = [2*numpy.random.uniform()-1 for i in range(2)]

print("bz : ", bz)
print("by : ", by)

train_size = int(len(data)*0.3)
training_data = data[0:train_size]
test_data = data[train_size+1:]

train_data = []
train_data_indices = []

for a in range(train_size):
    index = random.randint(0, len(data) - 1)
    train_data_indices.append(index)
    train_data.append(data[index])

print("train_data_indices : ", train_data_indices)
print("train_data : ", train_data)

for itr in range(1000):

    error = 0

    for row in train_data:

        inp = numpy.transpose(row[:6])

        for j in range(len(zin)):
            zin[j] = 0
            for i in range(len(inp)):
                zin[j] += inp[i]*w[i][j]
            zin[j] += bz[j]
            z[j] = sigmoid(zin[j], learning_rate)

        for k in range(len(yin)):
            yin[k] = 0
            for j in range(len(z)):
                yin[k] += z[j]*v[j][k]
            yin[k] += by[k]
            y[k] = sigmoid(yin[k], learning_rate)

        for j in range(len(z)):
            v[j][0] -= (y[0] - row[-2])*y[0]*(1 - y[0])*z[j]
            v[j][1] -= (y[1] - row[-1])*y[1]*(1 - y[1])*z[j]

        sigma1 = 0
        sigma2 = 0

        for j in range(len(z)):
            sigma1 += (y[0] - row[-2])*y[0]*(1 - y[0])*v[j][0]
            sigma2 += (y[1] - row[-1])*y[1]*(1 - y[1])*v[j][1]

        sigma = sigma1 + sigma2

        for i in range(len(inp)):
            for j in range(len(z)):
                w[i][j] -= sigma*z[j]*(1 - z[j])*inp[i]

        by[0] -= (y[0] - row[-2]) * y[0] * (1 - y[0])
        by[1] -= (y[1] - row[-1]) * y[1] * (1 - y[1])

        for j in range(len(z)):
            bz[j] -= sigma * z[j] * (1 - z[j])

        # print("W : ")
        # for i in range(6):
        #     for j in range(4):
        #         print(w[i][j])

        error += 0.5 * (y[0] - row[-2]) ** 2 + 0.5 * (y[1] - row[-1]) ** 2
    print("error : ", error)


print("w:", w)
print("v:", v)

ct = 0

for itr in range(len(data)):

    if itr not in train_data_indices:

        row = data[itr]
        inp = numpy.transpose(row[:6])

        for j in range(len(zin)):
            zin[j] = 0
            for i in range(len(inp)):
                zin[j] += inp[i] * w[i][j]
            zin[j] += bz[j]
            z[j] = sigmoid(zin[j], learning_rate)

        for k in range(len(yin)):
            yin[k] = 0
            for j in range(len(z)):
                yin[k] += z[j] * v[j][k]
            yin[k] += by[k]
            y[k] = sigmoid(yin[k], learning_rate)
            if y[k] < threshold:
                y[k] = 0
            else:
                y[k] = 1
        if y[0] != row[-2] or y[1] != row[-1]:
            ct += 1
        print("Exp : ", row[6:], " Pred : ", y)

print("Count of wrong predictions : ", ct)





