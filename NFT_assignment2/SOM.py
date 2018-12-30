from pandas import read_csv
import numpy


def dist(weights, col, inp):
    d = 0
    for r in range(len(inp)):
        d += (weights[r][col] - inp[r])**2
    return d


def update_weights(weights, col, inp, learning_rate):
    for r in range(len(inp)):
        weights[r][col] += learning_rate*(inp[r] - weights[r][col])


data = read_csv('dataset.csv', sep='\t')
data = data.as_matrix()
numpy.random.shuffle(data)

learning_rate = 0.5

train = int(0.4*len(data))
training_data = data[:train]
testing_data = data[train+1:]

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

w = [[numpy.random.uniform() for i in range(4)] for j in range(6)]

print(w)

d = [0 for i in range(4)]

while learning_rate > 0.001:

    for row in training_data:
        inp = row[0:6]
        min_d = 0
        for i in range(4):
            d[i] = dist(w, i, inp)
            if d[min_d] > d[i]:
                min_d = i

        update_weights(w, min_d, inp, learning_rate)

    learning_rate /= 2

Map = [[0 for i in range(4)] for j in range(4)]

# 0 - 0 0
# 1 - 0 1
# 2 - 1 0
# 3 - 1 1

for row in training_data:

    inp = row[0:6]
    min_d = 0
    for i in range(4):
        d[i] = dist(w, i, inp)
        if d[min_d] > d[i]:
            min_d = i

    clss = 2*row[6] + row[7]
    Map[clss][min_d] += 1

final_mapping = [0 for i in range(4)]

for i in range(4):
    maxx = 0
    for j in range(4):

        if maxx < Map[i][j]:
            maxx = Map[i][j]
            final_mapping[i] = j


ct = 0
success_ct = 0

for row in testing_data:
    inp = row[0:6]

    min_d = 0
    for i in range(4):
        d[i] = dist(w, i, inp)
        if d[min_d] > d[i]:
            min_d = i

    if final_mapping[2*row[6] + row[7]] == min_d:
        # print("Correct Output")
        success_ct += 1

    ct += 1

print("Efficiency = ", success_ct/ct*100)

# print(w)




