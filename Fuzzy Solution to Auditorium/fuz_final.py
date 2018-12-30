import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections


class Entry:
    def __init__(self, x1, x2, m, c, region):
        self.x1 = x1
        self.x2 = x2
        self.m = m
        self.c = c
        self.region = region
        

audience_edges = np.array([[[0, 0], [10, 1], [30, 1], [40, 1]],
                           [[30, 0], [40, 1], [60, 1], [70, 0]],
                           [[60, 0], [70, 1], [90, 1], [100, 0]]])

fan_edges = np.array([
    [[1, 1], [30, 0]],
    [[20, 0], [40, 1], [60, 1], [80, 0]],
    [[70, 0], [100, 1]]
])

noise_edges = np.array([
    [[1, 1], [30, 0]],
    [[20, 0], [40, 1], [60, 1], [80, 0]],
    [[70, 0], [100, 1]]
])


plt.axis([0, 105, 0, 1.2])

plt.figure(1)
for i in range(len(audience_edges)):  #iterator for different fuzzy sets
    plt.subplot(111)
    plt.ylim(0, 5)
    
    x_vals=[]
    y_vals=[]
    for j in range(len(audience_edges[i])):
        x_vals.append(audience_edges[i][j][0])
        y_vals.append(audience_edges[i][j][1])
    plt.plot(x_vals,y_vals)

plt.figure(2)
for i in range(len(fan_edges)):  #iterator for different fuzzy sets
    plt.subplot(111)
    plt.ylim(0, 5)
    x_vals=[]
    y_vals=[]
    for j in range(len(fan_edges[i])):
        x_vals.append(fan_edges[i][j][0])
        y_vals.append(fan_edges[i][j][1])
    plt.plot(x_vals,y_vals)


plt.figure(3)
for i in range(len(noise_edges)):  #iterator for different fuzzy sets
    plt.subplot(111)
    plt.ylim(0, 5)
    x_vals=[]
    y_vals=[]
    for j in range(len(noise_edges[i])):
        x_vals.append(noise_edges[i][j][0])
        y_vals.append(noise_edges[i][j][1])
    plt.plot(x_vals,y_vals)

# plt.show()


give_op_region=np.empty(27).reshape(3, 3, 3)
for i in range(3):
    for j in range(3):
        for k in range(3):
            if i+j+k<=1:
                give_op_region[i][j][k]=0
            elif i+j+k<=4:
                give_op_region[i][j][k] = 1
            else:
                give_op_region[i][j][k] = 2


audience_line_segment = []
fan_line_segment = []
noise_line_segment = []
for i in range(3):
    temp=[]
    if(i==0):
        temp=audience_edges
    elif i==1:
        temp=fan_edges
    else:
        temp=noise_edges
    seg_details = []
    for j in range(len(temp)):
        for k in range(len(temp[j]) - 1):
            slope = ((temp[j][k + 1][1] - temp[j][k][1]) * 1.0) / (
                (temp[j][k+1][0] - temp[j][k][0]))
            intercept = temp[j][k + 1][1] - slope * temp[j][k + 1][0]
            seg_details.append(Entry(temp[j][k][0], temp[j][k + 1][0], slope, intercept,j))
    if i == 0:
        audience_line_segment=seg_details[:]
    elif i == 1:
        fan_line_segment = seg_details[:]
    else:
        noise_line_segment = seg_details[:]

queries=int(input().strip())

for q in range(queries):

    audience = int(input())
    no_of_fans = int(input())
    noise_value = int(input())

    fuzzy_set_of_audience = [0, 0, 0]

    for audi in audience_line_segment:
        if audi.x1 <= audience <= audi.x2:
            membership_value = audience*audi.m + audi.c
            fuzzy_set_of_audience[audi.region] = membership_value

    fuzzy_set_of_fans = [0, 0, 0]

    for fan in fan_line_segment:
        if fan.x1 <= no_of_fans <= fan.x2:
            membership_value = no_of_fans*fan.m + fan.c
            fuzzy_set_of_audience[fan.region] = membership_value

    fuzzy_set_of_noise = [0, 0, 0]

    for noise in noise_line_segment:
        if noise.x1 <= noise_value <= noise.x2:
            membership_value = noise_value*noise.m + noise.c
            fuzzy_set_of_audience[noise.region] = membership_value

    #All the fuzzy sets are available

    height_of_op_tower=np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                height_of_op_tower[give_op_region[i][j][k]]=max(height_of_op_tower[give_op_region[i][j][k]],min(fuzzy_set_of_audience[i],fuzzy_set_of_fans[j],fuzzy_set_of_noise[k]))


                # Cutting height obtained
                #Calculating centroid height
                
    fuzzy_tower_coords = []
    mc_segment = []
    
    numerator = 0
    denominator = 0
    
    x_coords = []
    
    base_diff = 33
    base = [33, 66, 99]
        
    for pair in mc_segment:
        
        x = (pair.region - pair.c)/pair.m
        x_coords.append(x)
        
        if x > pair.x1:
            numerator += (pair.m / 3.0) * (x ** 3 - pair.x1 ** 3) + (pair.c/2.0)*(x ** 2 - pair.x1 ** 2)
        else:
            numerator += (pair.m / 3.0) * (pair.x2 ** 3 - pair.x ** 3) + (pair.c / 2.0) * (pair.x2 ** 2 - pair.x ** 2)
            
    for i in range(3):
        numerator += height_of_op_tower[i] * (x_coords[2 * i + 1] - x_coords[i])

    for i in range(3):
        denominator += height_of_op_tower[i] * ((x_coords[2 * i + 1] - x_coords[i]) + base_diff) / 2.0

    centroid = numerator/denominator
    
    print(centroid)
    
    if centroid < base[0]:
        print("Low")
    elif centroid < base[1]:
        print("Moderate")
    elif centroid < base[2]:
        print("High")

