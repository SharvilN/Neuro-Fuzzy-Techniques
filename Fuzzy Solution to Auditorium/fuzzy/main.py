import matplotlib.pyplot as plt
import numpy as np


class Entry:
    def __init__(self, x1, x2, m, c, region):
        self.x1 = x1
        self.x2 = x2
        self.m = m
        self.c = c
        self.region = region

    def __str__(self):
        ret="{} {} {} {} {}\n".format(self.x1,self.x2,self.m,self.c,self.region)
        return ret

    def __repr__(self):
        ret="{} {} {} {} {}\n".format(self.x1,self.x2,self.m,self.c,self.region)
        return ret


# class FinalGraphCords:
#     def __init__(self, x_cord, y_cord):
#         self.x = x_cord
#         self.y = y_cord


audience_edges = np.array([[[0, 0], [10, 1], [30, 1], [40, 0]],
                           [[30, 0], [40, 1], [60, 1], [70, 0]],
                           [[60, 0], [70, 1], [90, 1], [100, 0]]])

fan_edges = np.array([
    [[0, 1], [30, 0]],
    [[20, 0], [40, 1], [60, 1], [80, 0]],
    [[70, 0], [100, 1]]
])

noise_edges = np.array([
    [[0, 1], [30, 0]],
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

plt.show()

# give_op_region=np.dtype(np.int32)
a_temp=np.ndarray(27).reshape(3,3,3)
give_op_region=np.zeros_like(a_temp, dtype=np.int32)
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
        temp=audience_edges[:]
    elif i==1:
        temp=fan_edges[:]
    else:
        temp=noise_edges[:]
    seg_details = []
    for j in range(len(temp)):
        for k in range(len(temp[j]) - 1):
            slope = ((temp[j][k + 1][1] - temp[j][k][1]) * 1.0) / (
                (temp[j][k+1][0] - temp[j][k][0]))
            intercept = temp[j][k + 1][1] - slope * temp[j][k + 1][0]
            seg_details.append(Entry(temp[j][k][0], temp[j][k + 1][0], slope, intercept,j))
    if i == 0:
        audience_line_segment=seg_details[:]
    elif i== 1:
        fan_line_segment = seg_details[:]
    else:
        noise_line_segment = seg_details[:]

# print(audience_line_segment)
# print(fan_line_segment)
# print(noise_line_segment)

print("Enter the number of queries : ")
queries=int(input().strip())

for q in range(queries):

    print("Enter the head count of audience : ")
    audience = int(input())
    print("Enter the number of fans available in the auditorium : ")
    no_of_fans = int(input())
    print("Enter the noise value")
    noise_value = int(input())

    fuzzy_set_of_audience = [0, 0, 0]

    for audi in audience_line_segment:
        if audi.x1 <= audience <= audi.x2:
            membership_value = audience*audi.m + audi.c
            fuzzy_set_of_audience[audi.region] = membership_value

    # print(fuzzy_set_of_audience)

    fuzzy_set_of_fans = [0, 0, 0]

    for fan in fan_line_segment:
        if fan.x1 <= no_of_fans <= fan.x2:
            # print("In fan {}".format(no_of_fans))
            # print(fan)
            membership_value = no_of_fans*fan.m + fan.c
            # print(membership_value)
            fuzzy_set_of_fans[fan.region] = membership_value

    # print(fuzzy_set_of_fans)

    fuzzy_set_of_noise = [0, 0, 0]

    for noise in noise_line_segment:
        if noise.x1 <= noise_value <= noise.x2:
            # print("In noise {}".format(noise_value))
            # print(noise)
            membership_value = noise_value*noise.m + noise.c
            # print(membership_value)
            fuzzy_set_of_noise[noise.region] = membership_value

    # print(fuzzy_set_of_noise)

    #All the fuzzy sets are available

    height_of_op_tower = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                height_of_op_tower[give_op_region[i][j][k]]=max(height_of_op_tower[give_op_region[i][j][k]],min(fuzzy_set_of_audience[i],fuzzy_set_of_fans[j],fuzzy_set_of_noise[k]))


                # Cutting height obtained
                #Calculating centroid height
    ################# height of the tower is zero which is a problem ######
    # for i in height_of_op_tower:
    #     print("{} ".format(i))
    # print("")
    fuzzy_tower_coords = np.array([
        [[0, 0], [16.5, 1], [33, 0]],
        [[33, 0], [49.5, 1], [66, 0]],
        [[66, 0], [82.5, 1], [99, 0]],
    ])

    m_c_segment = []
    for j in range(len(fuzzy_tower_coords)):
        for k in range(len(fuzzy_tower_coords[j])-1):
            slope = ((fuzzy_tower_coords[j][k + 1][1] - fuzzy_tower_coords[j][k][1]) * 1.0) / (
                (fuzzy_tower_coords[j][k + 1][0] - fuzzy_tower_coords[j][k][0]))
            intercept = fuzzy_tower_coords[j][k + 1][1] - slope * fuzzy_tower_coords[j][k + 1][0]
            m_c_segment.append(Entry(fuzzy_tower_coords[j][k][0], fuzzy_tower_coords[j][k + 1][0], slope, intercept, j))

    # print(m_c_segment)

    numerator = 0
    denominator = 0

    x_coords = []

    base_diff = 33
    base = [33, 66, 99]

    final_graph_x_cords = []
    final_graph_y_cords = []

    odd = 1
    for pair in m_c_segment:

        x = (height_of_op_tower[pair.region] - pair.c)/pair.m
        x_coords.append(x)

        if odd == 1:
            numerator += ((pair.m / 3.0) * (x ** 3 - pair.x1 ** 3)) + ((pair.c / 2.0) * (x ** 2 - pair.x1 ** 2))
            denominator += ((pair.m / 2.0) * (x ** 2 - pair.x1 ** 2)) + ((pair.c) * (x - pair.x1))

            # for drawing final graph
            final_graph_x_cords.append(pair.x1)
            final_graph_x_cords.append(x)
            final_graph_y_cords.append(0)
            final_graph_y_cords.append(height_of_op_tower[pair.region])

        else:
            numerator += ((pair.m / 3.0) * (pair.x2 ** 3 - x ** 3)) + ((pair.c / 2.0) * (pair.x2 ** 2 - x ** 2))
            denominator += ((pair.m / 2.0) * (pair.x2 ** 2 - x ** 2)) + ((pair.c) * (pair.x2 - x))

            # for drawing final graph
            final_graph_x_cords.append(x)
            final_graph_x_cords.append(pair.x2)
            final_graph_y_cords.append(height_of_op_tower[pair.region])
            final_graph_y_cords.append(0)

        # print("{} {} {} {} {}".format(pair.x1, x, pair.x2,numerator,denominator))
        odd=1-odd
    # print(numerator)
    for i in range(3):
        numerator += height_of_op_tower[i] * (((x_coords[2 * i + 1]**2) - (x_coords[2*i]**2))/2.0)
        denominator += height_of_op_tower[i] * (x_coords[2 * i + 1] - x_coords[2*i])

    print("{} {}".format(numerator,denominator))
    centroid = (numerator*1.0)/denominator

    print("heights of tower :", height_of_op_tower)

    if centroid < base[0]:
        print("Answer : Low, ",end="")
    elif centroid < base[1]:
        print("Answer : Moderate, ",end="")
    else:
        print("Answer : High, ",end="")

    print(" Centroid : {:.2f}, Audience : {}, No_of_fans : {}, Noise value : {}".format(centroid,audience,no_of_fans,noise_value))

    plt.plot(final_graph_x_cords, final_graph_y_cords)
    plt.ylim(0, 5)
    plt.show()
