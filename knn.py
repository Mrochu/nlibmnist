import csv
import math
from collections import Counter

training_data = []

training_data_path = ''

k = 200

with open(training_data_path,'r') as file:
    reader = csv.reader(file)
    for row in reader:
        training_data.append([int(i) for i in row])
        pass

def knn(new):
    dist_list = []
    #귀찮으니까 정렬은 대충하
    for i in training_data:
        distance = euclid_dis(new,i[1:])
        dist_list.append([i[0],distance])
        if len(dist_list) > k:
            dist_list.sort(key=lambda x:x[1])
            del dist_list[k]

    labels = [i[0] for i in dist_list]
    return Counter(labels).most_common(1)[0][0]

def euclid_dis(new,old):
    dist = 0
    for i in range(len(new)):
        dist = dist + (new[i] - old[i])**2
    return math.sqrt(dist)

        
