import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import csv
with open('Data Repository\\groceries.csv')as f:
    reader_object = csv.reader(f)   # Returns a csv reader object which will have information about the file and data
    data = list(reader_object)   # This is a more secure way to extract the data from the reader object

    for row in data:
        print(row)

# data = pd.read_csv('groceries.csv')
# print(data)

# NON OF THIS IS NEEDED BECAUSE YOU DO NOT USE PANDAS!
# records = []
# for i in range(0, 7501):
    # "data" is a python list, there is no such attribute in the list as "values"
    # records.append([str(data.values[i, j]) for j in range(0, 20)])  # this --> data.values is the source of your bug.

association_rules = apriori(data, min_support=0.02, min_confidence=0.02, min_lift=3, min_length=3)  # Must be some issue here!
association_results = list(association_rules)                                                       # because this is an empty list..
#print(len(association_rules))
# print(association_rules[1])
print(association_rules)

for item in association_rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
