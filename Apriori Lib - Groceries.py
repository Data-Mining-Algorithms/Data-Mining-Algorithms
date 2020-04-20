import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import csv
with open('groceries.csv')as f:
  data = csv.reader(next(f))
  for row in data:
        print(row)
# data = pd.read_csv('groceries.csv')
# print(data)

records = []
for i in range(0, 7501):
    records.append([str(data.values[i, j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.5, min_confidence=0.2, min_lift=3, min_length=3)
association_results = list(association_rules)
# print(len(association_rules))
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