# importing libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# loading data set
data = pd.read_excel('Online Retail (4).xlsx')
# pd.set_option('display.max_columns', None)
print(data)
print("********************************")
# Drop rows with null values
data = data.dropna(axis=0)
print(data)

# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

# Dropping the rows without any invoice number
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]
print(data)
print("*****")

# # Splitting the data according to the region of transaction
# # Transactions done in France
basket = (data[data['Country'] == "France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print("**France**")
print(basket)

# # Transactions done in the United Kingdom
basket = (data[data['Country'] == "United Kingdom"]
             .groupby(['InvoiceNo', 'Description'])['Quantity']
             .sum().unstack().reset_index().fillna(0)
             .set_index('InvoiceNo'))
print("**UK**")
print(basket)

# # Transactions done in Portugal
basket = (data[data['Country'] == "Portugal"]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
print("**Por**")
print(basket)
# Transactions done in the Sweden
basket = (data[data['Country'] == "Sweden"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print("**Sweden**")
print(basket)
# # Transactions done in the Germany
# basket = (data[data['Country'] == "Germany"]
#                  .groupby(['InvoiceNo', 'Description'])['Quantity']
#                  .sum().unstack().reset_index().fillna(0)
#                  .set_index('InvoiceNo'))
# print("**Germany**")
# print(basket)
# # Transactions done in the Australia
# basket = (data[data['Country'] == "Australia"]
#                  .groupby(['InvoiceNo', 'Description'])['Quantity']
#                  .sum().unstack().reset_index().fillna(0)
#                  .set_index('InvoiceNo'))
# print("**Australia**")
# print(basket)
# # Transactions done in the Norway
# basket = (data[data['Country'] == "Norway"]
#                  .groupby(['InvoiceNo', 'Description'])['Quantity']
#                  .sum().unstack().reset_index().fillna(0)
#                  .set_index('InvoiceNo'))
# print("**Norway**")
# print(basket)
# # Defining the hot encoding function to make the data suitable
# # for the concerned libraries
def hot_encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

my_basket_sets = basket.applymap(hot_encode)
my_basket_sets.drop('POSTAGE',inplace=True, axis=1)

# # Encoding the data sets
# basket_encoded = basket_France.applymap(hot_encode)
# basket_France = basket_encoded
#
# basket_encoded = basket_UK.applymap(hot_encode)
# basket_UK = basket_encoded
#
# basket_encoded = basket_Por.applymap(hot_encode)
# basket_Por = basket_encoded
#
# basket_encoded = basket_Sweden.applymap(hot_encode)
# basket_Sweden = basket_encoded

# # Building the model
frq_items = apriori(my_basket_sets, min_support=0.07, use_colnames=True)
# # Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
# displaying rules
print(rules)

# Making Recomendation
my_basket_sets['SET OF 60 PANTRY DESIGN CAKE CASES'].sum()
my_basket_sets['SET OF 3 CAKE TINS PANTRY DESIGN'].sum()
# filtering rules based on condition
rules[(rules['lift']>=3)&
(rules['confidence']>=0.3)]
print(rules)
