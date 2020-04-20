# importing libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# loading data set
data = pd.read_excel('Online Retail (4).xlsx')
print(data)
print("********************************")

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
basket_France = (data[data['Country'] == "France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print(basket_France)
print("**France**")
# # Transactions done in the United Kingdom
basket_UK = (data[data['Country'] == "United Kingdom"]
             .groupby(['InvoiceNo', 'Description'])['Quantity']
             .sum().unstack().reset_index().fillna(0)
             .set_index('InvoiceNo'))
print(basket_UK)
print("**UK**")
# # Transactions done in Portugal
basket_Por = (data[data['Country'] == "Portugal"]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
print(basket_Por)
print("**Por**")
basket_Sweden = (data[data['Country'] == "Sweden"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print(basket_Sweden)
print("**Sweden**")

# # Defining the hot encoding function to make the data suitable
# # for the concerned libraries
def hot_encode(x):
    if (x <= 0):
        return 0
    if (x >= 1):
        return 1

# # Encoding the data sets
basket_encoded = basket_France.applymap(hot_encode)
basket_France = basket_encoded

basket_encoded = basket_UK.applymap(hot_encode)
basket_UK = basket_encoded

basket_encoded = basket_Por.applymap(hot_encode)
basket_Por = basket_encoded

basket_encoded = basket_Sweden.applymap(hot_encode)
basket_Sweden = basket_encoded

# # Building the model
frq_items = apriori(basket_France, min_support=0.07, use_colnames=True)
# # Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
# displaying rules
print(rules)
