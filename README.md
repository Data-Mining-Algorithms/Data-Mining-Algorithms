# Data-Mining-Algorithms

This project was developed for the Data Mining module at Teesside University with the aim to demonstrate and evaluate the use of popular computational techniques for data mining. The project contains implementations of popular Data Mining techniques such as Sequential Pattern Mining, Association Rules Mining, Collaborative Filtering and a variety of datasets to test on.

## Datasets:
* [Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail) - 500k Records, times series transaction data. 
* [Groceries](https://www.kaggle.com/irfanasrullah/groceries) - 10k Records, customer recite data. 
* [MovieLens](https://grouplens.org/datasets/movielens/100k/) - 100k Rating Records from 1000 users on 1700 movies. 


## Output Format
##### Association Rules
```
{Precedent itemset}, sup(support count), rel sup(relative support %) ---> {Antecedent itemset}, sup(support count), rel sup(relative support %)- conf(confidence value)
```

##### Frequent itemsets
```
{Frequent itemset}, sup(support count)
```

## Results
### --------------------------  Groceries Dataset --------------------------

| Support | Relative Support | Confidence | Num. Itemsets | Num. Rules |
|---------|------------------|------------|---------------|------------|
| 63      | 2.5%             | 50%        | 688           | 52         |
| 251     | 10%              | 50%        | 86            | 0          |
| 251     | 10%              | 10%        | 86            | 66         |
| 126     | 5%               | 50%        | 288           | 5          |


## Contributors
[Aleksandra Petkova](https://github.com/aleksandra1617) - Association Rules Mining Algorithm (Core Python);

[Nour Aldin](https://github.com/NourAldinAlmubarak) - Association Rules Mining Algorithm (MLextend, Xlrd, Python), Collaborative Filtering;

[Victor Essien](https://github.com/vicrichy87) - Collaborative Filtering.
