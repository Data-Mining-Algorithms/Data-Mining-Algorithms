# importing libraries
import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD
from surprise.model_selection import cross_validate
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# to load dataset from pandas df
ratings = pd.read_csv('Data Repository\\movies_smalltest.csv')
print(ratings)
# Read data into an array of strings
ratings_dict = {'users': list(ratings.users),
                'movies': list(ratings.movies),
                'rating': list(ratings.rating)}
df = pd.DataFrame(ratings_dict)
print(df)
# create ratings_mean_count dataframe and first add the average rating of each movie to this dataframe:
ratings_mean_count = pd.DataFrame(ratings.groupby('movies')['rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(ratings.groupby('movies')['rating'].count())
print(ratings_mean_count)
# plot a histogram for the number of ratings represented by the "rating_counts"
sns.set_style('dark')
# matplotlib inline

plt.figure(figsize=(8,6))
plt.title('number of ratings represented by the "rating_counts"')
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)
plt.show()
# plot a histogram for average ratings

plt.figure(figsize=(8,6))
plt.title('average ratings')
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating'].hist(bins=50)
plt.show()
# pivot ratings into movie features
df_movie_features = ratings.pivot(
    index='movies',
    columns='users',
    values='rating'
).fillna(0)
mat_movie_features = csr_matrix(df_movie_features.values)
print(df_movie_features)
# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(0, 5.0))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['users', 'movies', 'rating']], reader)

# Split data into 5 folds
# data.split(n_folds=5)
# Split the dataset into 5 folds and choose the algorithm
algo = SVD()
# Train and test reporting the RMSE and MAE scores
# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict a certain item
users = str(414)
movies = str(410)
actual_rating = 5
print(algo.predict(users, movies, actual_rating))
