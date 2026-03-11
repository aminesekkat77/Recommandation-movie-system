import pandas as pd
import numpy as np
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# reading the data from the preprocessed .csv file
data = pd.read_csv('data.csv')

# making the new column containing combination of all the features
data['comb'] = (
    data['genres'].astype(str) + ' ' +
    data['genres'].astype(str) + ' ' +
    data['genres'].astype(str) + ' ' +
    data['director_name'].astype(str) + ' ' +
    data['director_name'].astype(str) + ' ' +
    data['actor_1_name'].astype(str) + ' ' +
    data['actor_2_name'].astype(str) + ' ' +
    data['actor_3_name'].astype(str)
)

# creating a count matrix
cv = TfidfVectorizer()
count_matrix = cv.fit_transform(data['comb'])

# creating a similarity score matrix
sim = cosine_similarity(count_matrix)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix', sim)

# saving dataframe to csv for later use in main file
data.to_csv('data.csv',index=False)