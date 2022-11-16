# 2) Use CC_GENERAL.csv given in the folder and apply:
#    a) Preprocess the data by removing the categorical column and filling the missing values.
#    b) Apply StandardScaler() and normalize() functions to scale and normalize raw input data.
#    c) Use PCA with K=2 to reduce the input dimensions to two features.
#    d) Apply Agglomerative Clustering with k=2,3,4 and 5 on reduced features and visualize
#       result for each k value using scatter plot.
#    e) Evaluate different variations using Silhouette Scores and Visualize results with a bar chart.

import math
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

warnings.filterwarnings('ignore')


# cc_general is a dataframe where we load the csv data.
cc_general = pd.read_csv("C:/Users/Kiran Kumar Kongari/PycharmProjects/ML-Assignment-6/Dataset/CC GENERAL.csv")
print("\nThe Original Dataframe is : \n", cc_general)

# a) Preprocess the data by removing the categorical column and filling the missing values.
# dropping the categorical column i.e., CUST_ID column
customerDf = cc_general.drop(['CUST_ID'], axis='columns')

# Checking the columns having null values and displaying the resultant columns.
columnsWithNullValues = customerDf.isna().any()

# a. Replacing the null values with the mean
customerDf['CREDIT_LIMIT'] = customerDf['CREDIT_LIMIT'].fillna(customerDf['CREDIT_LIMIT'].mean())
customerDf['MINIMUM_PAYMENTS'] = customerDf['MINIMUM_PAYMENTS'].fillna(customerDf['MINIMUM_PAYMENTS'].mean())

# Verifying the dataframe again for null values
f = customerDf[customerDf.isna().any(axis=1)]
print('\nVerifying customer dataframe for null values again : ', f)

# -----------------------------------------------------------------------------------------------------------------------------
# b) Apply StandardScaler() and normalize() functions to scale and normalize raw input data.
# Performing Scaling
scaler = StandardScaler()
X_Scale = scaler.fit_transform(customerDf)

# Normalizing the data so that the data approximately
X_normalize = normalize(X_Scale)

# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalize)
print("\n The dataframe after performing the Scaling and normalizing is : \n ", X_normalized)

# -----------------------------------------------------------------------------------------------------------------------------
# c) Use PCA with K=2 to reduce the input dimensions to two features.
# Applying PCA (k=2)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_normalized)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2'])
print("\nThe Dataframe after applying PCA : \n", principalDf)

# -----------------------------------------------------------------------------------------------------------------------------
# d) Apply Agglomerative Clustering with k=2,3,4 and 5 on reduced features and visualize
#    result for each k value using scatter plot.

# With k=2
ac2 = AgglomerativeClustering(n_clusters=2)
# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
            c=ac2.fit_predict(principalDf), cmap='rainbow')
plt.show()

# With k=3
ac3 = AgglomerativeClustering(n_clusters=3)
# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
            c=ac3.fit_predict(principalDf), cmap='rainbow')
plt.show()

# With k=4
ac4 = AgglomerativeClustering(n_clusters=4)
# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
            c=ac4.fit_predict(principalDf), cmap='rainbow')
plt.show()

# With k=5
ac5 = AgglomerativeClustering(n_clusters=5)
# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
            c=ac5.fit_predict(principalDf), cmap='rainbow')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------
# e) Evaluate different variations using Silhouette Scores and Visualize results with a bar chart.
k = [2, 3, 4, 5]
# To display the number of clusters as integers in the bar plot
new_list = range(math.floor(min(k)), math.ceil(max(k))+1)
plt.xticks(new_list)

# Creating a list with the silhouette scores of the different models
silhouette_scores = [silhouette_score(principalDf, ac2.fit_predict(principalDf)),
                     silhouette_score(principalDf, ac3.fit_predict(principalDf)),
                     silhouette_score(principalDf, ac4.fit_predict(principalDf)),
                     silhouette_score(principalDf, ac5.fit_predict(principalDf))]

# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize=20)
plt.ylabel('silhouette_scores', fontsize=20)
plt.show()
