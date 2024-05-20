#!/usr/bin/env python
# coding: utf-8

# # ML Task-02

# Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.
# 
# Dataset :- https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

# In[64]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[65]:


df = pd.read_csv('Mall_Customers.csv')
df.head()


# In[66]:


df.describe()


# # Pre-processing 

# In[67]:


df.info()


# In[68]:


df = df.drop('Gender', axis=1)
df.head()


# In[69]:


df.isnull().sum()


# # Data Visualization

# In[70]:


sns.pairplot(df, 
             kind='scatter', 
             plot_kws={'alpha':0.4}, 
             diag_kws={'alpha':0.55, 'bins':40})


# # Clustering using K- means

# ### Select relevant columns

# In[79]:


X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X = np.nan_to_num(X)


# ### Standardize the data

# In[85]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.set_printoptions(threshold=5)  
print(X_scaled)


# In[81]:


from sklearn.cluster import KMeans


# # Determine the optimal number of clusters using the elbow method

# In[74]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# # Train the K-means model

# In[75]:


kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the original dataset
df['Cluster'] = y_kmeans
df.head()


# In[76]:


df.groupby('Cluster').mean()


# In[77]:


plt.figure(figsize=(10, 8))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette=['c', 'y', 'm', 'r', 'b'])
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# # Conclusion 

# Upon applying the K-Means algorithm with 5 clusters, we segmented the customers effectively. The clusters were visualized using a scatter plot with distinct colors representing each cluster. This visualization revealed clear groupings within the data, highlighting differences in spending behavior and income levels among customers.
