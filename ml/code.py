import pandas as pd # type: ignore
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

df = pd.read_csv('transaction_ml.csv')

# user_crm_id = df['user_crm_id']
df_drop = df.drop(['user_crm_id', 'Recency', 'F_Score','M_Score', 'RFM_Code', 'Customer_Segment'], axis=1)
df = df.drop(['R_Score', 'F_Score','M_Score', 'RFM_Code'], axis=1)

# user_crm_id = user_crm_id.to_numpy().astype(int)
transactions = df_drop.to_numpy()

scale = StandardScaler()
X = scale.fit_transform(transactions)

# min_clusters = 4
# max_clusters = 8
# cost=[]
# for i in range(min_clusters, max_clusters):
#     kmean= KMeans(i)
#     kmean.fit(X)
#     cost.append(kmean.inertia_)  
    
# plt.plot(cost, 'bx-')
# plt.show()

num_clusters = 5
kmeans = KMeans(num_clusters)
kmeans.fit(X)
labels = kmeans.labels_
labels_df = pd.DataFrame(labels, columns=['Label'])
df['Label'] = labels_df

for i in range(num_clusters):
    print("Number of " + str(i) + " is: " + str(np.sum(labels == i)))

df.to_csv('k_means_cluster.csv', index=False)

