# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 04:03:06 2023

@author: a0056407
"""
import pandas as pd, time, os, numpy as np, warnings, matplotlib.pyplot as plt, seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
from yellowbrick.cluster import SilhouetteVisualizer

start = time.time()
startptd = time.strftime('%X %x %Z')
print('The program start time and Date','\n',startptd)

#seting up a few parameters
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')

#Setting the working directory
os.chdir(r'C:\Users\a0056407\Desktop\ClusteringDataset\feb')

dfA = pd.read_excel(r'C:\Users\a0056407\Desktop\ClusteringDataset\feb\MichaelVADataAugust2022.xlsx')
#Converting the columns headers to lower case
dfA.columns = dfA.columns.str.lower() 
dfA['gender'] = 9
dfA['gender'][dfA['male'] == 'y'] = 1
dfA['gender'][dfA['female'] == 'y'] = 0
print(dfA['gender'].value_counts())
print(dfA['male'].value_counts())
print(dfA['female'].value_counts())
dfA["dod"] = pd.to_datetime(dfA["dod"])
dfA["yod"] = dfA['dod'].map(lambda x: x.year)
print(dfA.head())

dfB = pd.concat([dfA['household'], dfA['death_age'], dfA['gender'], dfA['yod']], axis = 1)
print(dfB.describe())
print(dfB.info())
print(len(dfB['household'].unique()))
print(len(dfB['death_age'].unique()))
print(len(dfB['gender'].unique()))
print(len(dfB['yod'].unique()))
#age groups
bins= [0,4,14,49,64,110]
labels = ['0-4','5-14','15-49','50-64','65+']
dfB['age_group'] = pd.cut(dfB['death_age'], bins=bins, labels=labels, right=False)
print(dfB.head())

print(dfB.groupby(['gender'])['death_age'].mean().round().to_frame())

sns.pairplot(dfB)
plt.savefig("pair_plot.png", dpi=1200)
plt.savefig("pair_plot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
plt.ylabel('Year of death', fontsize = 18)
plt.xlabel('Age at death', fontsize = 18)
# Basic 2D density plot
sns.kdeplot(x=dfB.death_age, y=dfB.yod)
fig.savefig("age_yod_kdeplot.png", dpi=1200)
fig.savefig("age_yod_kdeplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
plt.ylabel('Year of death', fontsize = 18)
plt.xlabel('Age at death', fontsize = 18)
# Custom the color, add shade and bandwidth
sns.kdeplot(x=dfB.death_age, y=dfB.yod, cmap="Reds", shade=True, bw_adjust=.5)
fig.savefig("age_yod_kdeplot_II.png", dpi=1200)
fig.savefig("age_yod_kdeplot_II.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
plt.ylabel('Year of death', fontsize = 18)
plt.xlabel('Age at death', fontsize = 18)
sns.kdeplot(x=dfB.death_age, y=dfB.yod, cmap="Blues", shade=True, thresh=0)
fig.savefig("age_yod_kdeplot_III.png", dpi=1200)
fig.savefig("age_yod_kdeplot_III.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
# Use the 'hue' argument to provide a factor variable
sns.lmplot( x="death_age", y="yod", data=dfB, fit_reg=False, hue='gender', legend=False)
plt.legend(loc='lower right')
plt.ylabel('Year of death', fontsize = 18)
plt.xlabel('Age at death', fontsize = 18)
fig.savefig("age_yod_lmplot.png", dpi=1200)
fig.savefig("age_yod_lmplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
plt.ylabel('Year of death', fontsize = 18)
plt.xlabel('Age at death', fontsize = 18)
plt.plot('death_age', 'yod', data=dfB, linestyle='-', marker='o')
fig.savefig("age_yod_connected_scatter.png", dpi=1200)
fig.savefig("age_yod_connected_scatter.pdf", dpi=1200)
plt.show()


fig = plt.figure(figsize=(15,6))
dfB["death_age"].hist()
plt.title('Age at the time of death', fontsize = 18)
plt.ylabel('Counts', fontsize = 18)
plt.xlabel('Age at death', fontsize = 18)
plt.tight_layout()
fig.savefig("age_at_death_histogram.png", dpi=1200)
fig.savefig("age_at_death_histogram.pdf", dpi=1200)
plt.show()

# KDE plots for each species
sns.kdeplot(data=dfB['death_age'], label="age distribution", shade=True)
# Add title
#plt.title("Distribution of age at the time of cancer diagnosis")
plt.xlabel('Age at the time of deaths')
plt.ylabel('Densities weights')
plt.savefig('age_at_death_density.png', dpi = 1200)
plt.savefig('age_at_death_density.pdf', dpi = 1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.countplot(x="age_group", hue="gender", data=dfB)
plt.title('Age groups counts per gender', fontsize = 18)
plt.ylabel('Counts', fontsize = 18)
plt.xlabel('Age groups', fontsize = 18)
plt.tight_layout()
fig.savefig("age_group_gender_countplot.png", dpi=1200)
fig.savefig("age_group_gender_countplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.countplot(x="yod", hue="age_group", data=dfB)
plt.title('Death counts per year per age group', fontsize = 18)
plt.ylabel('Counts', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("age_group_yod_countplot.png", dpi=1200)
fig.savefig("age_group_yod_countplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.catplot(x="yod", hue="age_group", col="gender",data=dfB, kind="count")
plt.title('Death counts per year per age group per gender', fontsize = 18)
plt.ylabel('Counts', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("age_group_yod_gender_catplot.png", dpi=1200)
fig.savefig("age_group_yod_gender_catplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.distplot(dfB['yod'])
plt.title('Density plot for deaths over the years', fontsize = 18)
plt.ylabel('Density counts', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("yod_distplot.png", dpi=1200)
fig.savefig("yod_distplot.pdf", dpi=1200)
plt.show()


fig = plt.figure(figsize=(15,6))
sns.countplot(x='gender',data=dfB)
plt.title('Gender counts', fontsize = 18)
plt.ylabel('Counts', fontsize = 18)
plt.xlabel('Gender', fontsize = 18)
plt.tight_layout()
fig.savefig("gender_countplot.png", dpi=1200)
fig.savefig("gender_countplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.countplot(x="yod", hue="gender", data=dfB)
plt.title('Distribution of gender for each year of death', fontsize = 18)
plt.ylabel('Counts', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("yod_gender_countplot.png", dpi=1200)
fig.savefig("yod_gender_countplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.boxplot(x="gender", y="death_age", data=dfB, showmeans=True)
plt.title('Box plot for age at death per gender', fontsize = 18)
plt.ylabel('Age at death', fontsize = 18)
plt.xlabel('Gender', fontsize = 18)
plt.tight_layout()
fig.savefig("age_gender_boxplot.png", dpi=1200)
fig.savefig("age_gender_boxplot.pdf", dpi=1200)
plt.show()


fig = plt.figure(figsize=(15,6))
sns.boxplot(x="yod", y="death_age", data=dfB, showmeans=True)
plt.title('Box plot for age at death per year', fontsize = 18)
plt.ylabel('Age at death', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("age_yod_boxplot.png", dpi=1200)
fig.savefig("age_yod_boxplot.pdf", dpi=1200)
plt.show()


fig = plt.figure(figsize=(15,6))
sns.lineplot(x='yod', y='death_age', data=dfB)
plt.title('Distribution of age at death', fontsize = 18)
plt.ylabel('Age at death', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("age_yod_lineplot.png", dpi=1200)
fig.savefig("age_yod_lineplot.pdf", dpi=1200)
plt.show()

fig = plt.figure(figsize=(15,6))
sns.lineplot(x = 'yod', y = 'death_age', data = dfB, hue = 'gender', style = 'gender')
plt.title('Distribution of age at death per gender', fontsize = 18)
plt.ylabel('Age at death', fontsize = 18)
plt.xlabel('Year of death', fontsize = 18)
plt.tight_layout()
fig.savefig("age_yod_gender_lineplot.png", dpi=1200)
fig.savefig("age_yod_gender_lineplot.pdf", dpi=1200)
plt.show()

dfC = pd.concat([dfA['household'], dfA['death_age'], dfA['gender'], dfA['yod']], axis = 1)
#calling the dataset
scaler = preprocessing.MaxAbsScaler().fit(dfC)
#scaler = preprocessing.PowerTransformer(method='box-cox', standardize=False)
print(scaler)
#print(scaler.mean_)
print(scaler.scale_)
#getting the scalled data
X_scaled = scaler.transform(dfC)
print(X_scaled)
#mean and std
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))


##################################################################################
#kMeans clusttering
##################################################################################
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    #print(kmeans.inertia_)
    #print(kmeans.cluster_centers_)
    #print(kmeans.n_iter_)  
fig = plt.figure(figsize=(15,6))
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow Joint')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.tight_layout()
fig.savefig("elbow_joint.png", dpi=1200)
fig.savefig("elbow_joint.pdf", dpi=1200)
plt.show()


kmeans = KMeans(n_clusters=5)
kmeans.fit(X_scaled)
y_kmeans = kmeans.predict(X_scaled)
fig = plt.figure(figsize=(15,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.tight_layout()
fig.savefig("kmeans_clustters.png", dpi=1200)
fig.savefig("kmeans_clustters.pdf", dpi=1200)
plt.show()

#getting the scores
ac = accuracy_score(y_kmeans, kmeans.labels_)
print('Accuracy Score: %.3f' % ac)
score = silhouette_score(X_scaled, kmeans.labels_, metric='euclidean')
print('KMeans Clusttering Silhouetter Score: %.3f' % score)

#silhouette visualizer for kmeans
fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    #Create KMeans instance for different number of clusters
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    #Create SilhouetteVisualizer instance with KMeans instance Fit the visualizer
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X_scaled)
#plt.xlabel('Predicted label', fontsize=18)
#plt.ylabel('True label', fontsize=18)
#plt.title('Silhouette plots for 2, 3, 4, and 5 clusters', fontsize=18)
plt.tight_layout()
fig.savefig("kmeans_silhouette_plots.png", dpi=1200)
fig.savefig("kmeans_silhouette_plots.pdf", dpi=1200)
plt.show()


#heatmap
conf_matrix=confusion_matrix(y_kmeans, kmeans.labels_) 
fig, ax = plt.subplots(figsize=(15, 15))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predicted label', fontsize=18)
plt.ylabel('True label', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.tight_layout()
fig.savefig("kmeans_heatmap.png", dpi=1200)
fig.savefig("kmeans_heatmap.pdf", dpi=1200)
plt.show()



def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

centers, labels = find_clusters(X_scaled, 4)
fig = plt.figure(figsize=(15,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis')
fig.savefig("kmeans_clustters_pairwise_distances.png", dpi=1200)
fig.savefig("kmeans_clustters_pairwise_distances.pdf", dpi=1200)
plt.show()


##################################################################################
#Spectral clusttering
##################################################################################
model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X_scaled)
fig = plt.figure(figsize=(15,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=50, cmap='viridis')
fig.savefig("spectral_clustering.png", dpi=1200)
fig.savefig("spectral_clustering.pdf", dpi=1200)
plt.show()

#getting the scores
score = silhouette_score(X_scaled, labels, metric='euclidean')
print('Spectral Clustering Silhouetter Score: %.3f' % score)

##################################################################################
#GMM clusttering
##################################################################################
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(X_scaled)
labels = gmm.predict(X_scaled)
fig = plt.figure(figsize=(15,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=40, cmap='viridis')
fig.savefig("gmm_clustering.png", dpi=1200)
fig.savefig("gmm_clustering.pdf", dpi=1200)
plt.show()

#getting the scores
score = silhouette_score(X_scaled, labels, metric='euclidean')
print('Gaussian Mixture Clustering Silhouetter Score: %.3f' % score)

##################################################################################
#Agglomerative clusttering
##################################################################################
agglo = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
labels = agglo.fit_predict(X_scaled)
fig = plt.figure(figsize=(15,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
fig.savefig("agglomerative_clustering.png", dpi=1200)
fig.savefig("agglomerative_clustering.pdf", dpi=1200)
plt.show()

#getting the scores
score = silhouette_score(X_scaled, labels, metric='euclidean')
print('Agglomerative Clustering Silhouetter Score: %.3f' % score)

##################################################################################
#Hierarchical clusttering
##################################################################################
fig = plt.figure(figsize=(15,6))
linkage_data = linkage(X_scaled, method='ward', metric='euclidean')
dendrogram(linkage_data)
fig.savefig("dendrogram_clustering.png", dpi=1200)
fig.savefig("dendrogram_clustering.pdf", dpi=1200)
plt.show()

print('halting...') 
stoptd = time.strftime('%X %x %Z')
print('\n','The program stop time and Date','\n',stoptd)
print('It took', (time.time()-start)/60, 'minutes to run the script.')


