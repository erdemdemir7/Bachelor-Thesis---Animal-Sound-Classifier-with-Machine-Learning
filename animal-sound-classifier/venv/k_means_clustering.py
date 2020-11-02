import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from functions import *
from pycm import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize

'''
bird_map = {'azrm rep and soundfiles':1,'bagm rep and soundfiles':2,'bgmr rep and soundfiles':3,'bgmy partial rep and soundfiles':4,'bhmb prelim rep display and soundfiles':5,
            'bmgw rep and soundfiles':6,'bmor display and soundfiles done':7,'bomp rep and soundfiles':8,'boom rep and soundfiles':9,'caim rep and soundfiles':10,
            'ccym rep and soundfiles':11,'cowm rep and soundfiles':12,'cyom rep and soundfiles':13,'ggmr rep and soundfiles':14,'ggrm rep and soundfiles':15}
'''


def KMEANS_CLUSTER(preferences):
    # Train-set initialized
    # X, y = preprocess()
    X = pre_process_unlabelled()
    is_optimized = preferences[-1]
    if is_optimized:
        n_cluster = 13
    else:
        n_cluster = int(preferences[0])

    # Parameters that are going to be taken from input GUI
    n_clusters = n_cluster

    k_means = KMeans(n_clusters=n_clusters)

    # Normalizing process for input data
    X_s = StandardScaler().fit_transform(X)
    X_norm = normalize(X_s)

    X_norm = pd.DataFrame(X_norm)

    # Dimension Reduction
    x_embedded = TSNE(n_components=2).fit_transform(X_norm)
    X_embedded = pd.DataFrame(x_embedded)
    X_embedded.columns = ['P1', 'P2']

    '''
    # To predict the number of clusters in the beginning minimize wcss
    wcss = []
    for i in range(1, 16):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_embedded)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 16), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    '''

    # Plot normalized data
    # plt.scatter(X_embedded['P1'], X_embedded['P2'], cmap='mediumpurple')
    # plt.title('K-Means Not Applied')
    # plt.show()

    # Fitting on the entire data
    model = k_means.fit(X_embedded)
    predictions = k_means.predict(X_embedded)
    centroids = k_means.cluster_centers_
    labels = k_means.labels_

    # Generate random color
    def random_color():
        r = random.uniform(0, 1)
        g = random.uniform(0, 1)
        b = random.uniform(0, 1)
        rgb = [r, g, b]
        return tuple(rgb)

    ### Output and color arrangement
    colors = {}
    lbl_names = []
    indexes = np.arange(len(np.unique(labels)))
    for x in range(len(indexes)):
        colors[x] = random_color()
        lbl_names.append(f'Centroid {x}')

    colors_set = []
    for x in range(len(colors) - 1):
        colors_set.append(plt.scatter(X_embedded['P1'], X_embedded['P2'], color=colors[x]))

    # Plotted results
    plt.clf()
    plt.scatter(X_embedded['P1'], X_embedded['P2'], cmap='mediumpurple')
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], cmap=random_color(), s=120)
    plt.title('K-Means Applied')
    plt.legend(tuple(colors_set), tuple(lbl_names), loc='best', fontsize=8)
    plt.savefig('K-Means.png')
    # plt.show()

    # Output is initialized
    output = []
    tmp = ''

    tmp = f'\nSilhouette Coefficient: {metrics.silhouette_score(X_embedded, labels)}'
    output.append(tmp)
    # print(tmp)

    return output

    '''
    # Plotting data for intervals
    for x in range(n_clusters):
        interval = x+1

        plt.figure(figsize=(10, 5))
        plt.scatter(X[:(interval*interval_len),:20], X[:(interval*interval_len), 20:])
        plt.scatter(centroids[:interval, :20], centroids[:interval, 20:], c='r', s=80)
        plt.title(f'Centroid interval: #{interval}')
        if x == n_clusters-1:
            plt.savefig('K-Means.png')
        plt.show()

    # Print some metrics
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(y, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(y, labels,
                                               average_method='arithmetic'))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, predictions)))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y, predictions))
    '''

