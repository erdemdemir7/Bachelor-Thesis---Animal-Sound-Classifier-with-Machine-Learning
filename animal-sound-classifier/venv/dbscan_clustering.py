import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.manifold import TSNE
from functions import *
from pycm import *


'''
bird_map = {'azrm rep and soundfiles':1,'bagm rep and soundfiles':2,'bgmr rep and soundfiles':3,'bgmy partial rep and soundfiles':4,'bhmb prelim rep display and soundfiles':5,
            'bmgw rep and soundfiles':6,'bmor display and soundfiles done':7,'bomp rep and soundfiles':8,'boom rep and soundfiles':9,'caim rep and soundfiles':10,
            'ccym rep and soundfiles':11,'cowm rep and soundfiles':12,'cyom rep and soundfiles':13,'ggmr rep and soundfiles':14,'ggrm rep and soundfiles':15}
'''

# Generate random color
def random_color():
    r = random.uniform(0, 1)
    g = random.uniform(0, 1)
    b = random.uniform(0, 1)
    rgb = [r, g, b]
    return tuple(rgb)


def DBSCAN_CLUSTER(preferences):
    # Train-set initialized
    #X, y = preprocess()
    X = pre_process_unlabelled()
    labelled = True
    is_optimized = preferences[-1]
    eps = min_samples = 0
    if is_optimized:
       pass
    else:
        min_samples = int(preferences[1])
        eps = float(preferences[0])
    if eps >= 0.5:
        is_optimized = True
    elif min_samples >= 5:
        is_optimized = True
    if is_optimized:
        min_samples = 3
        eps = 2.2
        if labelled:
            eps = 0.11
    else:
        min_samples = int(preferences[1])
        eps = float(preferences[0])

    dbscan = DBSCAN(eps = eps, min_samples = min_samples)

    # Normalizing process for input data
    X_s = StandardScaler().fit_transform(X)
    X_norm = normalize(X_s)

    X_norm = pd.DataFrame(X_norm)

    # Dimension Reduction
    x_embedded = TSNE(n_components=2).fit_transform(X_norm)
    X_embedded = pd.DataFrame(x_embedded)
    X_embedded.columns = ['P1','P2']

    #plt.scatter(X_embedded['P1'],X_embedded['P2'])
    #plt.show()

    # Fitting on the entire data
    model = dbscan.fit(X_embedded)
    labels = model.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    ### Output and color arrangement
    colors = {}
    lbl_names = []
    indexes = np.arange(len(np.unique(labels)))
    for x in range(len(indexes)):
        colors[x] = random_color()
        lbl_names.append(f'Label {x}')
    colors[-1] = random_color()
    lbl_names.append(f'Label {-1}')

    cvec = [colors[lbl] for lbl in labels]
    colors_set = []
    for x in range(len(colors)-1):
        colors_set.append(plt.scatter(X_embedded['P1'],X_embedded['P2'], color= colors[x]))

    # Add for -1
    colors_set.append(plt.scatter(X_embedded['P1'],X_embedded['P2'], color= colors[-1]))

    # DBSCAN Clustering result is plotted
    plt.clf()
    plt.scatter(X_embedded['P1'], X_embedded['P2'], c=cvec)
    plt.legend(tuple(colors_set), tuple(lbl_names), loc='best', fontsize=3.5)
    plt.title('DBSCAN')
    plt.savefig('DBSCAN.png')
    #plt.show()

    # Output is initialized
    output = []
    tmp = ''

    tmp = f'\nSilhouette Coefficient: {metrics.silhouette_score(X_embedded, labels)}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nEstimated number of clusters: {n_clusters_}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nEstimated number of noise points: {n_noise_}'
    output.append(tmp)
    # print(tmp)

    return output
    '''
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(y, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(y, labels,
                                               average_method='arithmetic'))

    # Print more metrics to compare
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, labels)))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y, labels))
    '''
