from json import load
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import myutils.point_cloud as pc
from sklearn.cluster import KMeans 

def test_with_Kmean(features, points, config, range_for_K = 7, save_pc=False, index = 0):
    """Testing clustering the points with Kmeans_alg from sklearn
    ----------
    Parameters
    features: is the information we use to cluster
    points:
    config: the file
    
    output: None
    
    """

    print('[Now create the clustering with Kmeans...]')
    for i in range(2, range_for_K):
        labels = KMeans(n_clusters=i).fit(features).labels_
        #labels for points in the same cluster

        # print(labels)
            
        pc.show_point_clouds_with_labels(points, list(labels))
        if save_pc: pc.save_point_cloud_with_labels((255*points).astype(np.int32), list(labels),  config, K=i)
        # print('Clustering')