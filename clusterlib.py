import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import warnings
import statistics

from collections import Counter
from sklearn.cluster import KMeans, SpectralClustering, Birch, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from timeit import default_timer as timer
from torch.utils.data import DataLoader

from base import BaseDataset, get_data, get_cifar100_coarse
from cluster import cluster_diff_alg, clusterKMEANS
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def gmm(x, seed=None, kind=None, n_components=[2, 3]):
    """
    Run clustering for the given x.
    The number of components are 2 and 3.
    kind: "none" for GMM, "kmeans" for KMeans
    Return:
        - the model that gives the highest silhouette score
        - the predictions for x
    """
    # n_components = np.arange(2, 4)
    print("BUILDING GMM WITH N_COMPONENTS:", n_components)    
    models = []
    silhouette, predictions = [], [] 
    time_log = {}
    for n in n_components:
        # m = GaussianMixture(n, covariance_type="full", random_state=seed)
        # time_start = timer()
        m = cluster_factory(n, seed, kind=kind)
        pred = m.fit_predict(x)
        # time_end = timer()
        # print(f"cluster_time: {(time_end - time_start)}")
        models.append(m)
        predictions.append(pred)       
        if n != 1:
            silhouette.append(silhouette_score(x, predictions[-1]))
            # time_log[n] = time_end - time_start
            print("silhouette:", silhouette)
        
    
    if len(n_components) == 1:
        model = models[0]
        pred = predictions[0]
    else:
        num_cluster = silhouette.index(max(silhouette)) + 2
        model = models[num_cluster-2]
        pred = predictions[num_cluster-2]
    return model, pred, time_log

def cluster_factory(k, seed=None, kind=None):
    if kind is None or kind == "gmm":
        return GaussianMixture(k, covariance_type="full", random_state=seed)
    elif kind == "kmeans":
        return KMeans(n_clusters=k, random_state=seed)
    elif kind == "spectral":
        return SpectralClustering(n_clusters=k, random_state=seed)
    elif kind == "birch":
        # Birch does not have random_state
        return Birch(n_clusters=k)
    elif kind == "agglomerative":
        return AgglomerativeClustering(n_clusters=k)


def class_gmm(x, y, logger={}, n_components=[2, 3]):
    """
    Run GMM for each class in x
    The number of components are 2 and 3.

    logger: a dictionary to store the class and learning time in format: {class: time}

    Return:
        - a dictionary of {class: gmm}
    """    
    class_gmm = {}
    y_unique = np.unique(y)
    # print(type(x), type(y), type(y_unique))
    print(y_unique)
    for y_u in y_unique:
        print("Process GMM for class:", y_u)
        x_ = x[y == y_u]
        
        gmm_start = time.process_time()
        model, _, _ = gmm(x_, n_components=n_components)
        gmm_end = time.process_time()

        logger[y_u] = gmm_end - gmm_start

        class_gmm[y_u] = model
    return class_gmm        

def get_class_mean(x, y):
    """
    Calculate the class mean for given x and class y
    Return:
        - the class means
        - the order of y
    """    
    cls_means, cls_order = [], []
    y_unique = np.unique(y)
    for y_u in y_unique:
        x_ = x[y == y_u]
        cls_order.append(y_u)
        # m_ = np.mean(x_, axis=0)
        m_ = class_mean_norm(x_)
        cls_means.append(m_)
    return cls_means, cls_order

def class_mean_norm(x):
    """
    Calculate the class mean for given x while taking the norm into account.
    
    Concretely, it does weighted mean and covariance:
        Mean: 
            Calculate the Frobenius norm of each element and use them as weights for a weighted mean. 
            This gives more importance to elements with larger magnitudes in the Frobenius norm sense.
        Covariance: 
            Apply the same weighting scheme to the centered data (subtract the mean calculated above from each element) 
            before calculating the covariance matrix.
            
    Caveat: the norm of the class mean is used to calculate the final class mean.
    
    param:
    x - the samples in numpy array
    
    return: the mean of xs
    """
    class_mean = np.sum(x / np.linalg.norm(x, axis=1, keepdims=True), axis=0) / x.shape[0]
    class_mean = class_mean / np.linalg.norm(class_mean)
    return class_mean 

def take_out_samples(class_order, take_out_indices, x, y):
    class_to_exclude, samples_to_exclude = [], []
    for idx in take_out_indices:
        y_ = class_order[idx]
        indices = y == y_
        x_ = x[indices]
        
        samples_to_exclude.extend(x_)
        class_to_exclude.extend([y_] * len(x_))
        
        x = np.delete(x, indices, axis=0)
        y = np.delete(y, indices, axis=0)
        
    return x, y, np.array(class_to_exclude), np.array(samples_to_exclude)
        

def class_mean_cluster(x, y, cls2cluster, seed=None, kind=None):
    """
    Run class mean clustering for the given x and y
    Return:
        - the model that gives the highest silhouette score
        - the mapping of class/label to cluster
        - the gmm for each class
    """    
    cls2cluster_ = {}
    cluster_shift = max(cls2cluster, key=int) + 1 if cls2cluster else 0
    cls_means, cls_order = get_class_mean(x, y)
    
    """
    run gmm for the whole sample means.
    the component number for a particular sample mean will be its cluster.
    """
    model, predictions, time_log = gmm(cls_means, seed=seed, kind=kind)
    
    """
    find all clusters with single element.
    exclude those classes from current task
    """
    unique_counts = np.unique(predictions, return_counts=True)
    single_elements = unique_counts[0][unique_counts[1] == 1]
    single_indices = np.where(np.isin(predictions, single_elements))[0]
    
    class_to_exclude, samples_to_exclude = None, None
    # print(f"clusterlib.cls_order: {cls_order}")
    cls2cluster_ = {y_: (p + cluster_shift) for y_, p in zip(cls_order, predictions)}
    # print(f"clusterlib.cls2cluster_: {cls2cluster_}")
    # if single_elements.size > 0:
    #     """if there's a cluster with a single class in it
    #     take that class into the next task"""
    #     # print("found single elements")        
    #     x, y, class_to_exclude, samples_to_exclude = take_out_samples(cls_order, single_indices, x, y)
        
    #     # remove the excluded class from cls_order and predictions
    #     cls_order = np.delete(np.array(cls_order), single_indices).tolist()
    #     predictions = np.delete(predictions, single_indices)
        
    #     # reorder the cluster index by excluding the excluded class(es)
    #     # keep following the cluster_shift
    #     predictions_array = np.array(predictions)
    #     predictions_copy = np.copy(predictions_array)
    #     for idx, val in enumerate(np.unique(predictions_array)):
    #         predictions_copy[predictions_array == val] = idx + cluster_shift
        
    #     cls2cluster_ = {y_: p for y_, p in zip(cls_order, predictions_copy.tolist())}                

    """
    run gmm for each class sample.
    the gmm associated for that class will be it's pseudo-rehearsal generator.
    """
    cls2gmm = class_gmm(x, y)
    
    return model, cls2cluster_, cls2gmm, class_to_exclude, samples_to_exclude, time_log
    # return model, cls2cluster_