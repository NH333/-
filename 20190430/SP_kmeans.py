#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8
# coding: unicode_escape
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cluster import SpectralClustering
import numpy as np
from itertools import cycle, islice

# X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)

##########
'''
X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], 
                  random_state =9)

plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()
'''
##########

# y_pred = KMeans(n_clusters=2,random_state=9).fit_predict(X)
# plt.scatter(X[:,0],X[:,1],c=y_pred)
# plt.show()

# def CH_score(X, label):
'''
"""k_means 源码"""
def k_means(X, n_clusters, sample_weight=None, init='k-means++',
            precompute_distances='auto', n_init=10, max_iter=300,
            verbose=False, tol=1e-4, random_state=None, copy_x=True,
            n_jobs=None, algorithm="auto", return_n_iter=False):

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    # avoid forcing order when copy_x=False
    order = "C" if copy_x else None
    X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32],
                    order=order, copy=copy_x)
    # 验证给出的样本数大于k
    if _num_samples(X) < n_clusters:
        raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
            _num_samples(X), n_clusters))

    tol = _tolerance(X, tol)

    # If the distances are precomputed every job will create a matrix of shape
    # (n_clusters, n_samples). To stop KMeans from eating up memory we only
    # activate this if the created matrix is guaranteed to be under 100MB. 12
    # million entries consume a little under 100MB if they are of type double.
    if precompute_distances == 'auto':
        n_samples = X.shape[0]
        precompute_distances = (n_clusters * n_samples) < 12e6
    elif isinstance(precompute_distances, bool):
        pass
    else:
        raise ValueError("precompute_distances should be 'auto' or True/False"
                         ", but a value of %r was passed" %
                         precompute_distances)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, '__array__'):
            init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None
    if n_clusters == 1:
        # elkan doesn't make sense for a single cluster, full will produce
        # the right result.
        algorithm = "full"
    if algorithm == "auto":
        algorithm = "full" if sp.issparse(X) else 'elkan'
    if algorithm == "full":
        kmeans_single = _kmeans_single_lloyd
    elif algorithm == "elkan":
        kmeans_single = _kmeans_single_elkan
    else:
        raise ValueError("Algorithm must be 'auto', 'full' or 'elkan', got"
                         " %s" % str(algorithm))
    if effective_n_jobs(n_jobs) == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, n_clusters, max_iter=max_iter, init=init,
                verbose=verbose, precompute_distances=precompute_distances,
                tol=tol, x_squared_norms=x_squared_norms,
                random_state=random_state)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_single)(X, sample_weight, n_clusters,
                                   max_iter=max_iter, init=init,
                                   verbose=verbose, tol=tol,
                                   precompute_distances=precompute_distances,
                                   x_squared_norms=x_squared_norms,
                                   # Change seed to ensure variety
                                   random_state=seed)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    distinct_clusters = len(set(best_labels))
    if distinct_clusters < n_clusters:
        warnings.warn("Number of distinct clusters ({}) found smaller than "
                      "n_clusters ({}). Possibly due to duplicate points "
                      "in X.".format(distinct_clusters, n_clusters),
                      ConvergenceWarning, stacklevel=2)

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia
'''

##########
'''
y_pred = SpectralClustering().fit_predict(X)
print("score:",metrics.calinski_harabaz_score(X,y_pred))
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
'''
##########

##########
'''
for index, gamma in enumerate((0.01,0.1,1,10)):
    for index, k in enumerate((3,4,5,6)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        print ("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred))
        plt.scatter(X[:,0],X[:,1],c=y_pred)
        plt.show()
'''
##########

def RBF(v1,v2,sigma=1.0):
    L2_distance = np.sum((v1-v2)**2)
    denominator = 2*(sigma**2)
    distance = np.exp(-L2_distance/denominator)
    return distance

def EuclideanDistance(v1,v2):
    distance = np.sum((v1-v2)**2)
    return distance 

def SimilarMatrix(v):
    s = np.zeros((len(v),len(v)))
    for i in range(len(v)):
        for j in range(i,len(v)):
            s[j][i] = RBF(v[i],v[j])
            s[i][j] = s[j][i]
    return s

"""返回邻接矩阵"""
def MyKnn(S,k):
    row = S.shape[0]
    col = S.shape[1]
    A = np.zeros((row,col))
    for i in range(row):
        tmp = S[i,:].copy()
        tmp = sorted(list(tmp))
        tmp = tmp[::-1]
        max_Wij_before_k = tmp[k-1]   
        for j in range(col):
            if S[i][j]>=max_Wij_before_k:
                A[i][j] = S[i][j]
                A[j][i] = A[i][j]
    return A

def DegreeMatrix(A):
    res = np.sum(A,axis=0) #按列叠加
    res = np.diag(res)
    return res

def LaMatrix(D,W):
    L = D - W
    # D_change = (D**(-0.5))
    """正则化"""
    # L_normall = D_change*L*D_change
    # L_normall = np.dot(np.dot(D_change,L),D_change)
    # np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
    return L
        
def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y

def plot(X, y_sp, y_km):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_km) + 1))))
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")
    plt.show()
    

data, label = genTwoCircles(n_samples=500)
plt.scatter(data[:,0],data[:,1],s=10)
plt.show()

Similar_Matrix = SimilarMatrix(data)
Affinity_Matrix = MyKnn(Similar_Matrix,k=10)
Degree_Matrix = DegreeMatrix(Affinity_Matrix)
Laplacian_Matrix = LaMatrix(Degree_Matrix,Affinity_Matrix)

eigenvalue,eigenvector = np.linalg.eig(Laplacian_Matrix)
x = zip(eigenvalue, range(len(eigenvalue))) #把特征值和对应的样本编号组合
x = sorted(x, key=lambda x:x[0]) #重新进行排序

H = np.vstack([eigenvector[:,i] for (v, i) in x[:500]]).T #v表示特征值，i表示特征值对应样本的编号

sp_kmeans = KMeans(n_clusters=2).fit(H)
pure_kmeans = KMeans(n_clusters=2).fit(data)

plot(data, sp_kmeans.labels_, pure_kmeans.labels_)
input('')