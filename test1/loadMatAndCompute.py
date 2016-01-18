#coding:utf-8

'''
Created on 2016年1月17日
load feature mat calculate with matlab, applying ap clustering and count NMI
@author: Singleton
'''
import scipy.io as sio
import os
import numpy as np
import scipy.spatial.distance as sci
import sklearn.metrics.pairwise as skpair
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.cluster import normalized_mutual_info_score




# def getSimilarityMatrix(features):
#     numOfData = np.shape(features)[0]
#     S = [[0 for col in range(numOfData)] for row in range(numOfData)]  
#     for i in range(numOfData):  
#         for j in range(numOfData):  
#              print i
# 
#     return S


def  getClassificationID(nameList):
    classId = []
    for i in range(np.shape(nameList)[0]):
        name = nameList[i,0]
        strName = ''.join(name)
        classId.append(int(strName.split('_')[0]))
    return classId

dataPath = 'C:\Users\Administrator\Desktop\dataset'
# for mat of type v7
d = sio.loadmat(os.path.join(dataPath,'fea2.mat'))
features = d.get('feat_norm') 
nameList = d.get('imgNameList')
# Y = sci.pdist(features, 'euclidean')
S = - skpair.pairwise_distances(features, Y=None, metric='cityblock')
print np.shape(S)

pre = np.median(np.median(S, axis = 0))
print 'the median value is ' + str(pre)
af = AffinityPropagation(preference=pre, verbose=True,max_iter=500,affinity="precomputed").fit(S)
labels = af.labels_
classId = getClassificationID(nameList)
nmiScore = normalized_mutual_info_score(classId,labels)
print nmiScore     
print 'classId' + ''.join([str(e) for e in classId ,' '])
print 'clustering result' + ''.join([str(e) for e in labels,' '])
print 'the NMI Score of classification is:' + str(nmiScore)