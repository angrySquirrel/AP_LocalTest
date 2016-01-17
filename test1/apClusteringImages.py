#coding:utf-8
'''
Created on 2015年12月9日
set the image folders ,with images given format as "001_0001.jpg 001_0002.jpg" as groundtruth
computing the sift features, and calculate the distance between images
computing the affinity propagation clustering 
classified images to different clusters
compute the NMI
@author: Singleton
'''
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import platform

from scipy.stats.stats import itemfreq
#from skimage.feature import local_binary_pattern
from sklearn.cluster import AffinityPropagation
import numpy as np
from numpy import mean
from sklearn.metrics.cluster import normalized_mutual_info_score
#获得文件夹下所有的图像的路径
def get_Imlist(path):  
        paths = []
        classId = []
#         files = []
        for f in os.listdir(path): 
            if f.endswith('.jpg') :    
                paths.append(os.path.join(path,f))
                # input the image Name, get the label idx, eg, 001_0001.jpg means the label idx is 1
                classId.append(int(f.split('_')[0])) # convert string to integer  
#                 files.append(int(f.split('_')[-1].split('.')[0]))
#         files,paths = zip(*sorted(zip(files, paths)))
        return paths,classId
#获得全部图像特征
def get_Feature(imgFolder):     
        imList,classId = get_Imlist(imgFolder)
        counter = 0
        imgDes=[] 
        sift = cv2.SIFT()  # @UndefinedVariable
        dsize = (300,300)        
        for im in imList:      
                counter+=1
                print 'image: ' + str(im) + ' number: ' + str(counter)                
                img = cv2.imread(im,1) #read the color image                
                img = cv2.resize(img,dsize=dsize) 
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                
                #using sift
                kp,des = sift.detectAndCompute(img_gray,None)  # @UndefinedVariable
                print des.shape
                imgDes.append(des)
                # draw the keypoints
#                 img_show = cv2.drawKeypoints(img_gray,kp)
#                 cv2.imwrite('sift'+str(counter)+'.jpg',img_show)
        return imgDes,counter,imList,classId

#遍历文件夹生成相似度矩阵
def  get_SimilairtyMatrix(imgDes,num):
        bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
        similarity = np.zeros((num, num)) 
         
        for i in range((num)):
                for j in range((num)):                                
                        des1 = imgDes[i]
                        des2 = imgDes[j]                    
           
                        #     matches = bf.match(des1,des2)
                        #     matches = sorted(matches, key = lambda x:x.distance)
                        matches = bf.knnMatch(des1,des2,k=2)
                        goodMatch = 0
                        for m,n in matches:
                                if m.distance < 0.75 * n.distance:
                                        goodMatch +=1
                        similarity[i,j] = goodMatch

#         print similarity        
        # 减去均值
        similarity= similarity -  mean(similarity, axis = 0)[np.newaxis,:] - mean(similarity,axis = 1)[:, np.newaxis]
#         print similarity
        return  similarity


    

if __name__ == '__main__':
 
    # imgFolder = '/home/hadoop/桌面/132/small'
#     if platform.system() == "Windows" :
#         
#     else:
    
    imgFolder = 'C:\Users\Administrator\Desktop\dataset'
    pathImages = imgFolder
    
    # delete existed folders
    
    if os.path.exists(os.path.join(imgFolder,'CenterImages')):
        shutil.rmtree(os.path.join(imgFolder,'CenterImages'))
    if os.path.exists(os.path.join(imgFolder,'Clusters')):
        shutil.rmtree(os.path.join(imgFolder,'Clusters'))
    
    # pathImages = imgFolder+'/img2'
    imgDes,numOfImgs,imPaths,classId=get_Feature(pathImages)
    S=get_SimilairtyMatrix(imgDes,numOfImgs)
#     pre = S.min()
    pre = np.median(np.median(S, axis = 0))
    print 'the median value is ' + str(pre)
    af = AffinityPropagation(preference=pre, verbose=True,max_iter=500,affinity="precomputed").fit(S)
    labels = af.labels_
    nmiScore = normalized_mutual_info_score(classId,labels)
    print 'classId' + ''.join([str(e) for e in classId ,' '])
    print 'clustering result' + ''.join([str(e) for e in labels,' '])
    print 'the NMI Score of classification is:' + str(nmiScore)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    #################################################################
    #
    # Create folder with central images for each cluster
    #
    #################################################################  
          
    
    #obtain representative images for each cluster
    central_ims =  cluster_centers_indices
    
    central_folder = os.path.join(imgFolder,'CenterImages')
    if not os.path.exists(central_folder):
            os.makedirs(central_folder)    
    
    count=0
    for central_im in central_ims:
            filename = os.path.join(central_folder,'Cluster_'+str(count)+'.jpg')
            img = cv2.imread(imPaths[central_im],1)
            cv2.imwrite(filename, img)             
            count = count + 1
             
    #ADDED
    #################################################################
    #
    # Separate Clusters into folders
    #
    #################################################################           
       
    clusters_folder = os.path.join(imgFolder,'Clusters')
    if not os.path.exists(clusters_folder):
            os.makedirs(clusters_folder) 
    clust_dir = []
    print 'number of clusters : ' + str(n_clusters_ )
    for iclust in range(0,n_clusters_ ):
            direc = os.path.join(clusters_folder,'Cluster_'+str(iclust))
            if not os.path.exists(direc):
                    os.makedirs(direc)         
            clust_dir.append(direc)
          
    for im in range(0,len(imPaths)):
    #         im_name = imPaths[im].split('/')[-1] # for linux
            im_name = imPaths[im].split('\\')[-1]
            print clust_dir[int(labels[im])]
            filename = os.path.join(clust_dir[int(labels[im])],im_name)
            #print filename
            img = cv2.imread(imPaths[im],1)
            cv2.imwrite(filename, img)          

    
   
                
                
