import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings
import scalogram
warnings.filterwarnings('ignore')


def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            


def plot3Classes(data1,name1,data2,name2,data3,name3):
    plt.subplot(3,1,1)
    plt.plot(data1)
    plt.title(name1)
    plt.subplot(3,1,2)
    plt.plot(data2)
    plt.title(name2)
    plt.subplot(3,1,3)
    plt.plot(data3)
    plt.title(name3)
    plt.show()
    waitforEnter()
    
## é preciso rever esta função
def breakTrainTest(data,oWnd=300,trainPerc=0.5):
    nSamp,nCols=data.shape
    nObs=int(nSamp/oWnd)
    data_obs=data[:nObs*oWnd,:].reshape((nObs,oWnd,nCols))
    
    order=np.random.permutation(nObs)
    order=np.arange(nObs)    #Comment out to random split
    
    nTrain=int(nObs*trainPerc)
    
    data_train=data_obs[order[:nTrain],:,:]
    data_test=data_obs[order[nTrain:],:,:]
    
    return(data_train,data_test)

def extractFeatures(data,Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class

    for i in range(nObs):
        M1=np.mean(data[i,:,:],axis=0)
        Md1=np.median(data[i,:,:],axis=0)
        Std1=np.std(data[i,:,:],axis=0)
        #S1=stats.skew(data[i,:,:])
        #K1=stats.kurtosis(data[i,:,:])
        p=[75,90,95]
        Pr1=np.array(np.percentile(data[i,:,:],p,axis=0)).T.flatten()
        
        #faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
        faux=np.hstack((M1,Md1,Std1,Pr1))
        features.append(faux)
        
    return(np.array(features),oClass)

# ## -- 4 -- ##
# def plotFeatures(features,oClass,f1index=0,f2index=1):
#     nObs,nFea=features.shape
#     colors=['b','g','r']
#     for i in range(nObs):
#         plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

#     plt.show()
#     waitforEnter()
    
# def logplotFeatures(features,oClass,f1index=0,f2index=1):
#     nObs,nFea=features.shape
#     colors=['b','g','r']
#     for i in range(nObs):
#         plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

#     plt.show()
#     waitforEnter()


def extratctSilence(data,threshold=256):
    if(data[0]<=threshold):
        s=[1]
    else:
        s=[]

    for i in range(1,len(data)):
        if(data[i-1]>threshold and data[i]<=threshold):
            s.append(1)
        elif (data[i-1]<=threshold and data[i]<=threshold):
            s[-1]+=1
    
    return(s)
    
def extractFeaturesSilence(data,Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class

    for i in range(nObs):
        silence_features=np.array([])
        for c in range(nCols):
            silence=extratctSilence(data[i,:,c],threshold=0)
            if len(silence)>0:
                silence_features=np.append(silence_features,[np.mean(silence),np.var(silence)])
            else:
                silence_features=np.append(silence_features,[0,0])
            
            
        features.append(silence_features)
        
    return(np.array(features),oClass)


def extractFeaturesWavelet(data,scales=[2,4,8,16,32],Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class

    for i in range(nObs):
        scalo_features=np.array([])
        for c in range(nCols):
            #fixed scales->fscales
            scalo,fscales=scalogram.scalogramCWT(data[i,:,c],scales)
            scalo_features=np.append(scalo_features,scalo)
            
        features.append(scalo_features)
        
    return(np.array(features),oClass)
    

def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))


### MAIN ###
def main():
    classes = {
        0: 'Youtube',
        1: 'Browsing',
        2: 'Netflix',
        3: 'Spotify',
        4: 'Mining'
    }

    # datasets
    youtube = np.loadtxt('data/youtube.dat')
    browsing = np.loadtxt('data/browsing.dat')
    netflix = np.loadtxt('data/netflix.dat')
    spotify = np.loadtxt('data/spotify.dat')
    mining = np.loadtxt('data/mining.dat')


    # Get training and testing sets
    youtube_train, youtube_test = breakTrainTest(youtube)
    browsing_train, browsing_test = breakTrainTest(browsing)
    netflix_train, netflix_test = breakTrainTest(netflix)
    spotify_train, spotify_test = breakTrainTest(spotify)
    mining_train, mining_test = breakTrainTest(mining)

    # Extract features train set
    trainFeatures_youtube, oClass_youtube = extractFeatures(youtube_train, Class=0)
    trainFeatures_browsing, oClass_browsing = extractFeatures(browsing_train, Class=1)
    trainFeatures_netflix, oClass_netflix = extractFeatures(netflix_train, Class=2)
    trainFeatures_spotify, oClass_spotify = extractFeatures(spotify_train, Class=3)
    trainFeatures_mining, oClass_mining = extractFeatures(mining_train, Class=4)

    trainFeatures = np.vstack((trainFeatures_youtube, trainFeatures_browsing, trainFeatures_netflix, 
                                    trainFeatures_spotify, trainFeatures_mining))


    # Extract features silence train set
    trainFeatures_youtubeS, oClass_youtbe = extractFeaturesSilence(youtube_train, Class=0)
    trainFeatures_browsingS, oClass_browsing = extractFeaturesSilence(browsing_train, Class=1)
    trainFeatures_netflixS, oClass_netflix = extractFeaturesSilence(netflix_train, Class=2)
    trainFeatures_spotifyS, oClass_netflix = extractFeaturesSilence(spotify_train, Class=3)
    trainFeatures_miningS, oClass_mining = extractFeaturesSilence(mining_train, Class=4)

    trainFeaturesS = np.vstack((trainFeatures_youtubeS, trainFeatures_browsingS, trainFeatures_netflixS, 
                                    trainFeatures_spotifyS, trainFeatures_miningS))

    
    # Extract features wavelet train set
    scales=[2,4,8,16,32,64,128]
    trainFeatures_youtubeW, oClass_youtube = extractFeaturesWavelet(youtube_train, scales, Class=0)
    trainFeatures_browsingW, oClass_browsing = extractFeaturesWavelet(browsing_train, scales, Class=1)
    trainFeatures_netflixW, oClass_netflix = extractFeaturesWavelet(netflix_train, scales, Class=2)
    trainFeatures_spotifyW, oClass_spotify = extractFeaturesWavelet(spotify_train, scales, Class=3)
    trainFeatures_miningW , oClass_mining = extractFeaturesWavelet(mining_train, scales, Class=4)

    trainFeaturesW = np.vstack((trainFeatures_youtubeW, trainFeatures_browsingW, trainFeatures_netflixW, 
                                    trainFeatures_spotifyW, trainFeatures_miningW))

    trainClass = np.vstack((oClass_youtube, oClass_browsing, oClass_netflix, oClass_spotify, oClass_mining))
    trainFeatures = np.hstack((trainFeatures, trainFeaturesS, trainFeaturesW))


    # Extract features test set
    testFeatures_youtube, oClass_youtube = extractFeatures(youtube_test, Class=0)
    testFeatures_browsing, oClass_browsing = extractFeatures(browsing_test, Class=1)
    testFeatures_netflix, oClass_netflix = extractFeatures(netflix_test, Class=2)
    testFeatures_spotify, oClass_spotify = extractFeatures(spotify_test, Class=3)
    testFeatures_mining, oClass_mining = extractFeatures(mining_test, Class=4)

    testFeatures = np.vstack((testFeatures_youtube, testFeatures_browsing, testFeatures_netflix, 
                                testFeatures_spotify, testFeatures_mining))

    # Extract features silence test set
    testFeatures_youtubeS, oClass_youtbe = extractFeaturesSilence(youtube_test, Class=0)
    testFeatures_browsingS, oClass_browsing = extractFeaturesSilence(browsing_test, Class=1)
    testFeatures_netflixS, oClass_netflix = extractFeaturesSilence(netflix_test, Class=2)
    testFeatures_spotifyS, oClass_netflix = extractFeaturesSilence(spotify_test, Class=3)
    testFeatures_miningS, oClass_mining = extractFeaturesSilence(mining_test, Class=4)

    testFeaturesS = np.vstack((testFeatures_youtubeS, testFeatures_browsingS, testFeatures_netflixS, 
                                    testFeatures_spotifyS, testFeatures_miningS))


    # Extract features wavelet test set
    scales=[2,4,8,16,32,64,128]
    testFeatures_youtubeW, oClass_youtube = extractFeaturesWavelet(youtube_test, scales, Class=0)
    testFeatures_browsingW, oClass_browsing = extractFeaturesWavelet(browsing_test, scales, Class=1)
    testFeatures_netflixW, oClass_netflix = extractFeaturesWavelet(netflix_test, scales, Class=2)
    testFeatures_spotifyW, oClass_spotify = extractFeaturesWavelet(spotify_test, scales, Class=3)
    testFeatures_miningW , oClass_mining = extractFeaturesWavelet(mining_test, scales ,Class=4)

    testFeaturesW = np.vstack((testFeatures_youtubeW, testFeatures_browsingW, testFeatures_netflixW, 
                                    testFeatures_spotifyW, testFeatures_miningW))

    
    testClass = np.vstack((oClass_youtube, oClass_browsing, oClass_netflix, oClass_spotify, oClass_mining))
    testFeatures = np.hstack((testFeatures, testFeaturesS, testFeaturesW))


    # Feature normalization
    scaler = MaxAbsScaler()
    trainFeaturesN = scaler.fit_transform(trainFeatures)
    testFeaturesN = scaler.transform(testFeatures)

    # Principal Components Analysis (PCA)
    pca = PCA(n_components=5, svd_solver='full')

    trainPCA = pca.fit(trainFeaturesN)
    trainFeaturesPCA = trainPCA.transform(trainFeaturesN)
    testFeaturesPCA = pca.transform(testFeaturesN)



########### Main Code #############
#plt.ion()
#nfig=1
#plt.figure(1)
#plot3Classes(yt,'YouTube',browsing,'Browsing',mining,'Mining')

# plt.figure(2)
# plt.subplot(3,1,1)
# for i in range(10):
#     plt.plot(yt_train[i,:,0],'b')
#     plt.plot(yt_train[i,:,1],'g')
# plt.title('YouTube')
# plt.ylabel('Bytes/sec')
# plt.subplot(3,1,2)
# for i in range(10):
#     plt.plot(browsing_train[i,:,0],'b')
#     plt.plot(browsing_train[i,:,1],'g')
# plt.title('Browsing')
# plt.ylabel('Bytes/sec')
# plt.subplot(3,1,3)
# for i in range(10):
#     plt.plot(mining_train[i,:,0],'b')
#     plt.plot(mining_train[i,:,1],'g')
# plt.title('Mining')
# plt.ylabel('Bytes/sec')
# plt.show()
# waitforEnter()

## -- 3 -- ##
# features_yt,oClass_yt=extractFeatures(yt_train,Class=0)
# features_browsing,oClass_browsing=extractFeatures(browsing_train,Class=1)
# features_mining,oClass_mining=extractFeatures(mining_train,Class=2)

# features=np.vstack((features_yt,features_browsing,features_mining))
# oClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

# print('Train Stats Features Size:',features.shape)

## -- 4 -- ##
# plt.figure(4)
# plotFeatures(features,oClass,0,1)#0,8

## -- 5 -- ##
# features_ytS,oClass_yt=extractFeaturesSilence(yt_train,Class=0)
# features_browsingS,oClass_browsing=extractFeaturesSilence(browsing_train,Class=1)
# features_miningS,oClass_mining=extractFeaturesSilence(mining_train,Class=2)

# featuresS=np.vstack((features_ytS,features_browsingS,features_miningS))
# oClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

# print('Train Silence Features Size:',featuresS.shape)
# plt.figure(5)
# plotFeatures(featuresS,oClass,0,2)


## -- 6 -- ##
# import scalogram
# scales=range(2,256)
# plt.figure(6)

# i=0
# data=yt_train[i,:,1]
# S,scalesF=scalogram.scalogramCWT(data,scales)
# plt.plot(scalesF,S,'b')

# nObs,nSamp,nCol=browsing_train.shape
# data=browsing_train[i,:,1]
# S,scalesF=scalogram.scalogramCWT(data,scales)
# plt.plot(scalesF,S,'g')

# nObs,nSamp,nCol=mining_train.shape
# data=mining_train[i,:,1]
# S,scalesF=scalogram.scalogramCWT(data,scales)
# plt.plot(scalesF,S,'r')

# plt.show()
# waitforEnter()

## -- 7 -- ##
# scales=[2,4,8,16,32,64,128,256]
# features_ytW,oClass_yt=extractFeaturesWavelet(yt_train,scales,Class=0)
# features_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_train,scales,Class=1)
# features_miningW,oClass_mining=extractFeaturesWavelet(mining_train,scales,Class=2)

# featuresW=np.vstack((features_ytW,features_browsingW,features_miningW))
# oClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))

# print('Train Wavelet Features Size:',featuresW.shape)
# plt.figure(7)
# plotFeatures(featuresW,oClass,3,10)

## -- 8 -- ##
#:1
# trainFeatures_yt,oClass_yt=extractFeatures(yt_train,Class=0)
# trainFeatures_browsing,oClass_browsing=extractFeatures(browsing_train,Class=1)
# trainFeatures=np.vstack((trainFeatures_yt,trainFeatures_browsing))

# trainFeatures_ytS,oClass_yt=extractFeaturesSilence(yt_train,Class=0)
# trainFeatures_browsingS,oClass_browsing=extractFeaturesSilence(browsing_train,Class=1)
# trainFeaturesS=np.vstack((trainFeatures_ytS,trainFeatures_browsingS))

# trainFeatures_ytW,oClass_yt=extractFeaturesWavelet(yt_train,scales,Class=0)
# trainFeatures_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_train,scales,Class=1)
# trainFeaturesW=np.vstack((trainFeatures_ytW,trainFeatures_browsingW))

# o2trainClass=np.vstack((oClass_yt,oClass_browsing))
# i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))

#:2
# trainFeatures_yt,oClass_yt=extractFeatures(yt_train,Class=0)
# trainFeatures_browsing,oClass_browsing=extractFeatures(browsing_train,Class=1)
# trainFeatures_mining,oClass_mining=extractFeatures(mining_train,Class=2)
# trainFeatures=np.vstack((trainFeatures_yt,trainFeatures_browsing,trainFeatures_mining))

# trainFeatures_ytS,oClass_yt=extractFeaturesSilence(yt_train,Class=0)
# trainFeatures_browsingS,oClass_browsing=extractFeaturesSilence(browsing_train,Class=1)
# trainFeatures_miningS,oClass_mining=extractFeaturesSilence(mining_train,Class=2)
# trainFeaturesS=np.vstack((trainFeatures_ytS,trainFeatures_browsingS,trainFeatures_miningS))

# trainFeatures_ytW,oClass_yt=extractFeaturesWavelet(yt_train,scales,Class=0)
# trainFeatures_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_train,scales,Class=1)
# trainFeatures_miningW,oClass_mining=extractFeaturesWavelet(mining_train,scales,Class=2)
# trainFeaturesW=np.vstack((trainFeatures_ytW,trainFeatures_browsingW,trainFeatures_miningW))

# o3trainClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))
# i3trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))

#:3
# testFeatures_yt,oClass_yt=extractFeatures(yt_test,Class=0)
# testFeatures_browsing,oClass_browsing=extractFeatures(browsing_test,Class=1)
# testFeatures_mining,oClass_mining=extractFeatures(mining_test,Class=2)
# testFeatures=np.vstack((testFeatures_yt,testFeatures_browsing,testFeatures_mining))

# testFeatures_ytS,oClass_yt=extractFeaturesSilence(yt_test,Class=0)
# testFeatures_browsingS,oClass_browsing=extractFeaturesSilence(browsing_test,Class=1)
# testFeatures_miningS,oClass_mining=extractFeaturesSilence(mining_test,Class=2)
# testFeaturesS=np.vstack((testFeatures_ytS,testFeatures_browsingS,testFeatures_miningS))

# testFeatures_ytW,oClass_yt=extractFeaturesWavelet(yt_test,scales,Class=0)
# testFeatures_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_test,scales,Class=1)
# testFeatures_miningW,oClass_mining=extractFeaturesWavelet(mining_test,scales,Class=2)
# testFeaturesW=np.vstack((testFeatures_ytW,testFeatures_browsingW,testFeatures_miningW))

# o3testClass=np.vstack((oClass_yt,oClass_browsing,oClass_mining))
# i3testFeatures=np.hstack((testFeatures,testFeaturesS,testFeaturesW))

## -- 9 -- ##
#from sklearn.preprocessing import MaxAbsScaler

# i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
# i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

# i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)  
# i3trainFeaturesN=i3trainScaler.transform(i3trainFeatures)

# i3AtestFeaturesN=i2trainScaler.transform(i3testFeatures)
# i3CtestFeaturesN=i3trainScaler.transform(i3testFeatures)

# print(np.mean(i2trainFeaturesN,axis=0))
# print(np.std(i2trainFeaturesN,axis=0))

## -- 10 -- ##
# from sklearn.decomposition import PCA

# pca = PCA(n_components=3, svd_solver='full')

# i2trainPCA=pca.fit(i2trainFeaturesN)
# i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

# i3trainPCA=pca.fit(i3trainFeaturesN)
# i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

# i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)
# i3CtestFeaturesNPCA = i3trainPCA.transform(i3CtestFeaturesN)

# plt.figure(8)
# plotFeatures(i2trainFeaturesNPCA,o2trainClass,0,1)

## -- 11 -- ##
from sklearn.preprocessing import MaxAbsScaler
centroids={}
for c in range(2):  # Only the first two classes
    pClass=(o2trainClass==c).flatten()
    centroids.update({c:np.mean(i2trainFeaturesN[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

AnomalyThreshold=1.2
print('\n-- Anomaly Detection based on Centroids Distances --')
nObsTest,nFea=i3AtestFeaturesN.shape
for i in range(nObsTest):
    x=i3AtestFeaturesN[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
    else:
        result="OK"
       
    print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))

## -- 12 -- ##
centroids={}
for c in range(2):  # Only the first two classes
    pClass=(o2trainClass==c).flatten()
    centroids.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
print('All Features Centroids:\n',centroids)

AnomalyThreshold=1.2
print('\n-- Anomaly Detection based on Centroids Distances (PCA Features) --')
nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3AtestFeaturesNPCA[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1])]
    if min(dists)>AnomalyThreshold:
        result="Anomaly"
    else:
        result="OK"
       
    print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*dists,result))


## -- 13 -- ##
from scipy.stats import multivariate_normal
print('\n-- Anomaly Detection based Multivariate PDF (PCA Features) --')
means={}
for c in range(2):
    pClass=(o2trainClass==c).flatten()
    means.update({c:np.mean(i2trainFeaturesNPCA[pClass,:],axis=0)})
#print(means)

covs={}
for c in range(2):
    pClass=(o2trainClass==c).flatten()
    covs.update({c:np.cov(i2trainFeaturesNPCA[pClass,:],rowvar=0)})
#print(covs)

AnomalyThreshold=0.05
nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3AtestFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1])])
    if max(probs)<AnomalyThreshold:
        result="Anomaly"
    else:
        result="OK"
    
    print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e}] -> Result -> {}'.format(i,Classes[o3testClass[i][0]],*probs,result))


## -- 14 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesNPCA)  

L1=ocsvm.predict(i3AtestFeaturesNPCA)
L2=rbf_ocsvm.predict(i3AtestFeaturesNPCA)
L3=poly_ocsvm.predict(i3AtestFeaturesNPCA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

## -- 15 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesN)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesN)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesN)  

L1=ocsvm.predict(i3AtestFeaturesN)
L2=rbf_ocsvm.predict(i3AtestFeaturesN)
L3=poly_ocsvm.predict(i3AtestFeaturesN)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesN.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))


## -- 16 -- ##
centroids={}
for c in range(3):  # All 3 classes
    pClass=(o3trainClass==c).flatten()
    centroids.update({c:np.mean(i3trainFeaturesNPCA[pClass,:],axis=0)})
print('PCA Features Centroids:\n',centroids)

print('\n-- Classification based on Centroids Distances (PCA Features) --')
nObsTest,nFea=i3CtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3CtestFeaturesNPCA[i]
    dists=[distance(x,centroids[0]),distance(x,centroids[1]),distance(x,centroids[2])]
    ndists=dists/np.sum(dists)
    testClass=np.argsort(dists)[0]
    
    print('Obs: {:2} ({}): Normalized Distances to Centroids: [{:.4f},{:.4f},{:.4f}] -> Classification: {} -> {}'.format(i,Classes[o3testClass[i][0]],*ndists,testClass,Classes[testClass]))


## -- 17-- # 
from scipy.stats import multivariate_normal
print('\n-- Classification based on Multivariate PDF (PCA Features) --')
means={}
for c in range(3):
    pClass=(o3trainClass==c).flatten()
    means.update({c:np.mean(i3trainFeaturesNPCA[pClass,:],axis=0)})
#print(means)

covs={}
for c in range(3):
    pClass=(o3trainClass==c).flatten()
    covs.update({c:np.cov(i3trainFeaturesNPCA[pClass,:],rowvar=0)})
#print(covs)

nObsTest,nFea=i3CtestFeaturesNPCA.shape
for i in range(nObsTest):
    x=i3CtestFeaturesNPCA[i,:]
    probs=np.array([multivariate_normal.pdf(x,means[0],covs[0]),multivariate_normal.pdf(x,means[1],covs[1]),multivariate_normal.pdf(x,means[2],covs[2])])
    testClass=np.argsort(probs)[-1]
    
    print('Obs: {:2} ({}): Probabilities: [{:.4e},{:.4e},{:.4e}] -> Classification: {} -> {}'.format(i,Classes[o3testClass[i][0]],*probs,testClass,Classes[testClass]))


## -- 18 -- #
print('\n-- Classification based on Support Vector Machines --')
svc = svm.SVC(kernel='linear').fit(i3trainFeaturesN, o3trainClass)  
rbf_svc = svm.SVC(kernel='rbf').fit(i3trainFeaturesN, o3trainClass)  
poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3trainFeaturesN, o3trainClass)  

L1=svc.predict(i3CtestFeaturesN)
L2=rbf_svc.predict(i3CtestFeaturesN)
L3=poly_svc.predict(i3CtestFeaturesN)
print('\n')

nObsTest,nFea=i3CtestFeaturesN.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))


## -- 19 -- #
print('\n-- Classification based on Support Vector Machines  (PCA Features) --')
svc = svm.SVC(kernel='linear').fit(i3trainFeaturesNPCA, o3trainClass)  
rbf_svc = svm.SVC(kernel='rbf').fit(i3trainFeaturesNPCA, o3trainClass)  
poly_svc = svm.SVC(kernel='poly',degree=2).fit(i3trainFeaturesNPCA, o3trainClass)  

L1=svc.predict(i3CtestFeaturesNPCA)
L2=rbf_svc.predict(i3CtestFeaturesNPCA)
L3=poly_svc.predict(i3CtestFeaturesNPCA)
print('\n')

nObsTest,nFea=i3CtestFeaturesNPCA.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],Classes[L1[i]],Classes[L2[i]],Classes[L3[i]]))
    

## -- 20a -- ##
from sklearn.neural_network import MLPClassifier
print('\n-- Classification based on Neural Networks --')

alpha=1
max_iter=100000
clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
clf.fit(i3trainFeaturesN, o3trainClass) 
LT=clf.predict(i3CtestFeaturesN) 

nObsTest,nFea=i3CtestFeaturesN.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))

## -- 20b -- ##
from sklearn.neural_network import MLPClassifier
print('\n-- Classification based on Neural Networks (PCA Features) --')

alpha=1
max_iter=100000
clf = MLPClassifier(solver='lbfgs',alpha=alpha,hidden_layer_sizes=(20,),max_iter=max_iter)
clf.fit(i3trainFeaturesNPCA, o3trainClass) 
LT=clf.predict(i3CtestFeaturesNPCA) 

nObsTest,nFea=i3CtestFeaturesNPCA.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Classification->{}'.format(i,Classes[o3testClass[i][0]],Classes[LT[i]]))