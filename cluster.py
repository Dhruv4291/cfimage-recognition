# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:02:06 2018

@author: dhruv.mahajan
"""
import boto3
import pandas as pd  
import numpy as np
from google.cloud import vision
from google.cloud.vision import types
import io
import pandas as pd
from google.oauth2 import service_account
import boto3
import pickle
import pandas as pd
import numpy as np
import pylab as pl
import numpy as np
import botocore 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.semi_supervised import label_propagation
from scipy import stats
import sklearn

idkeysfinal = pickle.load(open("idkeysfinal", "rb")) #keys of labeled ids

shopkeysfinal = pickle.load(open("shopkeysfinal", "rb")) #keys of labeled shop

boardkeys = pickle.load(open("boardkeys", "rb")) #keys of labeled board
activitykeys = pickle.load(open("activitykeys", "rb"))
selfiekeys = pickle.load(open("selfiekeys", "rb"))
stockkeys = pickle.load(open("stockkeys", "rb"))

#THESE WILL BE YOUR OWN
googlelabels = pickle.load(open("googlelabels", "rb"))

amazonlabels = pickle.load(open("amazonlabels", "rb"))

panaadhar = pickle.load(open("Google_Aadhar_PAN_image-recognitioncf.p", "rb"))

panaadharg = pickle.load(open("Google_Aadhar_PAN_image-recognitioncf.p", "rb"))

panaadhar = pickle.load(open("app_aadhar_pan_image-recognitioncf.p", "rb"))


akey = get_akey_labelamazon("imagecluster1.cf")[0]
amazonlabels = get_akey_labelamazon("imagecluster1.cf")[1]
googlelabels = get_labelgoogle("imagecluster1.cf")
modelimages = clusterimages(amazonlabels,googlelabels,akey) #0-1 model for shop/non-shop
modelshop = clustershops(amazonlabels,googlelabels)#different type of shops

#function to get keys(file names) and labels from amazon rekognition taking input as bucket name
def get_akey_labelamazon(repository):
    s3 = boto3.resource('s3') #creating a boto3 resource
    bucket1 = s3.Bucket(repository) #(our  bucket in s3)
    
    akey = [] #the keys
    #passing the file names to an array akey
    for key in bucket1.objects.all():
        akey.append(str(key.key))
        
    
    #creating a rekognition client
    client = boto3.client('rekognition')
    #iterating over the  images
    labels = []
    for i in range(len(akey)):
    	try :
			labels.append(client.detect_labels(Image={'S3Object': 
	            {'Bucket':repository ,'Name': akey[i],},}, 
	        MaxLabels = 20,
	        MinConfidence=50,)['Labels'])
		except:
			labels.append('error')
			akey[i] = 'error'
	
	labels = [x for x in labels if labels!= "error"]
	akey = [x for x in akey if akey!= "error"]
    return(akey,labels)     

#function to get akey and url of the images in a bucket
#the akey will remain same so you dont need to save it again, just save the urls    
def get_akey_url_google(repository):
    s3 = boto3.resource('s3')
    bucket1 = s3.Bucket(repository) #(our aadhar bucket in s3)
    client = boto3.client('s3')
    akey = []
    #passing the file names to an array akey
    for key in bucket1.objects.all():
        akey.append(str(key.key))
    
    url = []
    a_url = []
    for key in akey:
        a_url = client.generate_presigned_url(
                'get_object', Params={'Bucket': repository, 'Key': key})
        url.append(a_url) #urls of the images in our bucket
    return(url,akey)
    
        

#function of google vision api to get labels
def detect_labels_google(uri):
    """Detects labels in the file located in Google Cloud Storage or on the
    Web."""
    #setting up the google vision api
    credentials = service_account.Credentials.from_service_account_file('C:/Users/dhruv.mahajan/Downloads/Geocoding-8ac3b1c0e30b.json')        
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    client = vision.ImageAnnotatorClient(credentials=scoped_credentials)
    
   #running the google vision api for getting labels 
    image = types.Image()
    image.source.image_uri = uri
    response = client.label_detection(image=image)
    labels = response.label_annotations
    description = [] #description is basically the labels
    score = [] #score is basically the confidence
    for label in labels:
        description.append(label.description)
        score.append(label.score)
    
    labels = [description, score]
    return(labels)    

#function to run loop over all the images 
def get_labelgoogle(repository):
    urls = get_akey_url_google(repository)[0] #urls from our previous function

    Response = []
    
    for url in urls:
        try:
            Response.append(detect_labels_google(url))
        except:
            Response.append('error')
	Response = [x for x in Response if Response!= "error"]
    return(Response)

#getting the z-matrix for google taking labels and keys as input    
def get_zmatrix_google(labels,keys):
    columns = ['Index','Label','Confidence']
    d0 = pd.DataFrame(columns=columns)
    for i in range(len(labels)):
        #passing the index as file name, labels as labels and confidence as confidence of various labels
        d = {'Index': keys[i] ,'Label': labels[i][0], 'Confidence': labels[i][1]}
        #sometimes the apis return empty list, so to debug it we use this
        if (d['Confidence'] == []):
            d['Confidence'] = [0]
            d['Label'] = ['NULL']
        df = pd.DataFrame(data=d)
        frames = [d0, df]
        #concating frames again and again until we run over the loop
        result = pd.concat(frames)                      
        d0 = result
    d0['Confidence'] = 100*d0['Confidence']
    #creating a pivot table
    d0 = pd.pivot_table(d0,values='Confidence',index='Index',columns='Label')
    x = d0.isnull().sum()/d0.shape[0]
    x = x[x <0.9] #thresholding
    d0 = d0[x.index]
    d0 = d0.fillna(0) #filling nas with zero
    return(d0)


#getting the z-matrix for amazon taking labels and keys as input 
    
def get_zmatrix_amazon(labels,keys):
    columns = ['Index','Label','Confidence']
    d0 = pd.DataFrame(columns=columns)
    d0 = d0.fillna(0)                
    names = []
    names2 = []
    for i in range(len(labels)):
        names_tmp = []
        names_tmp2 = []
        for j in labels[i]:
            names_tmp.append(j['Name'])#(k,v) for k,v in j.items()
            names_tmp2.append(j['Confidence'])
        d = {'Index': keys[i] ,'Label': names_tmp, 'Confidence': names_tmp2}
        if (d['Confidence'] == []):
            d['Confidence'] = [0]
            d['Label'] = ['NULL']
        df = pd.DataFrame(data=d)
        frames = [d0, df]
        result = pd.concat(frames)                      
        d0 = result 
    return(d0)

#final model function to cluster as 0(id) or 1( not- id)
#panaadhara and panaadharg will come from atal's function for extracting pan from amazon and google respectively

# to run a semisupervised algorithm we need two things :
    #1) the z matrix 
    #2) an arr with labels assgined to the known elements and -1 assigned to unknown elements
def clusterimages(labelsamazon,labelsgoogle,akey):
    arr = []
    for i in range(len(akey)):
        arr.append(-1)
    
    indices = [i for i, x in enumerate(akey) if x in idkeysfinal] #getting the index for ids in akey
    for i in indices:
        arr[i] = 0 #assigning ids as 0 in our array
    indices = [i for i, x in enumerate(akey) if x in shopkeysfinal]#getting the index for shops in akey
    for i in indices: #assigning shops as 1 in our array
        arr[i] = 1
    
    aadhara = panaadhar['aadhar'] 
    aadhara = list(aadhara) #getting aadhar
    aadharg = panaadharg['aadhar']
    aadharg = list(aadharg)
    pang = panaadharg['pan']
    pang = list(pang) #getting pan
    pana = panaadhar['pan']
    pana = list(pana)
    #getting indices of pan and aadhar where they are present
    indicesaa = [i for i, x in enumerate(aadhara) if x != "error"]
    indicesga =[i for i, x in enumerate(aadharg) if x != "error"]
    indicesap = [i for i, x in enumerate(pana) if x != "error"] 
    indicesgp = [i for i, x in enumerate(pang) if x != "error"]
    
    #assigning and updating our array with the newly found labels of shops
    for i in range(len(akey)):
        if(i in indicesaa ):
            arr[i] = 0
        if(i in indicesap ):
            arr[i] = 0
        if( i in indicesga):
            arr[i] = 0
        if ( i in indicesgp):
            arr[i] == 0
    #getting the z-matrices for amazon and google
    z1 = get_zmatrix_amazon(labelsamazon,akey)
    z2 = get_zmatrix_google(labelsgoogle,akey) 
    z = pd.concat([z1,z2],axis= 1)
    
    lp_model = sklearn.semi_supervised.LabelSpreading(kernel = 'knn',n_neighbors = 125)
    #auc = 0.93
    lp_model.fit(z,arr)
    indices = [i for i, x in enumerate(arr) if x == -1]
    predicted_labels = lp_model.transduction_[indices]
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T) #predicting entropy
    uncertainty_index = np.argsort(pred_entropies)[-100:] #worst 100 entropies
    keys=[] #keys for the unlabeled data
    for i in indices:
        keys.append(akey[i])
    
    z = z.ix[keys]
    silhouette = metrics.silhouette_score(z,predicted_labels , metric='euclidean') #finding the silhoutte score
    predgoogle = pd.DataFrame({'Keys' : keys, 'Predictedlabels' :predicted_labels })
    keys1 = predgoogle[predgoogle["Predictedlabels"] == 1]
    keys1 = keys1['Keys'].tolist()
    return(predicted_labels,silhouette,uncertainty_index,pred_entropies,keys1)
#keys 1 is the keys of shops predicted from the above function    

#Model to cluster shops into different catergories    
def clustershops(labelsamazon,labelsgoogle):
    shopkeys = clusterimages(labelsamazon,labelsgoogle)[4]#getting the shopkeys
    shopkeyslabeled = shopkeysfinal #labeled keys from my saved data   #513
    shopkeys = shopkeyslabeled + shopkeys
    indicesboard = [i for i, x in enumerate(shopkeys) if x in boardkeys] #incides of board
    indicesactivity = [i for i, x in enumerate(shopkeys) if x in activitykeys]
    indicesselfie = [i for i, x in enumerate(shopkeys) if x in selfiekeys]
    indicesstock = [i for i, x in enumerate(shopkeys) if x in stockkeys]

    arr = []
    for i in range(len(shopkeys)):
        arr.append(-1)
    #updating array with labels
    for i in indicesboard:
        arr[i] = 0
    for i in indicesactivity:
        arr[i] = 1    
    for i in indicesselfie:
        arr[i] = 2    
    for i in indicesstock:
        arr[i] = 3
    #so that we dont have to run the label function again
    #we just subset the previous saved labels using the keys
    indices = [i for i, x in enumerate(akey) if x in shopkeys]
    shopamazon = []
    for i in indices:
        shopamazon.append(amazonlabels[i])
    shopgoogle = []
    for i in indices:
        shopgoogle.append(googlelabels[i])
    
    z1 = get_zmatrix_amazon(shopamazon,shopkeys)
    z2 = get_zmatrix_google(shopgoogle,shopkeys)
    z = pd.concat([z1,z2],axis= 1)
    lp_model = label_propagation.LabelSpreading(kernel = 'knn',alpha = 0.8,n_neighbors = 10)
    #acc= 0.6, kappa  = 0.47
    lp_model.fit(z,arr)
    indices = [i for i, x in enumerate(arr) if x == -1]
    predicted_labels = lp_model.transduction_[indices]
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
    uncertainty_index = np.argsort(pred_entropies)[-100:]
    keys=[]
    for i in indices:
        keys.append(shopkeys[i])
    silhouette = metrics.silhouette_score(z,predicted_labels , metric='euclidean')
    return(predicted_labels,silhouette,uncertainty_index,pred_entropies)

#TUNING    
alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
n = [10,20,50.....]
silhoutte = []
lp_model = []
for i in range(len(n or alpha)):
    #MODEL for different n and alphas
    #Do tuning for n and alpha seperately
    lp_model.append(sklearn.semi_supervised.LabelSpreading(kernel = 'knn',n_neighbors = n,alpha = alpha)
    #You will have to save z and arr, which will be from either clusterimages or clustershops
    lp_model[i].fit(z,arr)
    indices = [i for i, x in enumerate(arr) if x == -1]
    predicted_labels = lp_model[i].transduction_[indices]
    silhoutte.append(metrics.silhouette_score(z,predicted_labels , metric='euclidean'))

#downloading the files into folders
#set the defualt folder of spyder in which you need to store it	

for key in keystest:
   s3.Bucket('image-recognitioncf').download_file(key, key)
   
   
dockeys = []   
for i in docindices:
	dockeys.append(akey[i])
