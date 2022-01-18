'''DO NOT DELETE ANY PART OF CODE
We will run only the evaluation function.

Do not put anything outside of the functions, it will take time in evaluation.
You will have to create another code file to run the necessary code.
'''

# import statements

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsne
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# other functions

def get_LabelCount(labels,clusters,index):
    
    label_count=[0,0,0,0,0,0,0]
    
    for i in range(len(clusters)):
        if(clusters[i]==index):
            label_count[labels[i]]= label_count[labels[i]] + 1
    
    #print(index)
    #print(label_count)
    return label_count

def get_Mapping(labels,clusters):
    
    cluster_labelsCount=[]
    
    for i in range(7):
        labelCount=get_LabelCount(labels,clusters,i)        
        cluster_labelsCount.append(labelCount)
        
    #print( cluster_labelsCount)
    
    mapping=[0,0,0,0,0,0,0]
    
    for i in range(7):
        lb_count=cluster_labelsCount[i]
        
        max_count= max(lb_count)
        max_index = lb_count.index(max_count)
        
        mapping[i]=max_index
        
    return mapping

def get_Mapping1(labels,clusters):
    
    cluster_labelsCount=[]
    
    for i in range(7):
        labelCount=get_LabelCount(labels,clusters,i)        
        cluster_labelsCount.append(labelCount)
        
    #print( cluster_labelsCount)
    
    mapping=[0,0,0,0,0,0,0]
    
    for label in range(7):
        
        max_count=0
        max_cluter=0
        for cluster_index in range(len(cluster_labelsCount)):
            val=cluster_labelsCount[cluster_index][label]
            if val>max_count:
                max_count=cluster_labelsCount[cluster_index][label]
                max_cluster= cluster_index
                
        
        mapping[label]=max_cluster
        
    return mapping

def predict(test_set) :
    # find and load your best model
    # Do all preprocessings inside this function only.
    # predict on the test set provided
    '''
    'test_set' is a csv path "test.csv", You need to read the csv and predict using your model.
    '''
    
    '''
    prediction is a 1D 'list' of output labels. just a single python list.
    '''
    
    test = pd.read_csv(test_set)
    data= pd.read_csv('covtype_train.csv')
    
    label_encoder = preprocessing.LabelEncoder()
    data['Elevation']= label_encoder.fit_transform(data['Elevation'])
    data['Aspect']= label_encoder.fit_transform(data['Aspect'])
    data['Slope']= label_encoder.fit_transform(data['Slope'])
    data['Wilderness']= label_encoder.fit_transform(data['Wilderness'])
    data['Soil_Type']= label_encoder.fit_transform(data['Soil_Type'])
    data['Hillshade_9am']= label_encoder.fit_transform(data['Hillshade_9am']) 
    data['Hillshade_Noon']= label_encoder.fit_transform(data['Hillshade_Noon']) 
    data['Vertical_Distance_To_Hydrology']= label_encoder.fit_transform(data['Vertical_Distance_To_Hydrology']) 
    data['Horizontal_Distance_To_Fire_Points']= label_encoder.fit_transform(data['Horizontal_Distance_To_Fire_Points']) 
    
    test['Elevation']= label_encoder.fit_transform(test['Elevation'])
    test['Aspect']= label_encoder.fit_transform(test['Aspect'])
    test['Slope']= label_encoder.fit_transform(test['Slope'])
    test['Wilderness']= label_encoder.fit_transform(test['Wilderness'])
    test['Soil_Type']= label_encoder.fit_transform(test['Soil_Type'])
    test['Hillshade_9am']= label_encoder.fit_transform(test['Hillshade_9am']) 
    test['Hillshade_Noon']= label_encoder.fit_transform(test['Hillshade_Noon']) 
    test['Vertical_Distance_To_Hydrology']= label_encoder.fit_transform(test['Vertical_Distance_To_Hydrology']) 
    test['Horizontal_Distance_To_Fire_Points']= label_encoder.fit_transform(test['Horizontal_Distance_To_Fire_Points']) 
    
    data.target = data.target - 1
    
    features = data.loc[ : , data.columns != 'target']

    labels = data['target']
    
    test_features=  test.loc[:, test.columns != 'id']


    
    print(features)
    print(test_features)
    #print(labels)
    
    ## Training the model and getting the mappings 
    kmeans = KMeans(n_clusters=7) 
    kmeans.fit(features)
    
    prediction = kmeans.predict(features,labels)
    
    cluster = list(prediction)
    
    # print(len(cluster))
    # print(len(labels))
    
    # if(len(cluster)==len(labels)):
    #     print("equal")
    # else :
    #     print("not-equal")
    mapping_list=get_Mapping(labels,cluster)
    
    print(mapping_list)
    
    ## training on the data and predicting on the test set
    ## using the mapping gained from training data to convert predictions to labels
    
    prediction = kmeans.predict(test_features)

    import pickle

    dbfile = open('Clutering_trained_Model', 'ab')

    # source, destination
    pickle.dump(kmeans, dbfile) 
    dbfile.close()

    print("Model Saved")
    
    
    
    
    for i in range(len(prediction)):
        lb = prediction[i]
        prediction[i] = mapping_list[lb]
        prediction[i]=prediction[i]+1
  
    return prediction


pred= predict('test_set.csv')

df = pd.DataFrame(columns=['id','target'])



for i in range(len(pred)):
	# df['id'].append(i)
	# df['target'].append(pred[i])
	ls= [i, pred[i]]
	print(ls)
	df.loc[len(df.index)] = [i, pred[i]]
	# df.at[i, 'id'] = i
	# df.at[i, 'target'] = pred[i]

df.to_csv('2019315_2019395.csv', index=None)

for i in range(100):
	print(pred[i])
	print("\n")


print(df)