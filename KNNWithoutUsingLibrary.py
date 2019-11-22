import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from numpy.random import RandomState
import math
#The program implements the KNN without using the library
#Split the given data into training and testing
#'pima.csv' file contains column labels as x1 to x8 for features and y for output
#Only x2, x3, x4, and y features are selected and split into X_train, X_test, y_train, and y_test as lists  
def SplitData():
    df = pd.read_csv("pima.csv")
    df=df.drop(columns=['x1','x5','x6','x7','x8'])
    X=df.drop(columns=['y']).values.tolist()
    y=df.drop(columns=['x2','x3','x4']).values.reshape(-1).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
    return (X_train, X_test, y_train, y_test)

#to calculate the euclidean distance between train row and test row
def findDistance(testRow, trainRow):
    dist= ((testRow[0]-trainRow[0])**2+(testRow[1]-trainRow[1])**2+(testRow[2]-trainRow[2])**2)
    return (math.sqrt(dist))

#find the K closest point fron train data for a single test data
def findClass(testRow, X_train, K):
    eDistances=[]
    #find all eulidean distances
    i=0
    for trainRow in X_train:
        eDistances.append([findDistance(testRow,trainRow),y_train[i]])
        i+=1        
    eDistances=sorted(eDistances,key=lambda x: x[0])
    class0=0
    class1=0
    #calculate k closest point for the given testRow point
    for i in range(K):
        if(eDistances[i][1]==0):
            class0+=1
        else:
            class1+=1
    if class0>class1:
        #given train point belongs to class 0
        return(0)
    else:
        #given train point belongs to class 1
        return(1)

#The function calculates y_pred for predicted outputs
def KNNClassifier(X_train, X_test, y_train, y_test, K):
    y_pred=[]
    for row in X_test:
        output=findClass(row, X_train, K)
        y_pred.append(output)
    return(y_pred)

#calculates total correctly and incorrectly classified data by comparing y_pred and y_test lists
def findAccuracy(y_test, y_pred):
    #correct for total correct classification and wrong for total incorrect classification
    correct=0
    wrong=0
    for i in range(len(y_pred)):
        global X_pred
        if(y_test[i]==y_pred[i]):
            correct+=1
        else:
            wrong+=1
    return(correct, wrong)

listAccuracy=[]
for K in [1,5,11]:
    for i in range(10):
        X_train, X_test, y_train, y_test=SplitData()
        y_pred=KNNClassifier(X_train, X_test, y_train, y_test, K)
        correct, wrong= findAccuracy(y_test, y_pred)
        listAccuracy.append(float(correct)/(correct+wrong)*100)
        meanAccuracy=np.average(listAccuracy)
        sd=np.std(listAccuracy)
    print "\nDetails for K= ", K
    #print "List of Accuracy for correct classification in percentage\n", listAccuracy
    print "Mean accuracy= ", meanAccuracy
    print "Standard Deviation= ",sd 

#Random OUTPUT
# Details for K= 1
# Mean accuracy= 65.44270833333333
# Standard Deviation= 1.265367772214202

# Details for K= 5
# Mean accuracy= 67.72135416666667
# Standard Deviation= 2.6766919197484267

# Details for K= 11Mean accuracy= 69.07118055555556
# Standard Deviation= 3.0329940612374333

#'pima.csv' file outline
# x1,x2,x3,x4,x5,x6,x7,x8,y
# 6,148,72,35,0,33.6,0.627,50,1
# 1,85,66,29,0,26.6,0.351,31,0
# 8,183,64,0,0,23.3,0.672,32,1
# 1,89,66,23,94,28.1,0.167,21,0
# 0,137,40,35,168,43.1,2.288,33,1