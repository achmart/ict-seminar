#!UTF-8


#Dataset -- Kaggle crime statistics
#crim	tax	rm	age	ptratio	medv
training_dataset = """
0.00632	296	6.575	65.2	15.3	24
0.02731	242	6.421	78.9	17.8	21.6
0.03237	222	6.998	45.8	18.7	33.4
0.06905	222	7.147	54.2	18.7	36.2
0.08829	311	6.012	66.6	15.2	22.9
0.22489	311	6.377	94.3	15.2	15
0.11747	311	6.009	82.9	15.2	18.9
0.09378	311	5.889	39	15.2	21.7
0.62976	307	5.949	61.8	21	20.4
"""

training_dataset = [[float(f) for f in i.split("\t")] for i in  training_dataset.strip().split("\n")]

#Making into a binary classifier
training_dataset = [row[:-1]+[0 if row[-1]<20 else 1] for row in training_dataset]

#TODO: Download the entire dataset




#Cheating to make the dataset bigger

training_dataset = training_dataset*100

import numpy as np
X = np.array([i[0:5] for i in training_dataset])
Y = np.array([i[5] for i in training_dataset])



#Real data:

#import csv
#a = csv.reader(open("./HouseData/all/train.csv","r"))
#firstrow = []

#X = []
#Y = []
#firstrow = True
#for row in a:
#    if(firstrow):
#        firstrow = False
#        continue
#    X.append([float(i) for i in row[1:-1]])
#    if(float(row[-1])<20):
#        Y.append(1)
#    else:
#        Y.append(0)
#    #Y.append(float(row[-1]))
#X = np.array(X)
#Y = np.array(Y)
#import pdb;pdb.set_trace()
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import optimizers

np.random.seed(7)
model = Sequential()
model.add(Dense(5,input_dim=(len(X[0]))))
model.add(Dense(2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
#Standard learning rate = 0.1
model.compile(loss="mean_squared_error",optimizer="adam")


model.fit(X, Y, epochs=1000)
