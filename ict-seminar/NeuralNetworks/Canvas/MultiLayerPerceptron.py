#!UTF-8
# Make a prediction with weights
#def predict(row, weights):
#	activation = weights[0]
#	for i in range(len(row)-1):
#		activation += weights[i + 1] * row[i]
#	return 1.0 if activation >= 0.0 else 0.0
 
# test predictions
#dataset = [[2.7810836,2.550537003,0],
#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]


#weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

#for row in dataset:
#	prediction = predict(row, weights)
#	print("Expected=%d, Predicted=%d" % (row[-1], prediction))


#Dataset
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
#import pdb;pdb.set_trace()





#Plotting
#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np
#np_rocks = np.array(rocks)
#np_not_rocks = np.array(not_rocks)

#fig, ax = plt.subplots()
#ax.plot(np_rocks[:,0],np_rocks[:,1],'ro')
#ax.plot(np_not_rocks[:,0],np_not_rocks[:,1],'go')
#ax.plot(np.array(rocks))

#ax.grid()
#plt.show()

#weights = [w0, w1, w2, w3, w4, w5, w6, w7, w8]
weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]

#Random weights
import random
weights = [random.uniform(-1,1) for i in weights]


import math
def sigmoid(z):
    if(z<100):
        return 0
    if(z>100):
        return 1
    return 1.0/(1+math.exp(-z))


def firstLayer(row, weights):
    activation_1 = weights[0]
    
    activation_1 += weights[1]*row[0]
    activation_1 += weights[2]*row[1]

    activation_2 = weights[3]
    activation_2 += weights[4]*row[2]
    activation_2 += weights[5]*row[3]
    return sigmoid(activation_1),sigmoid(activation_2)


def secondLayer(row,weights):
    activation_3 = weights[6]
    activation_3 += weights[7]*row[0]
    activation_3 += weights[8]*row[1]
    return sigmoid(activation_3)
    #return 1.0 if activation_3 >= 0.0 else 0.0

def predict(row,weights):
    input_layer = row
    first_layer = firstLayer(row,weights)
    second_layer = secondLayer(first_layer,weights)
    return second_layer,first_layer

for d in training_dataset:
    print(predict(d,weights)[0],d[-1])

#import sys
#sys.exit(0)



def train_weights(train, learningrate, epochs):
    #weights = [random.uniform(-1,1) for i in range(len(train[0]))]
    last_error = 0.0
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            prediction,first_layer = predict(row,weights)
            error = row[-1]-prediction
            #print(error)
            sum_error += error**2#abs(error)#math.abs(error)#**2**0.5

            #First layer
            weights[0] = weights[0] + learningrate*error
            weights[3] = weights[3] + learningrate*error

            weights[1] = weights[1] + learningrate*error*row[0]
            weights[2] = weights[2] + learningrate*error*row[1]
            weights[4] = weights[4] + learningrate*error*row[2]
            weights[5] = weights[5] + learningrate*error*row[3]

            #Second layer
            weights[6] = weights[6] + learningrate*error
            weights[7] = weights[7] + learningrate*error*first_layer[0]
            weights[8] = weights[8] + learningrate*error*first_layer[1]
            
            #for i in range(len(row)-1):
            #    weights[i+1] = weights[i+1] + learningrate*error*row[i]
        if((epoch%100==0) or (last_error != sum_error)):
            print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
        last_error = sum_error
    return weights

learningrate = 0.01#0.00001
epochs = 10000
train_weights = train_weights(training_dataset,learningrate,epochs)
print(train_weights)


accuracy = 0.0
for row in training_dataset:
    prediction = predict(row,weights)
    print("Prediction:",prediction[0]," Real value:", row[-1])
    print("Error:",prediction[0]-row[-1])
    if(prediction[0]==row[-1]):
        accuracy += 1


accuracy = accuracy/len(training_dataset)
print("Accurary",accuracy)
