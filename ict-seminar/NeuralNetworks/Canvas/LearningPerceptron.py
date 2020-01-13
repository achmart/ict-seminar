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
#Rock or not rock

rocks = [
        [2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0]
        ]



not_rocks = [
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]
        
        ]

#Random dataset
import random
rocks = [ [random.uniform(-5,1),random.uniform(-5,1),0] for i in range(100)]
not_rocks = [ [random.uniform(-1,5),random.uniform(-1,5),0] for i in range(100)]

#import pdb;pdb.set_trace()
training_dataset = rocks[:int(len(rocks)/2)] + not_rocks[:int(len(not_rocks)/2)]
validation_dataset = rocks[int(len(rocks)/2):] + not_rocks[int(len(not_rocks)/2):]



#Plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np_rocks = np.array(rocks)
np_not_rocks = np.array(not_rocks)

fig, ax = plt.subplots()
ax.plot(np_rocks[:,0],np_rocks[:,1],'ro')
ax.plot(np_not_rocks[:,0],np_not_rocks[:,1],'go')
#ax.plot(np.array(rocks))

ax.grid()
plt.show()

#weights = [-0.1, 0.20653640140000007, -0.23418117710000003]



# Estimate Perceptron weights using stochastic gradient descent
#def train_weights(train, l_rate, n_epoch):
#	weights = [0.0 for i in range(len(train[0]))]
#	for epoch in range(n_epoch):
#		sum_error = 0.0
#		for row in train:
#			prediction = predict(row, weights)
#			error = row[-1] - prediction
#			sum_error += error**2
#			weights[0] = weights[0] + l_rate * error
#			for i in range(len(row)-1):
#				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
#		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
#	return weights

#weights = [bias, w2, w2]

#def predict(row, weights):
#	activation = weights[0]
#	for i in range(len(row)-1):
#		activation += weights[i + 1] * row[i]
#	return 1.0 if activation >= 0.0 else 0.0

def predict(row,weights):
    #activation = (w1 * X1) + (w2 * X2) + bias
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i+1]*row[i]
    return 1.0 if activation >= 0.0 else 0.0

import random

def train_weights(train, learningrate, epochs):
    weights = [random.uniform(-1,1) for i in range(len(train[0]))]
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            prediction = predict(row,weights)
            error = row[-1]-prediction
            sum_error += error**2
            weights[0] = weights[0] + learningrate*error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + learningrate*error*row[i]
        print("Epoch"+str(epoch) + "Learning rate" + str(learningrate) + " Error" + str(sum_error))
    return weights

learningrate = 0.5#0.00001
epochs = 10
train_weights = train_weights(training_dataset,learningrate,epochs)
print(train_weights)

accuracy = 0.0
for row in validation_dataset:
        prediction = predict(row, train_weights)
        if(prediction==row[-1]):
             accuracy += 1
        #print("Expected=%d, Predicted=%d" % (row[-1], prediction))
accuracy = accuracy/len(validation_dataset)
print("Accurary",accuracy)
