# Make a prediction with weights

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

dataset = rocks + not_rocks



#Plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np_rocks = np.array(rocks)
np_not_rocks = np.array(not_rocks)

fig, ax = plt.subplots()
ax.plot(np_rocks[:,0],np_rocks[:,1],'ro')
ax.plot(np_not_rocks[:,0],np_not_rocks[:,1],'go')

ax.grid()
plt.show()

weights = [-0.1, 0.20, -0.23]

def predict(row,weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i+1]*row[i]
    return 1.0 if activation >= 0.0 else 0.0

for row in dataset:
	prediction = predict(row, weights)
	print("Expected=%d, Predicted=%d" % (row[-1], prediction))
