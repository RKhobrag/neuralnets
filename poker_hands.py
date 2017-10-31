import math
import pickle
import numpy as np


def getFeatureMatrix(file_path):
	data = np.genfromtxt(file_path, delimiter=',', skip_header=1,usecols=range(0,10), dtype='int', max_rows=100000)
	data= data
	np.c_[data,	np.ones(data.shape[0])]
	return data 

def getLabels(file_path):
	data = np.genfromtxt(file_path, delimiter=',', skip_header=1,usecols=range(10,11), dtype='int', max_rows=100000)
	return data


def sigmoid_(x):
		return 1/(1+np.exp(-x))

def d_sigmoid_(x):
	return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
	sum = 0
	for i in x:
		sum += math.exp(-i)
	return np.exp(-x)/sum

sigmoid = np.vectorize(sigmoid_)
d_sigmoid = np.vectorize(d_sigmoid_)

def activation(input_values, input_weights):
	sum1 = np.transpose(input_weights).dot(input_values)
	# #print("sum1", sum1, "\n")
	t = (max(sum1)-min(sum1))
	if(t != 0):
		sum1 = (sum1-min(sum1))/t
	# #print("sum1 ",sum1.shape) #20*1
	return sigmoid(sum1), sum1

def train(featureMatrix, labels, learning_rate):

	# initialize w
	
	
	input_layer, hidden_layer1, hidden_layer2, output_layer = 10,20,20,10
	w1 = np.random.uniform(size=input_layer*hidden_layer1)
	w1 = np.split(w1,input_layer)
	w1 = np.array(w1)

	w2 = np.random.uniform(size=hidden_layer1*hidden_layer2)
	w2 = np.split(w2,hidden_layer1)
	w2 = np.array(w2)

	w3 = np.random.uniform(size=hidden_layer2*output_layer)
	# w2 = np.ones(hidden_layer*output_layer)
	w3 = np.split(w3,hidden_layer2)
	w3 = np.array(w3)

	#print("w1", w1, "\n")
	#print("w2", w2, "\n")

	for p in range(0, 10):
		for idx, x1 in enumerate(featureMatrix):
			print(idx)

			x2, sum1 = activation(x1, w1)

			x3, sum2 = activation(x2, w2)

			x4, sum3 = activation(x3, w3)

			x4 = softmax(sum3)
			#print("softmax", x3, "\n")
			y = np.argmax(x4)

			sig4 = sigmoid(sum3)
			#print("sigmoid(sum2)",sig2)
			if(y != labels[idx]):
				y_ = np.zeros(output_layer)
				y_[labels[idx]] = 1

				d_E3 = -np.divide(y_, sig4) + np.divide((1-y_),(1-sig4))
				#print("d_E2", d_E2, "\n")
				d_sig3 = d_sigmoid(sum3)
				#print("d_sig2", d_sig2, "\n")
				d_sum3 = x3

				t = np.multiply(d_E3, d_sig3)
				del_E3=[]
				for i in (d_sum3):
					del_E3.append(i*t)
				del_E3 = np.array(del_E3) 
				#print("del_E2 ",del_E2, "\n")

				# #print("w2 ",w2.shape)
				w3 = w3 - learning_rate*del_E3
				# print("w2 ",w2)

				d_E2 = w3.dot(np.multiply(d_E3, d_sig3))
				#print("d_E1 ",d_E1, "\n")
				d_sig2 = d_sigmoid(sum2)
				d_sum2 = x2
				#print("d_sum1 ",d_sum1, "\n")

				t=np.multiply(d_E2, d_sig2)
				del_E2=[]
				for i in (d_sum2):
					del_E2.append(i*t)
				del_E2 = np.array(del_E2)
				#print("del_E1", del_E1, "\n")
				
				# #print("w1 ",w1.shape)
				w2 = w2 - learning_rate*del_E2
				# print("w1 ",w1)

				d_E1 = w2.dot(np.multiply(d_E2, d_sig2))
				#print("d_E1 ",d_E1, "\n")
				d_sig1 = d_sigmoid(sum1)
				d_sum1 = x1
				#print("d_sum1 ",d_sum1, "\n")

				t=np.multiply(d_E1, d_sig1)
				del_E1=[]
				for i in (d_sum1):
					del_E1.append(i*t)
				del_E1 = np.array(del_E1)
				#print("del_E1", del_E1, "\n")
				
				# #print("w1 ",w1.shape)
				w1 = w1 - learning_rate*del_E1

	return w1, w2, w3


def test(featureMatrix, labels, w1, w2, w3):

	count = 0
	f = open('output.csv', 'w')

	print(w1,"\n", w2,"\n", w3)
	print("\n\n\n\n===========Testing===============\n\n\n\n")
	for idx, x1 in enumerate(featureMatrix):
		x2, sum1 = activation(x1, w1)
		
		x3, sum2 = activation(x2, w2)

		x4, sum3 = activation(x3, w3)
		x4 = softmax(sum3)

		# print("softmax ", x3, "\n")
		
		y = np.argmax(x4)

		# if(y == labels[idx]):
		# 	count = count + 1
		f.write(str(idx))
		f.write(",")
		f.write(np.array2string(y))
		f.write("\n")	

	# print(count/featureMatrix.shape(0)*100)


def main():
	learning_rate = 0.01
	file_name = 'train.csv'
	featureMatrix = getFeatureMatrix(file_name)

	labels = getLabels(file_name)
	w1,w2,w3 = train(featureMatrix, labels, learning_rate)

	# weights1 = open('w1', 'w')
	# pickle.dump(",".join(str(i) for i in j for j in w1),weights1)  
	# weights2 = open('w2','w')
	# pickle.dump(",".join(str(i) for i in j for j in w2),weights2)

	file_name = 'train.csv'
	featureMatrix = getFeatureMatrix(file_name)
	accuracy = test(featureMatrix, labels, w1,w2,w3)
	#print(accuracy)


if __name__ == "__main__":
	main()