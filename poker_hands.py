import math
import pickle
import numpy as np


def getFeatureMatrix(file_path):
	data = np.genfromtxt(file_path, delimiter=',', skip_header=1,usecols=range(0,10), dtype='int', max_rows=10000)
	data= data
	np.c_[data,	np.ones(data.shape[0])]
	return data 

def getLabels(file_path):
	data = np.genfromtxt(file_path, delimiter=',', skip_header=1,usecols=range(10,11), dtype='int', max_rows=10000)
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
	x2 = sigmoid(sum1)

def train(featureMatrix, labels, learning_rate):

	# initialize w
	
	
	input_layer, hidden_layer, output_layer = 10,40,10
	w1 = np.random.uniform(size=input_layer*hidden_layer)

	#####temp assignment TBD###########
	# w1 = np.ones(input_layer*hidden_layer)
	w1 = np.split(w1,input_layer)
	w1 = np.array(w1)
	w2 = np.random.uniform(size=hidden_layer*output_layer)
	# w2 = np.ones(hidden_layer*output_layer)
	w2 = np.split(w2,hidden_layer)
	w2 = np.array(w2)

	#print("w1", w1, "\n")
	#print("w2", w2, "\n")

	for p in range(0, 10):
		for idx, x1 in enumerate(featureMatrix):
			print(idx)
			#print('\n')
			# #print(w1 ",w1.shape) #10*20
			#print("x1 ",x1) #10*1

			sum1 = np.transpose(w1).dot(x1)
			# #print("sum1", sum1, "\n")
			t = (max(sum1)-min(sum1))
			if(t != 0):
				sum1 = (sum1-min(sum1))/t
			# #print("sum1 ",sum1.shape) #20*1
			x2 = sigmoid(sum1)
			#print("sigmoid(sum1) ",x2, "\n") #20*1

			sum2 = np.transpose(w2).dot(np.transpose(x2))
			#print("sum2", sum2, "\n")
			t = (max(sum2)-min(sum2))
			if(t != 0):
				sum2=(sum2-min(sum2))/t
			x3 = softmax(sum2)
			#print("softmax", x3, "\n")
			y = np.argmax(x3)

			sig2 = sigmoid(sum2)
			#print("sigmoid(sum2)",sig2)
			if(y != labels[idx]):
				y_ = np.zeros(output_layer)
				y_[labels[idx]] = 1
				#print("y_", y_)

				d_E2 = -np.divide(y_, sig2+0.1) + np.divide((1-y_),(1-sig2+0.1))
				#print("d_E2", d_E2, "\n")
				d_sig2 = d_sigmoid(sum2)
				#print("d_sig2", d_sig2, "\n")
				d_sum2 = x2

				t = np.multiply(d_E2, d_sig2)
				del_E2=[]
				for idx,i in enumerate(d_sum2):
					del_E2.append(i*t)
				del_E2 = np.array(del_E2) #20*10
				#print("del_E2 ",del_E2, "\n")

				# #print("w2 ",w2.shape)
				w2 = w2 - learning_rate*del_E2
				# print("w2 ",w2)

				d_E1 = w2.dot(np.multiply(d_E2, d_sig2)) #20*1
				#print("d_E1 ",d_E1, "\n")
				d_sig1 = d_sigmoid(sum1) #20*1
				d_sum1 = x1 #10*1
				#print("d_sum1 ",d_sum1, "\n")


				t=np.multiply(np.transpose(d_E1), d_sig1)
				# #print("t ",t.shape) #10*20

				del_E1=[]
				for idx,i in enumerate(d_sum1):
					del_E1.append(i*t)
				del_E1 = np.array(del_E1) #10*20
				#print("del_E1", del_E1, "\n")
				
				# #print("w1 ",w1.shape)
				w1 = w1 - learning_rate*del_E1
				# print("w1 ",w1)

	return w1, w2


def test(featureMatrix, labels, w1, w2):

	count = 0
	f = open('output.csv', 'w')
	print("\n\n\n\n===========Testing===============\n\n\n\n")
	for idx, x1 in enumerate(featureMatrix):
		sum1 = np.transpose(w1).dot(np.transpose(x1))
		sum1=(sum1-min(sum1))/(max(sum1)-min(sum1))
		x2 = sigmoid(sum1)
		sum2 = np.transpose(w2).dot(np.transpose(x2))
		sum2=(sum2-min(sum2))/(max(sum2)-min(sum2))
		x3 = softmax(sum2)
		print("softmax ", x3, "\n")
		y = (np.argmax(x3))

		# if(y == labels[idx]):
		# 	count = count + 1
		f.write(str(idx))
		f.write(",")
		f.write(np.array2string(y))
		f.write("\n")	

	# print(count/featureMatrix.shape(0)*100)


def main():
	learning_rate = 0.001
	file_name = 'train.csv'
	featureMatrix = getFeatureMatrix(file_name)

	labels = getLabels(file_name)
	w1,w2 = train(featureMatrix, labels, learning_rate)

	# weights1 = open('w1', 'w')
	# pickle.dump(",".join(str(i) for i in j for j in w1),weights1)  
	# weights2 = open('w2','w')
	# pickle.dump(",".join(str(i) for i in j for j in w2),weights2)

	file_name = 'train.csv'
	featureMatrix = getFeatureMatrix(file_name)
	accuracy = test(featureMatrix, labels, w1,w2)
	#print(accuracy)


if __name__ == "__main__":
	main()