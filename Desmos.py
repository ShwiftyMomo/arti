import numpy as np
import math

def sigmoid(x):
	return math.exp(x)/(1 + math.exp(x))

def prime(x):
	return sigmoid(x)*(1-sigmoid(x))

Input = np.asarray([ 
	[.1,.2,.3,.4,.5,.6,.7,.8,.9,1], 
	[.4,.5,.7,.1,.2,.4,.9,1,.4,.2], 
	[.9,.5,.3,.4,.6,.8,.1,1,.2,.7] ])

		#Weights for layer 1
W1 = np.asarray( [.5,-.5,.5,.5,.5,.5])
W2 = np.asarray([.5,-.5])

Gradient = [0,0,0,0,0,0,0,0]

for n in range(0,10000):
	cost = 0
		#My inputs
	for i in range(0,9):

			#Hidden Layer
		HL = np.asarray([
			sigmoid((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))
			,sigmoid((Input[0,i]*W1[3])+(Input[1,i]*W1[4])+(Input[2,i]*W1[5]))])

			#Output
		OU = sigmoid(HL[0]*W2[0]+HL[1]*W2[1])

			#Desired Output
		DO = ([.1,.2,.3,.4,.5,.6,.7,.8,.9,1])

		Gradient[0] = Gradient[0] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*W2[1]*prime((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))*Input[0,i]
		Gradient[1] = Gradient[1] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*W2[1]*prime((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))*Input[1,i]
		Gradient[2] = Gradient[2] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*W2[1]*prime((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))*Input[2,i]
		Gradient[3] = Gradient[3] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*W2[1]*prime((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))*Input[0,i]
		Gradient[4] = Gradient[4] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*W2[1]*prime((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))*Input[1,i]
		Gradient[4] = Gradient[5] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*W2[1]*prime((Input[0,i]*W1[0])+(Input[1,i]*W1[1])+(Input[2,i]*W1[2]))*Input[2,i]
		Gradient[6] = Gradient[6] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*HL[0]
		Gradient[7] = Gradient[7] + 2*(OU-DO[i])*prime(HL[0]*W2[0]+HL[1]*W2[1])*HL[1]
		Cost = Cost + (DO[i]-OU)*(DO[i]-OU)

	Gradient[0] = Gradient[0]/10
	Gradient[1] = Gradient[1]/10
	Gradient[2] = Gradient[2]/10
	Gradient[3] = Gradient[3]/10
	Gradient[4] = Gradient[4]/10
	Gradient[5] = Gradient[5]/10
	Gradient[6] = Gradient[6]/10
	Gradient[7] = Gradient[7]/10

	Cost = Cost/10

	for t in range (0,5):
		W1[t] = W1[t]-Gradient[t]

	W2[0] = W2[0] - Gradient[6]
	W2[1] = W2[1] - Gradient[7]

	if (n % 500) == 0:
		print("Cost:" + str(Cost))
