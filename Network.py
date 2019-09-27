#Created by Tristan Bester
import numpy as np
import random

class Neural_Network(object):
    
    def __init__(self, sizes):
        #Initialise the network by creating the required matrices and vectors
        self.num_layers = len(sizes)
        self.sizes = sizes
        #Filling the weight matrices and bias vectors with random float values as per standard normal distribution
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(nxt,lst,) for nxt,lst in zip(sizes[1:],sizes[:-1])]

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def feedforward(self, input):
        #Forward propagation
        #All weighted inputs(z) and activations(a) are stored for use during backpropagation
        self.z = []
        self.a = []
        self.a.append(input)

        for w,b in zip(self.weights,self.biases):
            self.z.append(np.dot(w,self.a[-1]) + b)
            self.a.append(self.sigmoid(self.z[-1]))
        #Return the activation vector for the output layer
        return self.a[-1]
    
    def cost_function(self, X, y):
        #Calculate output activations of network
        output_activation = self.feedforward(X)
        #Return cost function value
        return 0.5*((y - output_activation)**2)
    
    
    def cost_function_prime(self, X, y):
        #Backpropagation
        #Create lists to store the deltas(errors) and partial derivatives for all of the 
        #weights and biases in the network.
        self.feedforward(X)
        self.deltas = []
        self.delta_weights = []
        self.delta_biases = []
        
        #Calculate delta and partial derivatives for the output layer
        self.deltas.append(((self.a[-1] - y))*self.sigmoid_prime(self.z[-1]))
        self.delta_biases.append(self.deltas[-1])
        self.delta_weights.append(self.deltas[-1] * self.a[-2])

        #Loop through all of the other layers in the network calculating the errors and partial derivatives
        for i in range(1, self.num_layers-1):
            mat_mul = np.dot(self.weights[-i].T, self.deltas[-1])
            sig_prime = self.sigmoid_prime(self.z[-i-1])
            hadamard_product = np.multiply(mat_mul,sig_prime)
            self.deltas.append(hadamard_product)
            self.delta_biases.append(self.deltas[-1])
            self.delta_weights.append(np.dot(self.a[-i-2],self.deltas[-1].T))
        
        #Put the partial derivatives in correct order
        self.delta_weights.reverse()
        self.delta_biases.reverse()
        
        #Return calculated partial derivatives
        return self.delta_weights, self.delta_biases
    
    
    def get_weight_vector(self):
        #Helper method used in the calculation of numerical gradients
        vector = self.weights[0]
        for w in self.weights[1:]:
            vector = np.concatenate((vector,w.ravel()), axis=None)
        return vector

    def set_weights(self, weight_vector):
        #Helper method used in the calculation of numerical gradients
        passed = 0
        for i in range(len(self.weights)):
            vector = np.array(weight_vector[passed: passed+self.weights[i].size])
            passed += self.weights[i].size
            self.weights[i] = vector.reshape(self.weights[i].shape)
        
    
    def train(self, training_data, eta, iterations):
        #This method allows the network to be trained with online training/incremental learning
        #where the weights and biases are updated after every single training example
        for i in range(iterations):
            example = random.choice(training_data)
            self.cost_function_prime(example[0],example[1])
            
            for i,(w,dw) in enumerate(zip(self.weights, self.delta_weights)):
                self.weights[i] = w-(eta*dw.T)
            
            for i,(b,db) in enumerate(zip(self.biases, self.delta_biases)):
                self.biases[i] = b-(eta*db)
    
    def train_on_mini_batch(self, mini_batch, eta):
        #Create a list of all of the derivatives of cost function for each example in mini_batch
        nalba_w = [np.zeros(w.shape) for w in self.weights]
        nalba_b = [np.zeros(b.shape) for b in self.biases]
        
        #Calculate the derivatives of cost function for each example in the mini-batch and add all of the gradients up
        #storing the results in the lists
        for x,y in mini_batch:
            delta_w, delta_b = self.cost_function_prime(x,y)
            
            nalba_w = [nw+(dnw.T) for nw, dnw in zip(nalba_w,delta_w)]
            nalba_b = [nb+dnb for nb, dnb in zip(nalba_b,delta_b)]
        
        n = len(mini_batch)

        #Update all weights in the network by learing rate * average gradient for each weight
        for i,(w,dw) in enumerate(zip(self.weights, nalba_w)):
            self.weights[i] = w-((eta/n)*dw)
        #Update all biases in the network by learing rate * average gradient for each bias
        for i,(b,db) in enumerate(zip(self.biases, nalba_b)):
                self.biases[i] = b-((eta/n)*db)

    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        #Stochastic Gradient Descent
        n = len(training_data)
        cost = []
        for i in range(epochs):
            #For each epoch split all of the training data into mini-batches of size batch_size
            random.shuffle(training_data)
            #Create mini-batches
            mini_batches = [training_data[x:x+batch_size] for x in range(0,n,batch_size)]
            #Train the network on all of the mini-batches individually
            for x in mini_batches:
                self.train_on_mini_batch(x,eta)
            
            #If test data is provided test the network on all of the test data and return a list containing the average cost 
            #over all of the test data after each mini-batch, this can be plotted to visualise progress. Testing after every
            #mini-batch will slow down training substantially and should only be done during tesing.
            if test_data:
                n_test = len(test_data)
                correct = 0
                cost_sum = 0
                for j in test_data:
                    output = self.feedforward(j[0])
                    cst = self.cost_function(j[0],j[1])
                    cost_sum += cst[0]
                    if output > 0.5:
                        output = 1
                    else:
                        output = 0
                    if output == j[1]:
                        correct+=1
                cost.append(cost_sum/n_test)
                print("Epoch {0}: {1}/{2}".format(i,correct,n_test))
        return cost   
        
             
def calculate_numerical_gradients(net,X,y):
    #Calculate numerical gradients in order to test if backpropagation was implemented correctly
    initial_weight_vector = net.get_weight_vector();
    epsilon = 1e-4
    perturb_vector = np.zeros(initial_weight_vector.shape)
    numerical_gradients_vector = []
    
    #Calculate the numerical gradient for each weight in the network.
    for i in range(len(initial_weight_vector)):
        perturb_vector[i] = epsilon
        
        temp_vector = initial_weight_vector - perturb_vector
        net.set_weights(temp_vector)
        costOne = net.cost_function(X,y)
        
        temp_vector = initial_weight_vector + perturb_vector
        net.set_weights(temp_vector)
        costTwo = net.cost_function(X,y)

        ls = [(costTwo - costOne)/(2*epsilon)]
        numerical_gradients_vector.append(ls)
        perturb_vector[i] = 0
        
    #Reset the weights in the network
    net.set_weights(initial_weight_vector)
    numerical_gradients_vector = np.array(numerical_gradients_vector)
    return numerical_gradients_vector.ravel()
    
        
#Note: All inputs must be column vectors
#Example training data set, list = [coloumn vector input, label]:
#test_data = [[[[1],[1]],0],[[[0],[0]],0],[[[0],[1]],1],[[[1],[0]],1]]    
