# Place your EWU ID and Name here. 
# EWU ID: whiland - name: Reed Hiland

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



from re import L
from injector import T
import numpy as np
import math
import math_util as mu
import nn_layer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        if self.L == -1:
            self.layers.append(nn_layer.NeuralLayer(d, 'iden')) # adding iden() since first layer doesn't use an act
        else:
            self.layers.append(nn_layer.NeuralLayer(d, act))

        self.L += 1

        # pass
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''

        # Needs to be a matrix where every column is the wights, and every row is a node from the
        # previous layer.

        # Make matrix that size, then fill with random numbers limited be the sqrt above.

        # Need to skip the input layer, so start at 1

        for i in range(1, self.L+1):
            d_current = self.layers[i].d
            d_prev = self.layers[i-1].d  # number of non-bias nodes in previous layer
            limit = 1.0 / np.sqrt(d_current)
            # Weight matrix shape: (previous layer's nonbias nodes + bias, current layer's nonbias nodes)
            self.layers[i].W = np.random.uniform(-limit, limit, size=(d_prev + 1, d_current))
    
    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
        
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of labels, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        
        print("WIll check every 5000 iterations, and early stop if it hits below 3% (97% accuracy).")
        print("Should only take about 10000 iterations.")

        for t in range(iterations):
            # If using SGD, sample a minibatch; otherwise use the full dataset.
            if SGD:
                d_prime = np.random.choice(X.shape[0], mini_batch_size, replace=False)
                X_batch = X[d_prime]
                Y_batch = Y[d_prime]
            else:
                X_batch = X
                Y_batch = Y

             #add the bias feature column to the X matrix that has been randomly selected
            X_bias = np.ones((X_batch.shape[0], 1))
            X_tmp = np.hstack((X_bias, X_batch))
            
            # Adding the bias featue column for the first layer
            self.layers[0].X = X_tmp

            # Forward propagation: compute S and X for layers 1 to L.
            for l in range(1, self.L+1):
                current_layer = self.layers[l]
                prev_layer = self.layers[l-1]
                # S(l) = X(l-1) dot W(l)
                current_layer.S = np.dot(prev_layer.X, current_layer.W)
                # X(l) = [1, act(S(l))] where 1 is the bias node.
                current_layer.X = np.concatenate((np.ones((current_layer.act(current_layer.S).shape[0], 1)), current_layer.act(current_layer.S)), axis=1)

            # E = sum((X^(L) - Y)^2) / N'
            e = np.sum((self.layers[self.L].X[:,1:] - Y_batch)**2) / X_batch.shape[0]

            # Delta (L) 
            delta = 2 * (self.layers[self.L].X[:, 1:] - Y_batch) * self.layers[self.L].act_de(self.layers[self.L].S)

            # G(L) = X^(L-1) dot Delta(L)
            gradients = [None] * (self.L + 1)
            gradients[self.L] = np.einsum('ij,ik->jk', self.layers[self.L-1].X, delta) / X_batch.shape[0]


            for l in range(self.L-1, 0, -1):

                # Delta(l) = Delta(l+1) dot W(l+1) dot act'(S(l))
                delta = np.dot(delta, self.layers[l+1].W[1:, :].T) * self.layers[l].act_de(self.layers[l].S)

                # G(l) = X(l-1) dot Delta(l)
                gradients[l] = np.einsum('ij,ik->jk', self.layers[l-1].X, delta) / X_batch.shape[0]

            # Update weights
            for l in range(1, self.L+1):
                self.layers[l].W = self.layers[l].W - (eta * gradients[l])

            if t % 5000 == 0:
                err = self.error(X, Y)
                print("Iteration %d, error: %.2f%%" % (t, err))

                # Early stopping if error goes below 3%
                if err < 3:
                    print("Early stopping at iteration %d with error %.2f%%" % (t, err))
                    break

        # pass
    
    
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
            
            Note that the return of this function is NOT the sames as the return of the `NN_Predict` method
            in the lecture slides. In fact, every element in the vector returned by this function is the column
            index of the largest number of each row in the matrix returned by the `NN_Predict` method in the 
            lecture slides.
         '''

        X_bias = np.ones((X.shape[0], 1))
        X_tmp = np.hstack((X_bias,X))
        self.layers[0].X = X_tmp

        for l in range(1, self.L+1):
            current_layer = self.layers[l]
            prev_layer = self.layers[l-1]

            # S(l) = X(l-1) dot W(l)
            current_layer.S = np.dot(prev_layer.X, current_layer.W)
            # X(l) = [1 act(S(l))]
            current_layer.X = np.concatenate((np.ones((current_layer.act(current_layer.S).shape[0], 1)), current_layer.act(current_layer.S)), axis=1)

        return np.argmax(self.layers[-1].X[:, 1:], axis=1)
    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        
        estimated = self.predict(X)
        actual = np.argmax(Y, axis=1)
        return np.sum(estimated != actual) / X.shape[0] * 100