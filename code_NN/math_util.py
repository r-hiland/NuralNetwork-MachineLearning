# Place your EWU ID and name here

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        pass

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''

        # tanh'(x) = 1.0 - (tanh(x))^2.0

        
        pass

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''

        # You can compute logis via tanh as: logix(x) = (tanh(x/2.0) + 1.0) / 2.0
        
        pass

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        #logis'(x) = logis(x) * (1.0-logis(x)). 
        
        pass

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        pass

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        pass
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of the same shape of x, where every element is the max of: zero vs. the corresponding element in x.
        '''
        pass

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of the same shape of x, where every element is 1 if the correponding x element is positive; 0, otherwise. 
        '''
        pass

    