import numpy as np
from core.layer import Layer

class Activation(Layer):
    def __init__(self,activation_function,activation_derivative):
        super().__init__()
        self.activation_function=activation_function
        self.activation_prime=activation_derivative
    def forward(self, input):
        self.input=input
        self.output=self.activation_function(self.input)
        return self.output
    def backward(self, output_error, learning_rate):
        input_error=output_error*self.activation_prime(self.input)
        return input_error
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        def sigmoid_prime(x):
            s=sigmoid(x)
            return s*(1-s)
        super().__init__(sigmoid,sigmoid_prime)
            
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(x,0)
        def relu_prime(x):
            return np.where(x>0,1,0)
        super().__init__(relu,relu_prime)
        