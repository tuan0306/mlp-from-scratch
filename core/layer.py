import numpy as np

class Layer:
    def __init__(self):
        self.input=None
        self.output=None
    def forward(self,input):
        raise NotImplementedError
    def backward(self,output_error,learning_rate):
        raise NotImplementedError
    
class Dense(Layer):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.weights=np.random.rand(input_size,output_size)
        self.bias=np.random.rand(1,output_size)
    def forward(self, input):
        self.input=input
        self.output=np.dot(self.input,self.weights)+self.bias
        return self.output
    def backward(self, output_error, learning_rate):
        weights_error=np.dot(self.input.T,output_error)
        bias_error=np.sum(output_error,axis=0,keepdims=True)
        input_error=np.dot(output_error,self.weights.T)
        self.weights-=learning_rate*weights_error
        self.bias-=learning_rate*bias_error
        return input_error
        