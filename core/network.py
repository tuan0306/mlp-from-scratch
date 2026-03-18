import numpy as np

class Network:
    def __init__(self):
        self.loss=None
        self.loss_prime=None
        self.layers=[]
    def add(self,layer):
        self.layers.append(layer)
    def compile(self,loss,loss_prime):
        self.loss=loss
        self.loss_prime=loss_prime
    def predict(self,input):
        output=input.copy()
        for layer in self.layers:
            output=layer.forward(output)
        return output
    def fit(self,X_train,y_train,epochs,learning_rate,verbose=True):
        history=[]
        for epoch in range(epochs):
            output=self.predict(X_train)
            err=self.loss(y_train,output)
            history.append(err)
            error=self.loss_prime(y_train,output)
            for layer in reversed(self.layers):
                error=layer.backward(error,learning_rate)
            if verbose and (epoch+1)%10==0:
                print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {err:.6f}")
        return history