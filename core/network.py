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
    def fit(self,X_train,y_train,epochs,learning_rate,verbose=True,X_val=None,y_val=None):
        train_history=[]
        val_history=[]
        for epoch in range(epochs):
            output=self.predict(X_train)
            train_err=self.loss(y_train,output)
            train_history.append(train_err)
            
            error=self.loss_prime(y_train,output)
            for layer in reversed(self.layers):
                error=layer.backward(error,learning_rate)
                
            if X_val is not None and y_val is not None:
                val_output=self.predict(X_val)
                val_err=self.loss(y_val,val_output)
                val_history.append(val_err) 
                
            if verbose and (epoch+1)%100==0:
                msg=f"Epoch {epoch+1:4d}/{epochs} | Train_Loss: {train_err:.6f}"
                if X_val is not None and y_val is not None:
                    msg+=f' | Val_Loss: {val_err:.6f}'
                print(msg)
                    
        return train_history,val_history