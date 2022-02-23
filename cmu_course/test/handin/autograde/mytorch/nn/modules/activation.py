import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, Z):


        sigmoid_v = np.vectorize(self.sigmoid)

        self.A = sigmoid_v(Z) # TODO

        return self.A

        # return NotImplemented
    
    def backward(self):
    
        dAdZ = self.A - np.square(self.A)
        
        return dAdZ


class Tanh:

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, Z):

        tan_h_v = np.vectorize(self.tanh)
        self.A = tan_h_v(Z)
        
        return self.A
    
    def backward(self):
    
        dAdZ = 1 - np.square(self.A)
        
        return dAdZ


class ReLU:

    def relu(self, x):
        return max(x, 0)

    def relu_back(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def forward(self, Z):

        relu_v = np.vectorize(self.relu)
        self.A = relu_v(Z)
        
        return self.A
    
    def backward(self):

        relu_back_v = np.vectorize(self.relu_back)
        dAdZ = relu_back_v(self.A)
        return dAdZ
        
        
