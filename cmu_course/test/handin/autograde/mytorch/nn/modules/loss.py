import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        # error  = np.square(A - Y)/( N * C)
        # L      = np.sum(error) / N


        ones_C = np.ones((C, 1))
        ones_N = np.ones((N, 1))
        SE = (A - Y) * (A - Y)
        SSE = np.dot(np.dot(np.transpose(ones_N) , SE), ones_C )
        L = SSE / (N * C)

        return L
    
    def backward(self):
    
        dLdA = self.A - self.Y
        
        return dLdA

class CrossEntropyLoss:

    def calc_softmax(self, A, ones):

        def div(x):
            return 1/x

        exp_v = np.vectorize(np.exp)
        div_v = np.vectorize(div)
        a = exp_v(A)
        b = div_v(np.dot(exp_v(A), ones))

        return np.multiply(a, b)


    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        Ones   = np.ones((C, 1), dtype="f")

        log_v = np.vectorize(np.log)

        self.softmax = self.calc_softmax(A, Ones) # TODO
        crossentropy = np.multiply(-1 * Y, log_v(self.softmax))
        L = np.sum(crossentropy) / N
        
        return L
    
    def backward(self):
    
        dLdA = self.softmax - self.Y
        
        return dLdA
