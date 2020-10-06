import numpy as np
from vectorization import Vecto as vt
np.random.seed(1)

def sigmoid(signal):
    """
    sigmoid 함수

    Arguments:
        x:  scalar 또는 numpy array

    Return:
        s:  sigmoid(x)
    """

    signal = np.clip( signal, -500, 500 )

    # Calculate activation signal
    signal = 1.0/( 1 + np.exp( -signal ))

    return signal

def relu(x):
    """
    ReLU 함수

    Arguments:
        x : scalar 또는 numpy array

    Return:
        s : relu(x)
    """
    s = np.maximum(0,x)

    return s

class NeuralNetwork:
    def __init__(self,layerDims, nSample):
        '''
        학습할 네트워크.

        Arguments:
            layerDims [array]: layerDims[i] 는 레이어 i의 hidden Unit의 개수 (layer0 = input layer)
            nSample: 데이터셋의 샘플 수
        '''

        self.nSample = nSample
        self.nlayer = len(layerDims)-1

        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.vel = {}
        self.s = {}
        self.cache = {}
        self.initialize_optimizer(layerDims)

    def weightInit(self, layerDims):

        np.random.seed(1)
        parameters = {}

        for l in range(1, len(layerDims)):
            parameters['W' + str(l)] = np.random.randn(layerDims[l],layerDims[l-1])*np.sqrt(2/layerDims[l-1])
            parameters['b' + str(l)] = np.zeros((layerDims[l],1))

        return parameters

    # iniitialize parameter for optimizer
    def initialize_optimizer(self,layerDims):
        s = {}
        vel = {}
        for l in range(1,self.nlayer+1):
            s['sdW' + str(l)] = np.zeros((layerDims[l],layerDims[l-1]))
            s['sdb' + str(l)] = np.zeros((layerDims[l],1))
            vel['VdW' + str(l)] = np.zeros((layerDims[l],layerDims[l-1]))
            vel['Vdb' + str(l)] = np.zeros((layerDims[l],1))
            
            self.s = s
            self.vel = vel
        return

    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A23: network output
        '''

        ## 코딩시작 
        parameters = self.parameters
        cache = {}
        cache['A0'] = X
        cache['Y'] = 0
        for l in range(1,1+int(len(parameters)/2)):
            cache['Z'+str(l)]= np.dot((parameters['W' + str(l)]), cache['A'+str(l-1)]) + parameters['b'+str(l)]
            if(l != len(parameters)/2):
                cache['A'+str(l)]= relu(cache['Z' + str(l)])
            if(l == len(parameters)/2):
                AL = cache['A'+str(l)]= sigmoid(cache['Z' + str(l)])
        self.cache.update(cache)

        return AL

    def compute_cost(self, AL,Y, lambd=0.7):
        
        self.cache.update(Y=Y)
        W1, W2, W3, W4, W5 = self.parameters["W1"], self.parameters["W2"], self.parameters["W3"], self.parameters["W4"], self.parameters["W5"]

        def replace_zero(prob):
            result = np.where(prob > 0.000000002, prob, -20)
            np.log(result, out=result, where=result > 0)
            return result
        
        logprobs = -Y * replace_zero(AL) -(1-Y) * replace_zero(1-AL)
        cost = (1/ self.nSample) * np.sum(logprobs) + (lambd/self.nSample)*(np.sum(W1)+np.sum(W2)+np.sum(W3)+np.sum(W4)+np.sum(W5))
        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost

    def update_params(self, learning_rate=0.01, beta2=0.999, epsilon=1e-8):
        '''
        backpropagation을 통해 얻은 gradients를 update한다.

        Arguments:
            learning_rate:  학습할 learning rate

        Return:
        '''
        parameters = self.parameters
        grads = self.grads
        dims = int(len(parameters)/2)
        s = {}
        for l in range(1,dims+1):
            s["sdW" + str(l)] = beta2 *self.s["sdW" + str(l)] + (1-beta2)*(self.grads["dW" + str(l)]**2)
            s["sdb" + str(l)] = beta2 *self.s["sdb" + str(l)] + (1-beta2)*(self.grads["db" + str(l)]**2)
        self.s.update(s)
        for l in range(1,dims+1):
            parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * np.divide(grads["dW"+str(l)] , (np.sqrt(s["sdW"+str(l)]) + epsilon))
            parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * np.divide(grads["db"+str(l)] , (np.sqrt(s["sdb"+str(l)]) + epsilon))

        return 
    
    def compute_cost_with_regularization(self, AL, Y, lambd=0.7):
        '''
        cross-entropy loss에 regularization term을 이용하여 cost를 구한다.

        Arguments:
            A3 : network 결과값
            Y  : 정답 label(groud truth)
            lambd : 람다 값. 
        Return:
            cost
        '''

        self.cache.update(Y=Y)
        W = []
        sum = 0.0
        for l in range(1,self.nlayer+1):
            sum += self.parameters["W"+str(l)].sum()
            W.append(self.parameters["W"+str(l)])

        def replace_zero(prob):
            result = np.where(prob > 0.000000002, prob, -20)
            np.log(result, out=result, where=result > 0)
            return result
        
        logprobs = -Y * replace_zero(AL) -(1-Y) * replace_zero(1-AL)
        cost = (1/ self.nSample) * np.sum(logprobs) + (lambd/self.nSample)*(sum)

        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost

    def backward_with_regularization(self,lambd=0.7):
        '''
        regularization term이 추가된 backward propagation.

        Arguments:
            lambd: 

        Return:
        '''
        # regularization term이 추가된 cost에서 back-propagation을 진행, grads update
        parameters = self.parameters
        cache = self.cache
        Y = cache["Y"]
        grads = {}
        dims = int(len(parameters)/2) #(=5)
        A = cache["A"+str(dims)] #A5
        dZ = A-Y #dZ3 = A3 - Y
        while(dims>0): #dims = 5,4,3,2,1
            if(dims > 0):
                W = parameters["W"+str(dims)] #W5, W4, W3, W2, W1
                A = cache["A"+str(dims-1)] #A4, A3, A2, A1, A0
                dW = (1/self.nSample) * np.dot(dZ,A.T) #dW5 = (1/m)*(dZ5 dot A3.T), dW4, dW3
                db = (1/self.nSample) * np.sum(dZ,axis=1,keepdims=True) #db5 = (1/m)*sum(dZ5), db4, db2
                grads["dW"+str(dims)] = dW
                grads["db"+str(dims)] = db
                dims = dims-1
                if(dims >0):
                    Z = cache["Z"+str(dims)]#Z2, Z1
                    dZ = np.dot(W.T,dZ) * (Z>0) #dZ2 = W3.T dot dZ3 * g`(Z2),       dZ1 = ...
                    grads["dZ"+str(dims)] = dZ 
                
        self.grads.update(grads)
        for l in range(1,1+int(len(parameters)/2)):
            grads["dW"+str(l)] = self.grads["dW"+str(l)] + (lambd/self.nSample)*(np.sum(self.parameters["W"+str(l)]))
        self.grads.update(grads)
        ## 코딩
        return
    
    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''
        A = self.forward(X)
        predictions = np.rint(A)

        return predictions
  
    def training_with_regularization(self, X_train, Y_train, num_iterations, learning_rate):

        ## 코딩시작
        for i in range(0, num_iterations):
            AL = self.forward(X_train)               ##forward propagation 하기.
            cost = self.compute_cost_with_regularization(AL,Y_train)             ## cost function을 이용해 cost 구하기.
            pass                ## 위에 만든 함수를 이용하여 backpropagation 진행
            self.backward_with_regularization()
            pass                ## 위에 만든 함수를 이용하여 network의 weight update하기.
            self.update_params(learning_rate)
            
            if i % 100 ==0:
                learning_rate = learning_rate * 0.9
                print("Cost after iteration %i: %f" %(i,cost))

    def getAccuracy(self, X, Y):
        ### prediction
        predictions = self.predict(X)
        predictions_unvec = vt.unvectorization(predictions)
        Y_unvec = vt.unvectorization(Y)

        return (100 - float(np.abs(Y_unvec - predictions_unvec).sum()/(Y_unvec.size))*100)