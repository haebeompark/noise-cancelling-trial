from vectorization import Vecto as vt
from neuralNetwork import NeuralNetwork
import numpy as np
np.random.seed(1)

def main():
    np.random.seed(1)
    num_iterations=300
    learning_rate = 0.003
    count = 10
    
    ### dataset loading 하기.
    X_train = np.random.rand(1,500)
    Y_train = vt.vectorization(np.sin(X_train * np.pi),count)
    X_train = vt.vectorization(X_train,count)
    X_test = np.random.rand(1,200)
    Y_test = vt.vectorization(np.sin(X_test * np.pi),count)
    X_test = vt.vectorization(X_test,count)
    # plt.title("Data distribution")
    # plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0,:], s=20, cmap=plt.cm.RdBu)
    # plt.show()
    
    ## 코딩시작
    layerDims = [count,count,count,count,count,count]
    nSample = X_train.shape[1]
    ## 코딩시작

    simpleNN = NeuralNetwork(layerDims, nSample)

    ### training
    simpleNN.training_with_regularization(X_train,Y_train, num_iterations, learning_rate)

    ### prediction
    train_acc = simpleNN.getAccuracy(X_train, Y_train)
    test_acc = simpleNN.getAccuracy(X_test, Y_test)
    print ('train set Accuracy: %d' % train_acc + '%')
    print ('train set Accuracy: %d' % test_acc + '%')


    # decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train)
    # decision_boundary(lambda x: simpleNN.predict(x.T), X_test, Y_test)

if __name__=='__main__':
    main()
