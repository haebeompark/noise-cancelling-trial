from vectorization import Vecto as vt
from neuralNetwork import NeuralNetwork
from dataSet import dataSet
from decision import Decision, command
import numpy as np
np.random.seed(1)
def createSimpleSet(shape,nSample,count):
    X = np.random.rand(shape,nSample) #(파라미터 수, 데이터 수)
    Y = vt.vectorization(np.sin(X * np.pi),count)
    X = vt.vectorization(X,count)
    return dataSet(X,Y)

def main():
    np.random.seed(1)
    num_iterations = 1000
    count = 10

    while(True):
        userInput = input()
        cds = Decision.userInput(userInput)
        commands = cds.commands
        if cds.noErr:
            first = commands[0]
            second = commands[1]
            third = commands[2]
            if first == 0:  #createDataSet
                if second == 0: #train
                    trainSet = createSimpleSet(1,third,count)
                elif second == 1: #test
                    testSet = createSimpleSet(1,third,count)

    ### dataset loading 하기.

    # plt.title("Data distribution")
    # plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0,:], s=20, cmap=plt.cm.RdBu)
    # plt.show()

    nSample = trainSet.X.shape[1]  #batch크기 또는 train set 수

    # simpleNN = NeuralNetwork.Builder(nSample, count, numberOfHiddenLayers = 1)
    
    package = NeuralNetwork.autoBuilder(trainSet, nSample, layerStart = 0, layerLimit = 8, developmentMode=False)
    # autoBuilder가 learning rate와 hiddenLayer의 수를 알아서 정해준다. 작업이 오래 걸릴 수 있음.
    simpleNN = package[0]
    learning_rate = package[1]

    ### training
    simpleNN.training_with_regularization(trainSet, num_iterations, learning_rate)

    ### prediction
    train_acc = simpleNN.getAccuracy(trainSet)
    test_acc = simpleNN.getAccuracy(testSet)
    print ('train set Accuracy: %f' % train_acc + '%')
    print ('test set Accuracy: %f' % test_acc + '%')

if __name__=='__main__':
    main()
