from vectorization import Vecto as vt
from neuralNetwork import NeuralNetwork
from dataSet import dataSet
from decision import Decision, command

import numpy as np
import torch
np.random.seed(1)
def gpuInit():
    cuda = torch.device('cuda')
    print("cuda available : ", torch.cuda.is_available())
    print("cuda current device : ", torch.cuda.current_device())
    print("cuda device count : ", torch.cuda.device_count())
    print("cuda device name : ", torch.cuda.get_device_name(0))
    return cuda
    
def main():
    cuda = gpuInit()
    np.random.seed(1)
    num_iterations = 1000
    count = 10  # default
    nSample = 0
    layerStart = 0
    layerLimit = 8

    while(True):
        userInput = input()
        cds = Decision.userInput(userInput)
        commands = cds.commands
        learning_rate = 0
        simpleNN = 0
        # print(cds.noErr)
        # print(commands)
        if cds.noErr:
            first = commands[0]
            if first == 0:  #createDataSet
                count = 10
                second = commands[1]
                third = commands[2]
                if second == 0: #train
                    trainSet = dataSet.createSimpleSet(1,third,count)
                    nSample = trainSet.X.shape[1]  #batch크기 또는 train set 수
                elif second == 1: #test
                    testSet = dataSet.createSimpleSet(1,third,count)
               
                print("  | create datsSet complete ")
                print("  | nSample = ", third)

            elif first == 1:  #autoBuild
                if len(commands) ==1:
                    layerStart = 0
                    layerLimit = 8
                elif len(commands) == 2:
                    layerStart = commands[1]
                    layerLimit = max(8,layerStart)
                elif len(commands) == 3:
                    layerStart = commands[1]
                    layerLimit = commands[2]
                if nSample > 0:
                    package = NeuralNetwork.autoBuilder(trainSet, nSample, layerStart = 0, layerLimit = 8, developmentMode=True)
                    # autoBuilder가 learning rate와 hiddenLayer의 수를 알아서 정해준다. 작업이 오래 걸릴 수 있음.
                    simpleNN = package[0]
                    learning_rate = package[1]
                else:
                    print("  error : please load trainSet first")
                    print("  or create dataSet | use command   createDataSet")

            elif first == 2:  #model
                if len(commands) > 1:
                    second = commands[1]
                    if second == 0: #train
                        if (learning_rate != 0) and (simpleNN != 0):
                            simpleNN.training_with_regularization(trainSet, num_iterations, learning_rate)
                            train_acc = simpleNN.getAccuracy(trainSet)
                            print ('train set Accuracy: %f' % train_acc + '%')
                        else:
                            print("  error : please build NN first")
                            print("  | use command   autoBuild")
                    elif second == 1: #test
                        if (simpleNN != 0):
                            try:
                                train_acc = simpleNN.getAccuracy(trainSet)
                                test_acc = simpleNN.getAccuracy(testSet)
                                print ('train set Accuracy: %f' % train_acc + '%')
                                print ('test set Accuracy: %f' % test_acc + '%')
                            except:
                                print("  error : please load test set")
                        else:
                            print("  error : please build NN first")
                            print("  | use command   autoBuild")

            elif first == 3:  #loadDataSet
                second = commands[1]
                third = commands[2]
                count = 100
                if second == 0: #train
                    trainSet = dataSet.loadDataSet(count)
                    nSample = trainSet.X.shape[1]  #batch크기 또는 train set 수
                elif second == 1: #test
                    testSet = dataSet.loadDataSet(count)
               
                print("  | load datsSet complete ")
                print("  | nSample = ", nSample)
    ### dataset loading 하기.

    # plt.title("Data distribution")
    # plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0,:], s=20, cmap=plt.cm.RdBu)
    # plt.show()

    # simpleNN = NeuralNetwork.Builder(nSample, count, numberOfHiddenLayers = 1)

if __name__=='__main__':    
    main()
