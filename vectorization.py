import numpy as np
import torch

class Vecto:
    @classmethod
    def vectorization(cls, X,count = 10):
        '''
        x = 숫자 또는 numpy배열
        x는 1보다 작고 0보다 크다고 가정하자
        '''
        parameters = X.shape[0]
        result = np.array([0])
        #vec = np.zeros((count,X.shape[1]))
        c=0
        for splited in X:
            splited = np.array([splited])
            arr = np.zeros(splited.shape)
            for i in range(count):
                item = np.rint(splited).astype(bool) # 1또는 0
                #print(item)
                splited = splited - item /2
                splited = splited * 2
                arr = np.append(arr, item, axis = 0)
            arr = np.delete(arr, 0, axis = 0)
            if result.shape[0] == 1:
                result = arr
            else:
                result = np.append(result, arr, axis = 0)
        #result = result.reshape(parameters,count,X.shape[1])

        return result

    @classmethod
    def unvectorization(cls, arr, count = 10):
        parameters = int(arr.shape[0] / count)
        first = True
        X = None
        for j in range(parameters):
            splited = arr[j*count:(j+1)*count,:]
            temp = np.zeros((1,arr.shape[1]))
            for i in range(0, count):
                item = splited[count-i-1]
                temp = (temp + item) / 2
            if first:
                first = False
                X = temp
            else:
                X = np.append(X, temp, axis = 0)
        return X