import numpy as np

class Vecto:
    @classmethod
    def vectorization(cls, X,count = 10):
        '''
        x = 숫자 또는 numpy배열
        x는 1보다 작고 0보다 크다고 가정하자
        '''
        #vec = np.zeros((count,X.shape[1]))
        arr = np.zeros(X.shape)
        for i in range(count):
            item = np.rint(X) # 1또는 0
            #print(item)
            X = X - item /2
            X = X * 2
            arr = np.append(arr, item, axis = 0)
        arr = np.delete(arr, 0, axis = 0)
        return arr

    @classmethod
    def unvectorization(cls, arr, count = 10):
        X = np.zeros((1,arr.shape[1]))
        for i in range(0, count):
            item = arr[count-i-1]
            X = (X + item) / 2
        return X