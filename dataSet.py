import numpy as np
from vectorization import Vecto as vt
from wavIO import WavIO as WIO

class dataSet:
    X = None
    Y = None
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def sliceOf(self, forSetting):
        x = self.X[:,0:forSetting]
        y = self.Y[:,0:forSetting]
        return dataSet(x,y)
    def sliceDataSet(self, forSetting):
        self.X = self.X[:,0:forSetting]
        self.Y = self.Y[:,0:forSetting]

    @classmethod
    def createSimpleSet(cls,shape,nSample,count):
        X = np.random.rand(shape,nSample) #(파라미터 수, 데이터 수)
        Y = vt.vectorization(np.sin(X * np.pi),count)
        X = vt.vectorization(X,count)
        return dataSet(X,Y)

    @classmethod
    def loadDataSet(cls,count):
        X = WIO.load(count) #(파라미터 수, 데이터 수)
        Y = X
        return dataSet(X,Y)