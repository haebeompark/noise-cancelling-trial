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