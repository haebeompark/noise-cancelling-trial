import numpy as np
import librosa
import librosa.display
import IPython.display

from vectorization import Vecto as vt

class WavIO:
    @staticmethod
    def load(length,path = "data/sample/test.wav",count = 10):
        '''
        length : 소리 데이터를 자르는 길이단위; sampling된 소리데이터는 1차원 배열로 저장이 된다.
        vectorization을 통해 데이터를 1과 0으로 재 정렬함.
        '''
        x, sr = librosa.load(path) #샘플링하여 numpy로 로드한다.
        x = np.array(x)
        zeros = np.zeros((length - len(x) % length))
        x = np.append(x,zeros)
        x = np.reshape(x,(length,int(len(x) / length)))
        return x