import numpy as np
import librosa
import librosa.display
import IPython.display

class WavIO:
    @staticmethod
    def load(count,path = "data/sample/test.wav"):
        y, sr = librosa.load(path)
        y = np.array(y)
        zeros = np.zeros((count - len(y) % count))
        y = np.append(y,zeros)
        y = np.reshape(y,(count,int(len(y) / count)))
        print(y.shape)
        return y