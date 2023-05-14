import numpy as np
from numpy.core._multiarray_umath import sign
import matplotlib.pyplot as plt


def aLawComp(signal, A=87.6):
    x = np.zeros_like(signal)
    idx = np.where(np.abs(signal) < 1 / A)
    idx1 = np.where(((1 / A) <= np.abs(signal)) & (np.abs(signal) <= 1))
    x[idx] = np.sign(signal[idx]) * ((A * np.abs(signal[idx])) / (1 + np.log(A)))
    x[idx1] = np.sign(signal[idx1]) * ((1 + np.log(A * np.abs(signal[idx1]))) / (1 + np.log(A)))
    return x


def aLawDecomp(signal, A=87.6):
    y = np.zeros_like(signal)

    idx1 = np.where(np.abs(signal) < 1 / 1 + np.log(A))
    idx2 = np.where((1 / (1 + np.log(A)) <= np.abs(signal)) & (np.abs(signal) <= 1))

    y[idx1] = np.sign(signal[idx1]) * np.abs(signal[idx1]) * (1 + np.log(A)) / A
    y[idx2] = np.sign(signal[idx2]) * np.exp(np.abs(signal[idx2]) * (1 + np.log(A)) - 1) / A
    return y


def muLawComp(signal, mu=255):
    x = np.zeros_like(signal)
    idx = np.where((-1 <= x) & (x <= 1))
    x[idx] = np.sign(signal[idx]) * (np.log(1 + mu * np.abs(signal[idx]))) / (np.log(1 + mu))
    return x


def muLawDecomp(signal, mu=255):
    y = np.zeros_like(signal)
    idx = np.where((-1 <= y) & (y <= 1))
    y[idx] = np.sign(signal[idx]) * (1 / mu) * ((1 + mu) ** np.abs(signal[idx]) - 1)

    return y


def changeSize(data, byteResolution=2):
    if byteResolution < 2 or byteResolution > 32:
        print("We don't support that")
        return

    d = 2 ** byteResolution - 1
    dataType = data.dtype
    if np.issubdtype(data.dtype, np.floating):
        m = -1
        n = 1
    elif np.issubdtype(data.dtype, np.integer):
        m = np.iinfo(data.dtype).min
        n = np.iinfo(data.dtype).max
    elif np.iinfo(data.dtype).min == 0 and np.iinfo(data.dtype).max > 1:
        m = 0
        n = np.iinfo(data.dtype).max

    data = data.astype(float)
    data = (data - m) / (n - m)

    data = data * d
    data = np.round(data)
    data = data / d

    data = (data * (n - m)) + m

    return data.astype(dataType)


def DPCM_compress(x, bit):
    y = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = changeSize(x[i] - e, bit)
        e += y[i]
    return y


x = np.linspace(-1, 1, 1000)


# y1 = np.linspace(-1,1,1000)
# y2 = np.linspace(-1,1,1000)
# y3 = np.linspace(-1,1,1000)
# y4 = np.linspace(-1,1,1000)
#
# y1 = aLawComp(y1)
# y1 = changeSize(y1,byteResolution=8)
# y2 = aLawComp(y2)
# y3 = muLawComp(y3)
# y3 = changeSize(y3, byteResolution=8)
# y4 = muLawComp(y4)

# y1 = aLawDecomp(y1)
# y2 = aLawDecomp(y2)
#
# y3 = muLawDecomp(y3)
# y4 = muLawDecomp(y4)
#
# plt.plot(x,y1, label = "sygnal po kompresji a-law po kwantyzacji do 8bitow" )
# plt.plot(x,y2, label = "sygnal po kompresji a-law bez kwantyzacji")
# plt.plot(x,y3, label = "sygnal po kompresji mu-law po kwantyzacji do 8bitow")
# plt.plot(x,y4, label = "sygnal po kompresji mu-law bez kwantyzacji")
#
# plt.xlabel("Wartość sygnału wejsciowego")
# plt.ylabel("Wartość sygnału wyjsciowego")
# plt.legend(loc = 'upper left')
# plt.show()

# x=np.linspace(-1,1,1000)
# y=0.9*np.sin(np.pi*x*4)
# y1 = aLawComp(changeSize(0.9*np.sin(np.pi*x*4),8))
# y2 = muLawComp(changeSize(0.9*np.sin(np.pi*x*4),8))
# y3 = DPCM_compress(0.9*np.sin(np.pi*x*4),16)
# fig, axs = plt.subplots(4,1)
# axs[0].plot(x,y)
# axs[1].plot(x,y1)
# axs[2].plot(x,y2)
# axs[3].plot(x,y3)
# plt.show()

def chromaResampling(L, subsampling):
    if subsampling == "4:2:2":
        L = np.repeat(L,2,axis=1)
    return L
def chromaSubsampling(L, subsampling):
    if subsampling == "4:2:2":
        L = L[::,::2]
    return L

L = np.zeros((2,4))
L[0][0] = 6
L[0][1] = (5)
L[0][2] =(1)
L[0][3] = (5)
L[1][0] =(9)
L[1][1] =(9)
L[1][2] = (5)
L[1][3] =(2)


L = chromaResampling(L,"4:2:2")
L = chromaSubsampling(L,"4:2:2")
print(L)
