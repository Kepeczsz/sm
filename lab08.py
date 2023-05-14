import scipy.fftpack
import numpy as np
import cv2


class DataContainer:
    def __init__(self, Y, Cb, Cr, OGShape, Ratio="4:4:4", QY=np.ones((8, 8)), QC=np.ones((8, 8))):
        self.shape = OGShape
        self.Y = Y
        self.Cb = Cb
        self.Cr = Cr
        self.ChromaRatio = Ratio
        self.QY = QY
        self.QC = QC


def CompressBlock(block, Q):
    outBlock = scipy.fftpack.dct(block.astype(float), axis=0, norm = 'ortho')
    outBlock = scipy.fftpack.dct(outBlock, axis=1, norm = 'ortho')
    return zigzag(np.round(outBlock/Q))


def DecompressBlock(vector, Q):
    outvector = zigzag(vector)
    outvector = scipy.fftpack.icdt(outvector,axis = 0, norm = 'ortho')
    outvector = scipy.fftpack.icdt(outvector,axis = 1, norm = 'ortho')
    decompressedBlock = outvector * Q
    return np.clip(np.round(decompressedBlock), 0, 255)


def ConversionToYCrCb(img):
    img -= 128
    imageToYCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(int)
    return imageToYCrCb


def ConversionToRGB(img):
    imageToRgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    img += 128
    return imageToRgb


def CompressLayer(L, Q):
    S = np.array([])
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w:(w + 8), k:(k + 8)]
            S = np.append(S, CompressBlock(block, Q))
    return S


def DecompressLayer(S, Q):
    L = np.array([8])
    for idx, i in enumerate(range(0, S.shape[0], 64)):
        vector = S[i:(i + 64)]
        m = L.shape[0] / 8
        w = int((idx % m) * 8)
        k = int((idx // m) * 8)
        L[w:(w + 8), k:(k + 8)] = DecompressBlock(vector, Q)


src = cv2.imread('wycinek1.jpeg')

def compress(img, subsampling):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(int)
    shape = img.shape
    
def zigzag(A):
    template = n = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ])
    if len(A.shape) == 1:
        B = np.zeros((8, 8))
        for r in range(0, 8):
            for c in range(0, 8):
                B[r, c] = A[template[r, c]]
    else:
        B = np.zeros((64,))
        for r in range(0, 8):
            for c in range(0, 8):
                B[template[r, c]] = A[r, c]
    return B
