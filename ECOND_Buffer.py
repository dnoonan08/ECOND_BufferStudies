import numba
import numpy as np

class ECOND_Buffer:
    def __init__(self, nModules, nLinks, overflow=12*256):
        self.buffer = np.zeros(nModules,np.int16)    # np array to keep track of the words per bx
        self.nLinks = nLinks
        self.overflow = overflow

        self.maxSize = np.zeros(nModules,dtype=np.uint16)
        self.maxBX_First = np.zeros(nModules,dtype=np.uint32)
        self.maxBX_Last = np.zeros(nModules,dtype=np.uint32)
        self.overflowCount = np.zeros(nModules,dtype=np.uint32)
        #Create hist of buffer size for BX. Create N histograms of M bins, 
        #but locally they get stored as one MxN length array
        self.hist = np.array([0.]*nModules*overflow)
        #keep track of the starting index of each of the N histograms
        # when we want to fill a binX in one of the histograms, we will a
        #ctually fill bin X + histStarts
        self.histStarts = np.arange(nModules)*overflow

    def drain(self):
        #remove one word per link from the buffer
        self.buffer -= self.nLinks
        # if first entry in each module's buffer is empty (<0), set it to zero 
        self.buffer[self.buffer<0] = 0
            
    # add data to the buffer, counting overflows
    def write(self, data, i_BX):
        willOverflow = (self.buffer + data) > self.overflow
        self.overflowCount[willOverflow] += 1
        data[willOverflow] = 1
        self.buffer += data

        self.maxBX_First[(self.maxSize<self.buffer)] = i_BX
        self.maxSize = np.maximum((self.buffer),self.maxSize)
        self.maxBX_Last[(self.maxSize==self.buffer)] = i_BX            
    
    def fillHist(self):
        self.hist[self.buffer.astype(int)+self.histStarts] += 1

    
