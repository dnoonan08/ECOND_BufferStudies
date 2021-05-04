import numba
import numpy as np


@numba.jit(nopython=True)
def moveBuffer(buffContent, buffStarts, buffLen):
    for i in range(len(buffStarts)):
        x = buffStarts[i]
        buffContent[x:x+buffLen-2] = buffContent[x+1:x+buffLen-1]

@numba.jit(nopython=True)
def _size(buffContent, buffStarts, buffStops):
    buffSize = []
    for i in range(len(buffStarts)):
        buffSize.append(buffContent[buffStarts[i]:buffStops[i]].sum())
    return np.array(buffSize)
        
@numba.jit(nopython=True)
def _get(buffContent, buffStarts, buffStops):
    buffer = []
    for i in range(len(buffStarts)):
        buffer.append(buffContent[buffStarts[i]:buffStops[i]])
    return buffer

class ECOND_Buffer:
    def __init__(self, nModules, buffSize, nLinks, overflow=12*128):
        self.buffer = np.zeros(nModules*buffSize,np.int16)    # np array to keep track of the words per bx
        self.starts=np.arange(nModules)*buffSize              # starting point of buffer for each of the modules
        self.write_pointer=np.arange(nModules)*buffSize       # track a write pointer, ie the number of events curently in the buffer

        self.buffSize = buffSize
        self.nLinks = nLinks
        self.overflow = overflow

        self.maxSize = np.zeros(nModules,dtype=np.uint16)
        self.maxBX_First = np.zeros(nModules,dtype=np.uint32)
        self.maxBX_Last = np.zeros(nModules,dtype=np.uint32)
        self.overflowCount = np.zeros(nModules,dtype=np.uint32)


    def drain(self):
        #remove one word per link from the buffer

        bufferNotEmpty = self.write_pointer>self.starts
        self.buffer[self.starts[bufferNotEmpty]] -= self.nLinks

        # if first entry in each module's buffer is empty (<0), we need to read out words from the next event, and shift buffer
        isEmpty = self.buffer[ self.starts ]< 0
        while isEmpty.sum()>0:
            lenBuffer = self.write_pointer-self.starts  #get the current number of events in each module's buffer

            pushForward = self.starts[(isEmpty) & (lenBuffer>1)] #find modules which have data, but first entry is negative

            self.buffer[ pushForward + 1 ] += self.buffer[ pushForward ] #move the negative to the next entry (ie, read out some words from next event)

            #shift the array to remove the first entry, and adjust the write pointer
            moveBuffer(self.buffer, self.starts[isEmpty],self.buffSize) 
            self.write_pointer[isEmpty] -= 1 
            
            #update list of empty first entries
            isEmpty = self.buffer[ self.starts ]< 0

    def size(self):
        return _size(self.buffer.copy(), self.starts.copy(), self.write_pointer.copy())

    def get(self):
        return _get(self.buffer.copy(), self.starts.copy(), self.write_pointer.copy())

    # add data to the buffer, counting overflows
    def write(self, data, i_BX):
        willOverflow = (self.size() + data) > self.overflow
        self.overflowCount[willOverflow] += 1
        data[willOverflow] = 1

        self.buffer[self.write_pointer] += data
        self.write_pointer += 1

        buffSize=self.size()

        self.maxBX_First[(self.maxSize<buffSize)] = i_BX

        self.maxSize = np.maximum((buffSize),self.maxSize)
        self.maxBX_Last[(self.maxSize==buffSize)] = i_BX            

