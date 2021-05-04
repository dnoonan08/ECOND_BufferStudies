import pandas as pd
import numpy as np

import awkward

import datetime

t_start = datetime.datetime.now()
t_last = datetime.datetime.now()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N',default="40000000")
parser.add_argument('--source',default="eol")
args = parser.parse_args()


N_BX=int(eval(args.N))


if args.source=="oldTTbar":
    daq_Data = pd.read_csv(f'Data/ttbar_DAQ_data_0.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]

    for i in range(1,16):
        daq_Data = pd.concat([daq_Data, pd.read_csv(f'Data/ttbar_DAQ_data_{i}.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]])
elif args.source=="updatedTTbar":
    daq_Data = pd.read_csv(f'Data/updated_ttbar_DAQ_data_0.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]

    for i in range(1,16):
        daq_Data = pd.concat([daq_Data, pd.read_csv(f'Data/updated_ttbar_DAQ_data_{i}.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]])
elif args.source=="eol":
    jobNumbers = np.random.choice(range(8),2,replace=False)
    
    daq_Data = pd.read_csv(f'Data/ttbar_eolNoise_DAQ_data_{jobNumbers[0]}.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]

    for i in jobNumbers[1:]:
        print(i)
        daq_Data = pd.concat([daq_Data, pd.read_csv(f'Data/ttbar_eolNoise_DAQ_data_{i}.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]])
elif args.source=="startup":
    jobNumbers = np.random.choice(range(8),2,replace=False)

    daq_Data = pd.read_csv(f'Data/ttbar_startupNoise_DAQ_data_{jobNumbers[0]}.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]

    for i in jobNumbers[1:]:
        daq_Data = pd.concat([daq_Data, pd.read_csv(f'Data/ttbar_startupNoise_DAQ_data_{i}.csv')[['entry','layer','waferu','waferv','HDM','TotalWords']]])
else:
    print('unknown input')
    exit()
print(len(daq_Data))
entryList = daq_Data.entry.unique()
print(len(entryList))

daq_Data.set_index(['entry','layer','waferu','waferv'],inplace=True)
daq_Data.sort_index(inplace=True)

evt_Data = daq_Data.groupby(['layer','waferu','waferv']).any()[['HDM']]
evt_Data['Words'] = 0

print ('Finished Setup')
t_now = datetime.datetime.now()
print ('     ',(t_now-t_last))
t_last = t_now

bunchStructure = np.array(
    ((([1]*72 + [0]*8)*3 +[0]*30)*2 +
    (([1]*72 + [0]*8)*4 +[0]*31) )*3 +
    (([1]*72 + [0]*8)*3 +[0]*30)*3 +
    [0]*81)


triggerRate = 40e6/7.5e5 * sum(bunchStructure)/len(bunchStructure)

import numba

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
        self.buffer = np.zeros(nModules*buffSize,np.int16)
        self.starts=np.arange(nModules)*buffSize                                             
        self.write_pointer=np.arange(nModules)*buffSize        

        self.buffSize = buffSize
        self.nLinks = nLinks
        self.overflow = overflow

        self.maxSize = np.zeros(nModules,dtype=np.uint16)
        self.maxBX_First = np.zeros(nModules,dtype=np.uint32)
        self.maxBX_Last = np.zeros(nModules,dtype=np.uint32)
        self.overflowCount = np.zeros(nModules,dtype=np.uint32)

    def drain(self):
        bufferNotEmpty = self.write_pointer>self.starts
        self.buffer[self.starts[bufferNotEmpty]] -= self.nLinks

        isEmpty = self.buffer[ self.starts ]< 0
        while isEmpty.sum()>0:
            lenBuffer = self.write_pointer-self.starts

            pushForward = self.starts[(isEmpty) & (lenBuffer>1)]

            self.buffer[ pushForward + 1 ] += self.buffer[ pushForward ]

            moveBuffer(self.buffer, self.starts[isEmpty],self.buffSize)
            self.write_pointer[isEmpty] -= 1
            
            isEmpty = self.buffer[ self.starts ]< 0

    def size(self):
        return _size(self.buffer.copy(), self.starts.copy(), self.write_pointer.copy())

    def get(self):
        return _get(self.buffer.copy(), self.starts.copy(), self.write_pointer.copy())

    
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



econs = [ECOND_Buffer(163,50,nLinks=1,overflow=12*128),
         ECOND_Buffer(163,50,nLinks=2,overflow=12*128),
         ECOND_Buffer(163,50,nLinks=3,overflow=12*128),
         ECOND_Buffer(163,50,nLinks=4,overflow=12*128),
         ECOND_Buffer(163,50,nLinks=5,overflow=12*128),
         ECOND_Buffer(163,50,nLinks=6,overflow=12*128)
        ]




HGROCReadInBuffer = []
skipReadInBuffer=False


L1ACount=0

#start with an L1A issued in bx 0
evt = np.random.choice(entryList)
data  = evt_Data['Words'].add(daq_Data.loc[evt,'TotalWords'],fill_value=0).astype(np.int16).values
# print('    -- ',data[:3])

HGROCReadInBuffer.append(data)
ReadInDelayCounter=40

for iBX in range(1,N_BX+1):
    if iBX%(N_BX/50)==0:
        t_now = datetime.datetime.now()
        print('BX %i     '%iBX,(t_now-t_last))
        t_last = t_now

    orbitBX = iBX%3564

    hasL1A = np.random.uniform()<1./triggerRate and bunchStructure[orbitBX]
    
    for i in range(len(econs)):
        econs[i].drain()
    if ReadInDelayCounter >0:
        ReadInDelayCounter -= 1
    
    if hasL1A:
        evt = np.random.choice(entryList)
        data  = evt_Data['Words'].add(daq_Data.loc[evt,'TotalWords'],fill_value=0).astype(np.int16).values

        HGROCReadInBuffer.append(data)

    if len(HGROCReadInBuffer)>0 and (ReadInDelayCounter==0 or skipReadInBuffer):
        L1ACount += 1
        ReadInDelayCounter = 40
        data = HGROCReadInBuffer[0]
#         print('   --- ', data[:5])
        HGROCReadInBuffer = HGROCReadInBuffer[1:]

        for i in range(len(econs)): 
            econs[i].write(data.copy(), iBX)


print(f'{L1ACount} L1As issued')
print()
for i in range(len(econs)):
    print(f'{i+1} eTx')
    print('overflows=',econs[i].overflowCount.tolist())
    print('maxSize=',econs[i].maxSize.tolist())
    print('maxBX_First=',econs[i].maxBX_First.tolist())
    print('maxBX_Last=',econs[i].maxBX_Last.tolist())
    print()
