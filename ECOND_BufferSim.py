import pandas as pd
import numpy as np
import awkward
import datetime
import argparse
from ECOND_Buffer import ECOND_Buffer

t_start = datetime.datetime.now()
t_last = datetime.datetime.now()

parser = argparse.ArgumentParser()
#parser.add_argument('-N',default="4000000")
parser.add_argument('-N',default="40000")
parser.add_argument('--files', default=1, type=int)
parser.add_argument('--source',default="eol")
parser.add_argument('--freqNZS',default=0, type=int)
args = parser.parse_args()

N_BX=int(eval(args.N))

# #load in data from csv file outputs of getDAQ_Data.py
# branchList = ['entry','layer','waferu','waferv','HDM','TotalWords', "TotalWords_NZS"]
# fName = "ttbar"
# if args.source=="oldTTbar":
#     fName = "ttbar"
# elif args.source=="updatedTTbar":
#     fName = "updated_ttbar"
# elif args.source=="eol":
#     fName = "ttbar_eolNoise"
# elif args.source=="startup":
#     fName = "ttbar_startupNoise"
# else:
#     print('unknown input')
#     exit()
# for i in range(args.files):
#     fileName = f'Data/%s_DAQ_data_{i}.csv'%fName
#     print(fileName)
#     if i==0:
#         daq_Data = pd.read_csv(fileName)[branchList]
#     else:
#         daq_Data = pd.concat([daq_Data, pd.read_csv(fileName)[branchList]])
# print(len(daq_Data))
# df=pd.read_csv('Data/ttbar_eolNoise_DAQ_data_0.csv')
# daq_Data['waferLUV']=daq_Data.layer*10000 + daq_D.waferu*100 + df.waferv
# words_NZS=df.pivot(index='entry',columns='waferLUV',values='TotalWords_NZS').fillna(184).astype(int)
# words_ZS=df.pivot(index='entry',columns='waferLUV',values='TotalWords').fillna(10).astype(int)

words_ZS=pd.read_csv('PacketSizes_ttbar_eolNoise_0.csv').set_index('entry')
words_NZS=pd.read_csv('PacketSizes_ttbar_eolNoise_NZS_0.csv').set_index('entry')


#get a list of the unique entry numbers
entryList = words_ZS.index.values

# #index the pandas datagrame by entry, layer, and waferu/v
# daq_Data.set_index(['entry','layer','waferu','waferv'],inplace=True)
# daq_Data.sort_index(inplace=True)

# #creates an 'empty' dataframe to store the number of words from all modules for a given event
# #  this is necessary because of the zero suppression that goes into the MC ntuples, and will keep data in a fixed order
# evt_Data = daq_Data.groupby(['layer','waferu','waferv']).any()[['HDM']]
# evt_Data['Words'] = 0

print ('Finished Setup')
t_now = datetime.datetime.now()
print ('     ',(t_now-t_last))
t_last = t_now

#LHC bunch structure (1 == filled bunch, where L1A could come from, 0 = empty bunch)
bunchStructure = np.array(
    ((([1]*72 + [0]*8)*3 +[0]*30)*2 +
    (([1]*72 + [0]*8)*4 +[0]*31) )*3 +
    (([1]*72 + [0]*8)*3 +[0]*30)*3 +
    [0]*81)
# rate, in terms of 1/N BX, for which an L1A should happen
triggerRate = 40e6/7.5e5 * sum(bunchStructure)/len(bunchStructure)

# list of buffers, where we simulate with a given number of eTx
econs = [ECOND_Buffer(163,nLinks=1,overflow=12*256),
         ECOND_Buffer(163,nLinks=2,overflow=12*256),
         ECOND_Buffer(163,nLinks=3,overflow=12*256),
         ECOND_Buffer(163,nLinks=4,overflow=12*256),
         ECOND_Buffer(163,nLinks=5,overflow=12*256),
         ECOND_Buffer(163,nLinks=6,overflow=12*256)
        ]
HGROCReadInBuffer = []
skipReadInBuffer=False

L1ACount=0


#start with an L1A issued in bx 0
evt = np.random.choice(entryList)
data  = words_ZS.loc[evt].values
#data = evt_Data['Words'].add(daq_Data.loc[evt,'TotalWords'],fill_value=0).astype(np.int16).values
# print('    -- ',data[:3])

#list to keep track of what would be in the HGCROC buffer
HGROCReadInBuffer.append(data)
#delay between when consecutive L1A's can be transmitted (to be checked if this is supposed to be 40 or 41)
readInDelay = 40
ReadInDelayCounter=readInDelay

count = 0
for iBX in range(1,N_BX+1):
    if iBX%(N_BX/50)==0:
        t_now = datetime.datetime.now()
        print('BX %i     '%iBX,(t_now-t_last))
        t_last = t_now
    orbitBX = iBX%3564
    #randomly decide if an L1A is issued in this BX
    hasL1A = np.random.uniform()<1./triggerRate and bunchStructure[orbitBX]

    # fill hist for buffer size and drain each econ
    for i in range(len(econs)):
        econs[i].fillHist()
        econs[i].drain()

    # remove one from read in delay counter
    if ReadInDelayCounter >0:
        ReadInDelayCounter -= 1

    # randomly pick an event, and add it to the HGCROC buffer
    if hasL1A:
        count+=1
        evt = np.random.choice(entryList)
        # dataStr = "TotalWords"
        if not args.freqNZS==0 and count%args.freqNZS==0:
            data=words_NZS.loc[evt].values
        else:
            data=words_ZS.loc[evt].values
        # if not args.freqNZS==0 and count%args.freqNZS==0:
        #     dataStr = "TotalWords_NZS"
        # data  = evt_Data['Words'].add(daq_Data.loc[evt,dataStr],fill_value=0).astype(np.int16).values

        HGROCReadInBuffer.append(data)

    # add event from HGCROC buffer to the ECOND buffer, accounting for the read in delay (reset read in delay counter as well)
    if len(HGROCReadInBuffer)>0 and (ReadInDelayCounter==0 or skipReadInBuffer):
        L1ACount += 1
        ReadInDelayCounter = readInDelay
        data = HGROCReadInBuffer[0]
        HGROCReadInBuffer = HGROCReadInBuffer[1:]
        for i in range(len(econs)): 
            econs[i].write(data.copy(), iBX)

print(f'{L1ACount} L1As issued')
print()
for i in range(len(econs)):
    pass
    print(f'{i+1} eTx')
    print('overflows=',econs[i].overflowCount.tolist())
    print('truncations=',econs[i].truncateCount.tolist())
    print('maxSize=',econs[i].maxSize.tolist())
    print('maxBX_First=',econs[i].maxBX_First.tolist())
    print('maxBX_Last=',econs[i].maxBX_Last.tolist())
    print("sizeHist=", econs[i].hist.reshape(163,3072).tolist())
    print()
