import numpy as np
import pandas as pd


df=pd.read_csv('Data/ttbar_eolNoise_DAQ_data_0.csv')
df['waferLUV']=df.layer*10000 + df.waferu*100 + df.waferv
words_NZS=df.pivot(index='entry',columns='waferLUV',values='TotalWords_NZS').fillna(184).astype(int)
words_ZS=df.pivot(index='entry',columns='waferLUV',values='TotalWords').fillna(10).astype(int)

words_ZS.to_csv('Data/PacketSizes_ttbar_eolNoise_0.csv')
words_NZS.to_csv('Data/PacketSizes_ttbar_eolNoise_NZS_0.csv')
