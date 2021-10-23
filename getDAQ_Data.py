import uproot3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i',default=0,type=int)
parser.add_argument('--filesPerJob',default=10,type=int)
parser.add_argument('--source',default="eol")
args = parser.parse_args()

pd.options.mode.chained_assignment = None

adcLSB_ = 100./1024.
tdcLSB_ = 10000./4096.
tdcOnsetfC_ = 60.

jobNumber_eventOffset = 100000
negZ_eventOffset      =  50000
cassette_eventOffset  =   1000

linkMap = pd.read_csv('geomInfo/eLinkInputMapFull.csv')
calibrationCells = pd.read_csv('geomInfo/calibrationCells.csv')

waferRemap = pd.read_csv('geomInfo/WaferNumberingMatch.csv')[['layer','waferu','waferv','C1_waferu','C1_waferv','Cassette']]
waferRemap.set_index(['layer','waferu','waferv'],inplace=True)


def getTree(fNumber=1,fNameBase = 'root://cmseos.fnal.gov//store/user/lpchgcal/ConcentratorNtuples/L1THGCal_Ntuples/TTbar_v11/ntuple_ttbar_ttbar_v11_aged_unbiased_20191101_%i.root'):

    treeName = 'hgcalTriggerNtuplizer/HGCalTriggerNtuple'

    fName = fNameBase%fNumber
    print ("File %s"%fName)

    try:
        _tree = uproot3.open(fName,xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))[treeName]
        return _tree
    except:
        print ("---Unable to open file, skipping")
        return None



def getDF(_tree, fNumber, Nstart=0, Nstop=2, layerStart=5,layerStop=9):

    branchesOld = ['hgcdigi_zside','hgcdigi_layer','hgcdigi_waferu','hgcdigi_waferv','hgcdigi_cellu','hgcdigi_cellv','hgcdigi_wafertype','hgcdigi_data','hgcdigi_isadc','hgcdigi_dataBXm1','hgcdigi_isadcBXm1']
    branchesNew = ['hgcdigi_zside','hgcdigi_layer','hgcdigi_waferu','hgcdigi_waferv','hgcdigi_cellu','hgcdigi_cellv','hgcdigi_wafertype','hgcdigi_data_BX2','hgcdigi_isadc_BX2','hgcdigi_toa_BX2','hgcdigi_gain_BX2','hgcdigi_data_BX1','hgcdigi_isadc_BX1']
    # print(_tree.keys())

    if b'hgcdigi_data' in _tree.keys():
        fulldf = _tree.pandas.df(branchesOld,entrystart=Nstart,entrystop=Nstop)
    else:
        fulldf = _tree.pandas.df(branchesNew,entrystart=Nstart,entrystop=Nstop)
    fulldf.columns = ['zside','layer','waferu','waferv','cellu','cellv','wafertype','data','isadc','toa','gain','data_BXm1','isadc_BXm1']

    #select layers

    layerCut = (fulldf.layer>=layerStart) & (fulldf.layer<=layerStop)
    fulldf = fulldf[layerCut]

    #drop subentry from index
    fulldf.reset_index('subentry',drop=True,inplace=True)

    fulldf.reset_index(inplace=True)

    #update entry number for negative endcap
    fulldf['entry'] = fulldf['entry'] + fNumber*jobNumber_eventOffset
    fulldf.loc[fulldf.zside==-1, 'entry'] = fulldf.loc[fulldf.zside==-1, 'entry'] + negZ_eventOffset

    return fulldf


def processDF(fulldf, outputName="test.csv", append=False):

    #if data is ADC, charge = data * adcLSB
    #else data is TDC, charge = tdcStart  + data*tdcLSB

    fulldf["charge"] = np.where(fulldf.isadc==1,fulldf.data*adcLSB_, (int(tdcOnsetfC_/adcLSB_) + 1.0)*adcLSB_ + fulldf.data*tdcLSB_)
    fulldf["charge_BXm1"] = np.where(fulldf.isadc_BXm1==1,fulldf.data_BXm1*adcLSB_, (int(tdcOnsetfC_/adcLSB_) + 1.0)*adcLSB_ + fulldf.data*tdcLSB_)

    #ZS_thr = np.array([1.03 , 1.715, 2.575]) #0.5 MIP threshold, in fC, as found in CMSSW
    ZS_thr = np.array([5, 5, 5]) #5 ADC threshold for each wafer type
    ZS_thr_BXm1 = ZS_thr*5 #2.5 MIP threshold, in fC, as found in CMSSW
    # ZS_thr = np.array([0.7, 0.7, 0.7])
    # ToA_thr = 12. # below this, we don't send ToA, above this we do, 12 fC is threshold listed in TDF

    #drop cells below ZS_thr
    #Correction for leakage from BX1, following Pedro's suggestion
    #https://github.com/cms-sw/cmssw/blob/master/SimCalorimetry/HGCalSimAlgos/interface/HGCalSiNoiseMap.icc#L18-L26
    #80fC for 120um, 160 fC for 200 um and 320 fC for 300 um
    BX1_leakage = np.array([0.066/0.934, 0.153/0.847, 0.0963/0.9037])
    fulldf['data'] = fulldf.data-fulldf.data_BXm1*BX1_leakage[fulldf.gain]
    zsCut = np.where(fulldf.isadc==0, False, fulldf.data>ZS_thr[fulldf.wafertype])
    df_ZS = fulldf.loc[zsCut]

    df_ZS['BXM1_readout'] = np.where(df_ZS.isadc_BXm1==0, False,  df_ZS.data_BXm1>ZS_thr_BXm1[df_ZS.wafertype])
    df_ZS['TOA_readout'] = (df_ZS.toa>0).astype(int)
    df_ZS['TOT_readout'] = ~df_ZS.isadc

    df_ZS['Bits'] = 16 + 8*(df_ZS.BXM1_readout + df_ZS.TOA_readout)


    df_ZS.set_index(['zside','layer','waferu','waferv'],inplace=True)
    df_ZS['HDM'] = df_ZS.wafertype==0

    df_ZS.reset_index(inplace=True)
    df_ZS.set_index(['entry','zside','layer','waferu','waferv'],inplace=True)

    df_ZS = df_ZS.reset_index().merge(linkMap,on=['HDM','cellu','cellv']).set_index(['entry','zside','layer','waferu','waferv'])

    calCellDF = df_ZS.reset_index().merge(calibrationCells,on=['HDM','cellu','cellv']).set_index(['entry','zside','layer','waferu','waferv']).fillna(0).drop('isCal',axis=1)
    calCellDF['linkChannel'] = 32
    calCellDF.loc[calCellDF.HDM,'linkChannel'] = 36
    calCellDF = calCellDF.reset_index().set_index(['HDM','eLink','linkChannel'])
    calCellDF.SRAM_read_group = linkMap.set_index(['HDM','eLink','linkChannel']).SRAM_read_group
    calCellDF = calCellDF.reset_index().set_index(['entry','zside','layer','waferu','waferv'])

    df_ZS = pd.concat([df_ZS,calCellDF]).sort_index()


    group = df_ZS.reset_index()[['entry','zside','layer','waferu','waferv','eLink','HDM','Bits','BXM1_readout','TOA_readout']].groupby(['entry','zside','layer','waferu','waferv','eLink'])

    dfBitsElink = group.sum()
    dfBitsElink['HDM'] = group[['HDM']].any()
    dfBitsElink['occ'] = group['HDM'].count()

    dfBitsElink['eRxPacket_Words'] = (dfBitsElink.Bits/32+1).astype(int) + 2


    group = dfBitsElink.reset_index()[['entry','zside','layer','waferu','waferv','HDM','occ','eRxPacket_Words']].groupby(['entry','zside','layer','waferu','waferv'])
    del dfBitsElink
    dfBits = group.sum()
    dfBits['HDM'] = group[['HDM']].any()
    dfBits['NonEmptyLinks'] = group[['HDM']].count()
    dfBits['EmptyLinks'] = np.where(dfBits.HDM,12,6) - dfBits.NonEmptyLinks


    evt_headerWords = 2
    evt_trailerWords = 2
    dfBits['TotalWords'] = evt_headerWords + dfBits.eRxPacket_Words + dfBits.EmptyLinks + evt_trailerWords

    dfBits.reset_index(inplace=True)
    dfBits.set_index(['layer','waferu','waferv'],inplace=True)

    #relabel wafers to take advantage of phi symmetry
    dfBits = dfBits.merge(waferRemap,left_index=True,right_index=True)
    dfBits.reset_index(inplace=True)
    dfBits.entry = dfBits.entry + cassette_eventOffset*dfBits.Cassette
    dfBits.waferu = dfBits.C1_waferu
    dfBits.waferv = dfBits.C1_waferv
    dfBits.drop(['C1_waferu','C1_waferv','Cassette','zside'],axis=1,inplace=True)
    dfBits.set_index(['entry','layer','waferu','waferv'],inplace=True)

    dfBits.sort_index()

    if append:
        dfBits.to_csv(outputName,mode='a',header=False)
    else:
        dfBits.to_csv(outputName)

    del dfBits



if args.source=="old":
    jobs=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,27,28,29,30,35,36,37,38,39,43,45,46,47,48,51,53,54,55,56,57,58,59,60,61,62,63,64,65,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,84,85,87,88,89,90,91,92,93,94,95,96,98,99]
    fNameBase = 'root://cmseos.fnal.gov//store/user/lpchgcal/ConcentratorNtuples/L1THGCal_Ntuples/TTbar_v11/ntuple_ttbar_ttbar_v11_aged_unbiased_20191101_%i.root'
    outputName = f"Data/updated_ttbar_DAQ_data_{args.i}.csv"
elif args.source=="startup":
    print('Startup')
    jobs=range(60)
    fNameBase = 'root://cmseos.fnal.gov//store/user/dnoonan/HGCAL_Concentrator/TTbar_v11/ntuple_ttbar_D49_1120pre1_PU200_eolupdate_startup_qua_20200723_%i.root'
    outputName = f"Data/ttbar_startupNoise_DAQ_data_{args.i}.csv"
elif args.source=="eol": 
    jobs=range(1)
    fNameBase = 'root://cmseos.fnal.gov//store/user/rverma/Output/HGCAL_Concentrator/ntuple_ttbar_D49_1120pre1_PU200_eolupdate_qua_20200723_%i.root'
    outputName = f"Data/ttbar_eolNoise_DAQ_data_{args.i}.csv"
else:
    print('unknown source')
    exit()
print('Beginning')

append=False

for job in jobs[args.i*args.filesPerJob:(args.i+1)*args.filesPerJob]:
    print(job)

    _tree = getTree(fNumber=job,fNameBase=fNameBase)

    Nentries = _tree.numentries

    chunkSize=10



    for chunkStart in range(0, Nentries, chunkSize):
        print(chunkStart)
        fulldf = getDF(_tree, fNumber = job, Nstart=chunkStart, Nstop=chunkStart+chunkSize, layerStart=5, layerStop=9)

        processDF(fulldf, outputName=outputName, append=append)

        append=True
