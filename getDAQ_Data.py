import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i',default=0,type=int)
parser.add_argument('--filesPerJob',default=10,type=int)
parser.add_argument('--source',default="eol")
parser.add_argument('--layerStart',default=5)
parser.add_argument('--layerStop',default=9)
parser.add_argument('--chunkSize',default=10,type=int)
parser.add_argument('--maxEvents',default=None, type=int)
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
        _tree = uproot.open(fName,xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))[treeName]
        return _tree
    except:
        print ("---Unable to open file, skipping")
        return None

def processDF(fulldf, outputName="test.csv", append=False):

    #if data is ADC, charge = data * adcLSB
    #else data is TDC, charge = tdcStart  + data*tdcLSB

    fulldf["charge"] = np.where(fulldf.isadc==1,fulldf.data*adcLSB_, (int(tdcOnsetfC_/adcLSB_) + 1.0)*adcLSB_ + fulldf.data*tdcLSB_)
    fulldf["charge_BXm1"] = np.where(fulldf.isadc_BXm1==1,fulldf.data_BXm1*adcLSB_, (int(tdcOnsetfC_/adcLSB_) + 1.0)*adcLSB_ + fulldf.data_BXm1*tdcLSB_)

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

    dfBitsElink['eRxPacket_Words'] = np.ceil(dfBitsElink.Bits/32).astype(int) + 2
    dfBitsElink['eRxPacket_Words_NZS'] = np.ceil((37*24+ 8*dfBitsElink.TOA_readout)/32).astype(int)+2


    group = dfBitsElink.reset_index()[['entry','zside','layer','waferu','waferv','HDM','occ','eRxPacket_Words', 'eRxPacket_Words_NZS']].groupby(['entry','zside','layer','waferu','waferv'])
    del dfBitsElink
    dfBits = group.sum()
    dfBits['HDM'] = group[['HDM']].any()
    dfBits['NonEmptyLinks'] = group[['HDM']].count()
    dfBits['EmptyLinks'] = np.where(dfBits.HDM,12,6) - dfBits.NonEmptyLinks


    evt_headerWords = 2
    evt_trailerWords = 2
    dfBits['TotalWords'] = evt_headerWords + dfBits.eRxPacket_Words + dfBits.EmptyLinks + evt_trailerWords
    dfBits['TotalWords_NZS'] = evt_headerWords + dfBits.eRxPacket_Words_NZS + 30*dfBits.EmptyLinks +  evt_trailerWords

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
    fNumberList=['4807066', '4807067', '4807068', '4807069', '4807070', '4807071', '4807072', '4807073', '4807074', '4807075', '4807076', '4807077', '4807078', '4807079', '4807080', '4807081', '4807082', '4807083', '4807084', '4807085', '4807086', '4807087', '4807088', '4807089', '4807090', '4807091', '4807092', '4807093', '4807094', '4807095', '4807096', '4807097', '4807098', '4807099', '4807100', '4807101', '4807102', '4807103', '4807104', '4807105', '4807106', '4807107', '4807108', '4807109', '4807110', '4807111', '4807112', '4807113', '4807114', '4807115', '4807116', '4807117', '4807118', '4807119', '4807120', '4807121', '4807122', '4807123', '4807124', '4807125', '4807126', '4807127', '4807128', '4807129', '4807130', '4807131', '4807132', '4807133', '4807134', '4807135', '4807136', '4807137', '4807138', '4807139', '4807140', '4807141', '4807142', '4807143', '4807144', '4807145', '4807146', '4807147', '4807148', '4807149', '4807150', '4807151', '4807152', '4807153', '4807154', '4807155', '4807156', '4807157', '4807158', '4807159', '4807160', '4807161', '4807162', '4807163', '4807164', '4807165', '4807166', '4807167', '4807168', '4807169', '4807170', '4807171', '4807172', '4807173', '4807174', '4807175', '4807176', '4807177', '4807178', '4807179', '4807180', '4807181', '4807182', '4807183', '4807184', '4807185', '4807186', '4807187', '4807188', '4807189', '4807190', '4807191', '4807192', '4807193', '4807194', '4807195', '4807196', '4807197', '4807198', '4807199', '4807200', '4807201', '4807202', '4807203', '4807204', '4807205', '4807206', '4807207', '4807208', '4807209', '4807210', '4807211', '4807212', '4807213', '4807214', '4807215', '4807216', '4807217', '4807218', '4807219', '4807220', '4807221', '4807222', '4807223', '4807224', '4807225', '4807226', '4807227', '4807228', '4807229', '4807230', '4807231', '4807232', '4807233', '4807234', '4807235', '4807236', '4807237', '4807238', '4807239', '4807240', '4807241', '4807242', '4807243', '4807244', '4807245', '4807246', '4807247', '4807248', '4807249', '4807250', '4807251', '4807252', '4807253', '4807254', '4807255', '4807256', '4807257', '4807258', '4807259', '4807260', '4807261', '4807262', '4807263', '4807264', '4807265', '4807266', '4807267', '4807268', '4807269', '4807270', '4807271', '4807273', '4807274', '4807275', '4807276', '4807277', '4807278', '4807279', '4807280', '4807281', '4807282', '4807283', '4807284', '4807285', '4807286', '4807287', '4807288', '4807289', '4807290', '4807291', '4807292', '4807293', '4807294', '4807295', '4807296', '4807297', '4807298', '4807299', '4807300', '4807301', '4807302', '4807303', '4807304', '4807305', '4807306', '4807307', '4807308', '4807309', '4807310', '4807311', '4807312', '4807313', '4807314', '4807315', '4807316', '4807317', '4807318', '4807319', '4807320', '4807321', '4807322', '4807323', '4807324', '4807325', '4807326', '4807327', '4807328', '4807329', '4807330', '4807331', '4807332', '4807333', '4807334', '4807335', '4807336', '4807337', '4807338', '4807339', '4807340', '4807341', '4807342', '4807343', '4807344', '4807345', '4807346', '4807347', '4807348', '4807349', '4807350', '4807351', '4807352', '4807353', '4807354', '4807355', '4807356', '4807357', '4807358', '4807359', '4807360', '4807361', '4807362', '4807363', '4807364', '4807365', '4807366', '4807367', '4807368', '4807369', '4807370', '4807371', '4807372', '4807373', '4807374', '4807375', '4807376', '4807377', '4807378', '4807379', '4807380', '4807381', '4807382', '4807383', '4807384', '4807385', '4807387', '4807388', '4807390', '4807391', '4807393', '4807394', '4807395', '4807396', '4807398', '4807399', '4807401', '4807402', '4807404', '4807405', '4807406', '4807408', '4807410', '4807411', '4807413', '4807414', '4807415', '4807417', '4807418', '4807420', '4807421', '4807423', '4807424', '4807425', '4807427', '4807428', '4807430', '4807432', '4807433', '4807434', '4807436', '4807437', '4807439', '4807440', '4807442', '4807443', '4807445', '4807446', '4807448', '4807449', '4807451', '4807452', '4807453', '4807454', '4807456', '4807458', '4807459', '4807461', '4807462', '4807464', '4807465', '4807467', '4807468', '4807470', '4807472', '4807473', '4807475', '4807476', '4807477', '4807479', '4807480', '4807482', '4807483', '4807485', '4807486', '4807488', '4807489', '4807490', '4807492', '4807493', '4807495', '4807496', '4807498', '4807499', '4807501', '4807502']
    jobs=range(len(fNumberList))
    fNameBase = 'root://cmseos.fnal.gov//store/user/rverma/Output/cms-hgcal-econd/ntuple/ntuple_Events_%s_0.root'
    outputName = f"Data/ttbar_eolNoise_DAQ_data_{args.i}.csv"
else:
    print('unknown source')
    exit()
print('Beginning')

append=False
print(jobs)
for job in jobs[args.i*args.filesPerJob:(args.i+1)*args.filesPerJob]:
    print(job)
    print(fNumberList[job])
    _tree = getTree(fNumber=fNumberList[job],fNameBase=fNameBase)
    print(_tree.num_entries)
    branchesOld = ['hgcdigi_zside','hgcdigi_layer','hgcdigi_waferu','hgcdigi_waferv','hgcdigi_cellu','hgcdigi_cellv','hgcdigi_wafertype','hgcdigi_data','hgcdigi_isadc','hgcdigi_dataBXm1','hgcdigi_isadcBXm1']
    branchesNew = ['hgcdigi_zside','hgcdigi_layer','hgcdigi_waferu','hgcdigi_waferv','hgcdigi_cellu','hgcdigi_cellv','hgcdigi_wafertype','hgcdigi_data_BX2','hgcdigi_isadc_BX2','hgcdigi_toa_BX2','hgcdigi_gain_BX2','hgcdigi_data_BX1','hgcdigi_isadc_BX1']

    if b'hgcdigi_data' in _tree.keys():
        branches = branchesOld
    else:
        branches = branchesNew

    N=0
    for x in _tree.iterate(branches,entry_stop=args.maxEvents,step_size=args.chunkSize):
        print(N)
        N += args.chunkSize
        layerCut = (x['hgcdigi_layer']>=args.layerStart) & (x['hgcdigi_layer']<=args.layerStop)
        df = ak.to_pandas(x[layerCut])
        df.columns = ['zside','layer','waferu','waferv','cellu','cellv','wafertype','data','isadc','toa','gain','data_BXm1','isadc_BXm1']

        #drop subentry from index
        df.reset_index('subentry',drop=True,inplace=True)
        df.reset_index(inplace=True)

        #update entry number for negative endcap
        df['entry'] = df['entry'] + job*jobNumber_eventOffset + N
        df.loc[df.zside==-1, 'entry'] = df.loc[df.zside==-1, 'entry'] + negZ_eventOffset

        processDF(df, outputName=outputName, append=append)

        append=True
