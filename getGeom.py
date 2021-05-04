import uproot

#_tree = uproot.open("root://cmseos.fnal.gov//store/user/dnoonan/HGCAL_Concentrator/triggerGeomV11Inp2.root",xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))['hgcaltriggergeomtester/TreeTriggerCells']
_tree = uproot.open("root://cmseos.fnal.gov//store/user/dnoonan/HGCAL_Concentrator/triggerGeomV10-2.root",xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))['hgcaltriggergeomtester/TreeTriggerCells']

df = _tree.pandas.df(['zside','subdet','layer','waferu','waferv','triggercellu','triggercellv','c_n'])

df.loc[df.subdet>1,'layer'] = df.loc[df.subdet>1,'layer'] + 28

df['HDM'] = df.c_n>4

x = df.groupby(['zside','subdet','layer','waferu','waferv']).any()

x[['HDM']].to_csv('geomInfo/moduleGeomV11.csv')
