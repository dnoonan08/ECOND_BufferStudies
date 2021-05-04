#!/bin/bash

sampleType=$1
echo $sampleType
#If running on condor, checkout CMSSW and get extra libraries
if [ -z ${_CONDOR_SCRATCH_DIR} ] ; then 
    echo "Running Interactively" ; 
else
    echo "Running In Batch"
    (>&2 echo "Starting job on " `date`) # Date/time of start of job
    (>&2 echo "Running on: `uname -a`") # Condor job is running on this node
    (>&2 echo "System software: `cat /etc/redhat-release`") # Operating System on that node

    cd ${_CONDOR_SCRATCH_DIR}
    echo ${_CONDOR_SCRATCH_DIR}

    xrdcp root://cmseos.fnal.gov//store/user/dnoonan/envhgcalPythonEnv.tar.gz .
    tar -zxf envhgcalPythonEnv.tar.gz
    source hgcalPythonEnv/bin/activate

    rm envhgcalPythonEnv.tar.gz

    xrdcp root://cmseos.fnal.gov//store/user/dnoonan/TTbar_DAQ_Data.tgz .
    tar -zxf TTbar_DAQ_Data.tgz
    rm TTbar_DAQ_Data.tgz
fi

python ECOND_BufferSim.py --source ${sampleType}

if [ -z ${_CONDOR_SCRATCH_DIR} ] ; then 
    echo "Running Interactively" ; 
else
    echo "Cleanup" 
    rm ECOND_BufferSim.py
    rm -rf Data
    rm -rf hgcalPythonEnv

    rm *.tgz
    rm *py
fi
