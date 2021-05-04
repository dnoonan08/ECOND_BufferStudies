# ECON-D Buffer Simulation

Simulation of the buffer behavior of the ECON-D.  Data is taken from MC simulation, and event packet sizes are estimated.  The data sizes are then used in simulation of a series of BX, tracking the buffer size within the ECON-D through the whole process.

##### envSetup.sh
Shell script for setting up python virtual environment on cmslpc

##### getDAQ_Data.py
Reads data from MC, finding event packet sizes (based on number of zero suppressed cells on each eRx).

##### ECOND_BufferSim.py
Uses data packets sizes from output of getDAQ_Data.py, and simulates buffers on ECON-D.  Randomly decides if a BX will have an L1A issued (taking into account LHC bunch structure).  Each L1A, a packet size is randomly sampled from the MC data, and added to the buffer size.  Simulation is run assuming 1 to 6 eTx, and each BX, 32-bits are drained from the size.
