## Rice University Bidding System (RUBiS)

RUBiS is a web auction system that can adapt to workload/network changes by adjusting the number of servers to satisfy quality-of-service levels. We extend a [RUBiS simulator](https://github.com/cps-sei/swim) and implement our AMM mechanism (i.e.,  activeRUBiS.zip).

### The implementation of RUBiS is as follows:
* The activeRUBiS/simulations/rubisPIC directory contains system configuration (e.g., rubisPIC.ini) and system architecture (e.g., rubisPIC.ned) files.
* The activeRUBiS/simulations/workload directory contains user workloads (i.e., **environmental input**).
* The activeRUBiS/src/controlRUBiS directory implements control-based self-adaptive RUBiS.
* The activeRUBiS/src/supervision directory contains **AMM** (in subfolder detectors), mandatory controller and switcher (in subfolder safeguardX).
* The activeBSN/main.sh is the script of running RUBiS.

### Installation
* Download OMNeT++ 6.0 and install it following the installation guide 
* Make sure that OMNeT++ bin directory is in the PATH, otherwise add it:
   ```
   export PATH=$PATH:/home/your path/omnetpp-6.0/bin
   ```
* Install boost
   ```
   sudo apt-get install libboost-all-dev
   ```
* Install required packages: numpy, sklearn, scipy and cvxopt
* Unzip the files **queueinglib.zip** and **activeRUBiS.zip** in *RUBiS* directory
* Go to *RUBiS* and compile queueinglib
   ```
   cd queueinglib
   make
   ```
* Go to *RUBiS* and compile activeRUBiS
   ```
   cd activeRUBiS
   make  
   ```
   
### Running RUBiS
* Run the simulation
    ```
    cd RUBiS/activeRUBiS
    ./main.sh
    ```
* The trace file is in subfolder activeRUBiS/simulations/traces.