## Secure Water Treatment testbed (SWaT)

SWaT is a fully operational scaled-down water treatment plant that produces doubly-filtered drinking water. SWaT consists of five water-processing tanks, as well as the pipes that connect those tanks. The in-coming valve and out-going valve of each tank can be controlled remotely via network. The objective of a self-adaptive SWaT system is to enable safe (e.g., no overflow or underflow in any of the tanks) and efficient (e.g., maximum clean water production) water filtering under different environmental situations (e.g., the
initial water level of the five tanks, and the in-coming water flow of the first tank). 

We extend a [SWaT simulator](https://sav.sutd.edu.sg/research/physical-attestation/sp-2018-paper-supplementary-material/) and implement our AMM mechanism (i.e., activeBSN.zip).

### The implementation of SWaT is as follows:
* The activeBSN/controlSWaT directory implements control-based self-adaptive SWaT.
* The activeBSN/supervision directory contains **AMM** (in subfolder detectors) and switcher (in subfolder safeguard).
* The activeBSN/main.py implements SWaT's main funtion, it sets initial water tank levels (i.e., **environmental input**) and implements water treatment processes.

### Running SWaT
* Run the simulation
    ```
	python3 main.py
    ```
* The trace file is in subfolder activeBSN/controlSWaT/trace.
