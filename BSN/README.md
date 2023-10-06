## Body Sensor Network (BSN)

BSN is a real-time health monitoring system based on a wireless sensor network. It consists of six sensors that collect different types of vital signs (e.g., blood oxygen, heart rate, body temperature, etc.) and a data processing hub that provides an overall health assessment. The power consumption of each sensor is determined by its sampling rate (which can be adjusted remotely), and its sensing
noise (which is determined by the environmental uncertainty). In a self-adaptive BSN, each of the sensors is controlled by a PI-based managing system that adjusts the sensor’s power consumption to a setpoint by changing the sampling rate with a control signal.

We extend the BSN (https://github.com/lesunb/bsn) and implement our AMM mechanism (i.e., activebsn.zip).

### The implementation of BSN is as follows:
* The activebsn/controlbsn directory implements control-based self-adaptive BSN.
* The activebsn/controlbsn/src/sa-bsn/configurations/simulation directory contains sensor **environmental input** file.
* The activebsn/supervision directory contains **AMM** (in subfolder detectors).
* The activebsn/main.sh is the script of running BSN.

### Installation
* Requirements
	```
	Ubuntu 20.04 LTS
	ROS Noetic
	Python3 packages: numpy, sklearn, scipy, and cvxopt
	```

* Unzip the files **activebsn.zip**
* Go to *activebsn/controlbsn* and compile it:
	```
	cd activebsn/controlbsn
	catkin_make
	```

### Running BSN
* Run the simulation
    ```
    cd activebsn
    ./main.sh
    ```
* The trace file is in subfolder activebsn/controlbsn/src/sa-bsn/system_manager/enactor/traces.