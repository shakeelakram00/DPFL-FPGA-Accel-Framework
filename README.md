Steps to Implement DPFL-FPGA-Accel-Framework
1. [DPFL-FPGA-Accel-Framework_CompPro](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/DPFL-FPGA-Accel-Framework_CompPro) directory contains the DPFL-FPGA-Accel Framework complete flow for pre-built AI-Accel-1, AI-Accel-2, AI-Accel-3, and AI-Accel-4.
2. The directory also contains the [Dependecies](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/Dependencies) related to the DPFL-FPGA-Accel-Framework on FPGA ZCU102. It is required to make sure they are incorporated.
3. The file in each subdirectory for each AI-Accel contains the [DPFL-FPGA-Accel flow] (https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/DPFL_FPGA_Accel12_Framework/DPFL-FPGA-Accel_12_Flow.ipynb) file to run and do the Design Space Exploration (DSE) concerning throughput, timing, accuracy, privacy, loss, number of clients contributing in DPFL, Number of Global Rounds, number of local epochs at each user etc.
4. The DSE can help develop an FPGA-Accel in the DPFL environment according to the required performance and privacy.

Steps to regenerate the existing [bit files](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/bitfiles) built using [FINN](https://github.com/Xilinx/finn)
1. Set up a FINN docker image on your system. 
2. 


6. 
7. The DPFL is implemented on ZCU102 Board, to run for any other board, regenerate the bit files using 
8. Setup a FINN docker image on your system to regenerate the existing [bit files] using [AI-Accel_build_process] 

