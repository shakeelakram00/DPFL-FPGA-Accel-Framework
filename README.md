Steps to Implement DPFL-FPGA-Accel-Framework
1. [DPFL-FPGA-Accel-Framework_CompPro](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/DPFL-FPGA-Accel-Framework_CompPro) directory contains the DPFL-FPGA-Accel Framework complete flow for pre-built AI-Accel-1, AI-Accel-2, AI-Accel-3, and AI-Accel-4 for FPGA ZCU102.
2. The directory also contains the [Dependecies](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/Dependencies) related to the DPFL-FPGA-Accel-Framework on FPGA ZCU102. It is required to make sure they are incorporated.
3. The file in each subdirectory for each AI-Accel contains the [DPFL-FPGA-Accel flow] (https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/DPFL_FPGA_Accel12_Framework/DPFL-FPGA-Accel_12_Flow.ipynb) file to run and do the Design Space Exploration (DSE) concerning throughput, timing, accuracy, privacy, loss, number of clients contributing in DPFL, Number of Global Rounds, number of local epochs at each user etc.
4. The DSE can help develop an FPGA-Accel in the DPFL environment according to the required performance and privacy.

Steps to regenerate the existing [bit files](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/bitfiles) built using [FINN](https://github.com/Xilinx/finn)
1. Set up a FINN docker image on your system. 
2. Incorporate the [DPFL Dependencies](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/AI-Accel_build_Process/Dependencies.txt) in the image.
3. Run the [build files](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/AI-Accel_build_Process/AI-Accel-1and2/AI-Accel12_QDNNtoBit.ipynb) with your appropriate settings of PE, SIMD, InFiFo depth, and outFiFo depth in AI-AccelX_hw_config.json files.
4. Place the generated bit and hwh files with the updated configuration file in the [DPFL-FPGA-Accel-Framework_CompPro](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/DPFL-FPGA-Accel-Framework_CompPro) to run DPFL-FPGA-Accel Framework and do the DSE on the updated bit file.

Steps to build bit files on the QDNN Model setting other than the settings in [QDNNtoBit Files] (https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/AI-Accel_build_Process/AI-Accel-1and2/AI-Accel12_QDNNtoBit.ipynb) of AI-Accel-1, 2, 3, and 4.
1. Update the QDNN model in the [QDNNtoBit Files] (https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/AI-Accel_build_Process/AI-Accel-1and2/AI-Accel12_QDNNtoBit.ipynb).
2. Train the model using the same file and follow the file steps to generate the bit file on the new QDNN. Different PE, SIMD, InFifo depth, and outFiFo depth can be defined in the configuration file for the new QDNN. 
3. Change the existing model and .json configuration file name in [DPFL_Accel_5Ectopic_main](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/DPFL_FPGA_Accel34_Framework/DPFL_Accel_5Ectopic_main.py) with the new one.
4. Place the generated bit and hwh files with the updated configuration file in the [DPFL-FPGA-Accel-Framework_CompPro](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/DPFL-FPGA-Accel-Framework_CompPro) to run DPFL-FPGA-Accel Framework and do the DSE on the updated bit file. 


Steps to build bit files for applications other than the classification of cardiac arrhythmia
1. Gather the dataset and develop a QDNN on that. Make sure the dataset is in float32 data type, and to test the accelerator on the FPGA board generate the test dataset of uint8 data type. 
2. Update the QDNN model in the [QDNNtoBit Files] (https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/AI-Accel_build_Process/AI-Accel-1and2/AI-Accel12_QDNNtoBit.ipynb).
3. Train the model using the same file and follow the file steps to generate the bit file on the new QDNN. Different PE, SIMD, InFifo depth, and outFiFo depth can be defined in the configuration file for the new QDNN.
4. Change the existing model and .json configuration file name in [DPFL_Accel_5Ectopic_main](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/DPFL_FPGA_Accel34_Framework/DPFL_Accel_5Ectopic_main.py) with the new one.
5. Place the new dataset files in the same directory of DPFL-FPGA-Accel Flow and change the dataset files in [DPFL_Accel_5Ectopic_main](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/blob/main/DPFL-FPGA-Accel-Framework_CompPro/DPFL_FPGA_Accel34_Framework/DPFL_Accel_5Ectopic_main.py) with the new one.
6. Place the generated bit and hwh files with the updated configuration file in the [DPFL-FPGA-Accel-Framework_CompPro](https://github.com/shakeelakram00/DPFL-FPGA-Accel-Framework/tree/main/DPFL-FPGA-Accel-Framework_CompPro) to run DPFL-FPGA-Accel Framework and do the DSE on the updated bit file.

