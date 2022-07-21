## Experiments of the paper *Variance-Aware Weight Initialization for Point Convolutional Neural Networks*

This is the code of the experiments presented in the paper: *Variance-Aware Weight Initialization for Point Convolutional Neural Networks*.

**_NOTE:_**  This repository only contains the code of the experiments. The CUDA implementation of the different convolution operators and the point cloud library in Pytorch needs to be installed separately from the following [repository](https://github.com/phermosilla/pccnn_lib). See the *Required libraries* section in this file.

### Required libraries

This code was tested in python 3.7.
In order to install the required python libraries use pip with the following command:

        pip install -r requirements.txt

##### PCCNN Library

The library to process point clouds in Pytorch (including the convolution operators) can be found in the following [repository](https://github.com/phermosilla/pccnn_lib).
Install this library first in order to run the experiments.

### Data Download

##### ScanObjectNN

This data set can be downloaded [here](https://hkust-vgd.github.io/scanobjectnn/).
Once downloaded, uncompress the file in the *ScanObjectNN* folder and rename the folder to *data*.

##### ScanNet

This data set can be downloaded [here](http://www.scan-net.org/).

##### ModelNet40

The preprocessed dataset can be downloaded from the following [link](https://drive.google.com/file/d/1MPzU0EDwHS0Wzh9A73bsDPbmjR2sNq-R/view?usp=sharing).
Once downloaded, uncompress the file in the *ModelNet40* folder.

### Run Experiments

In order to run the experiments, first you will need to create the folder where the log files will be stored.
This is the code to create this folder for the *ScanObjectNN* task. 
For the other experiments we will need to create the same folder.

        cd ModelNet40
        mkdir runs
        
Once this folder is created, we can execute the bash script to run the different configurations (w/o our weight init scheme, w/o batch norm, w/o group norm, etc.).

        sh train_script.sh
        
If we want to use a different convolution operation, we need to modify the bash script and change the input parameter --conv_type with a valid convolution operation: ***mcconv***, ***kpconv***, ***kpconvn***, ***pccnn***, ***pointconv***, or ***sphconv***.

Once the training is finished we can validate our models with the following command substituting PATH by the path to the save model.

        python val.py --model_path PATH

### Citation

If you find this code useful please consider citing us:

        @article{hermosilla2022weightinit,
          title={Variance-Aware Weight Initialization for Point Convolutional Neural Networks},
          author={Hermosilla, Pedro and Schelling, Michael and Ritschel, Tobias and Ropinski, Timo},
          journal={European Conference on Computer Vision (ECCV)},
          year={2022}
        }
