## Experiments of the paper *Variance-Aware Weight Initialization for Point Convolutional Neural Networks*

This is the code of the experiments presented in the paper: *Variance-Aware Weight Initialization for Point Convolutional Neural Networks*.

**_NOTE:_**  This repository only contains the code of the experiments. The CUDA implementation of the different convolution operators and the point cloud library in Pytorch needs to be installed separately from the following [repository](https://github.com/phermosilla/pccnn_lib). See the *Required libraries* section in this file.

### Required libraries

This code was tested in python 3.7.
In order to install the required python libraries use pip with the following command:

        pip install -r requirements.txt

##### PCCNN Library

The library to process point cloud in Pytorch and the convolution operators are implemented in the following [repository](https://github.com/phermosilla/pccnn_lib).
Install this library first in order to run the experiments.

### Data Download

##### ScanObjectNN

This data set can be downloaded [here](https://hkust-vgd.github.io/scanobjectnn/).
Once downloaded, uncompress the file in the *ScanObjectNN* folder and rename the folder to *data*.

##### ScanNet

This data set can be downloaded [here](http://www.scan-net.org/).

##### ModelNet40

### Citation

If you find this code useful please consider citing us:

        @article{hermosilla2022weightinit,
          title={Variance-Aware Weight Initialization for Point Convolutional Neural Networks},
          author={Hermosilla, Pedro and Schelling, Michael and Ritschel, Tobias and Ropinski, Timo},
          journal={European Conference on Computer Vision (ECCV)},
          year={2022}
        }
