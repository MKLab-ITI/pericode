# SALIC

Social Active Learning for Image Classification

http://mklab.iti.gr/research/salic


Installation:

1. clone the repository

2. Download and compile LIBSVM for your architecture https://www.csie.ntu.edu.tw/~cjlin/libsvm/


3. Download and compile the ConvNet Feature Computation Package from http://www.robots.ox.ac.uk/~vgg/software/deep_eval/


4. Change the paths to the folders including the datasets in Wrapper.m, create the required files (img_Files.mat, tag_files.mat for each dataset) and run Wrapper.


Requirements:

1. There is only compatability for Linux (the ConvNet Feature Computation Package is not compatible with windows). If a different CNN feature extraction library is used that runs on Windows, the code should run on Windows as well (not tested)

2. For MIRFLICKR (1m images as the pool dataset), 64GB of RAM is minimum

3. The code was tested using Matlab 2012a


If you use this code cite the following paper:

Elisavet Chatzilari, Spiros Nikolopoulos, Yiannis Kompatsiaris, Josef Kittler, "SALIC: Social Active Learning for Image Classification", IEEE Transactions on Multimedia, 2016, accepted for publication

URL: http://dx.doi.org/10.1109/TMM.2016.2565440


