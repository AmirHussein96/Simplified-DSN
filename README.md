# Simplified-DSN
#### This work was officialy added to the TensorFlow GitHub page under Other implementations  
https://github.com/tensorflow/models/blob/master/research/domain_adaptation/README.md#other-implementations. 

This is a simplified implementation of the popular "[Domain Separation Neural Network](https://arxiv.org/abs/1608.06019) using Tensorflow. This work is just a humble attempt to simplify the official implementation see "https://github.com/tensorflow/models/tree/master/research/domain_adaptation", since there were several issues reported on the github and I faced them as well. After all it is our duty as engineers and future scientists to make people life easier and better, don't you think so? XD. 
<br/>
The main code parts including loss functions, encoders and the shared decoder were borrowed from the DSN official implementation. Big part of this implementation was inspired by the Domain Adaptation Neural Network official implementatio see "https://github.com/pumpikano". 


## Requirements
Tested with TensorFlow=1.14.0 and Python 3.6.


## DSN_Model
![Alt text](images/DSN.png?raw=true "DSN Model")


## MNIST Experiments

The `DSN.ipynb` notebook implements the MNIST->MNISTM experiments mentioned in the paper. The domain classifier with reversal layer (DANN) is used here as a domain discrepancy loss since it was reported to be better than $MMD^2$. by For more information about DANN check "http://jmlr.org/papers/volume17/15-239/15-239.pdf"

## Running the Experiment

Build MNIST-M dataset: MNIST-M dataset consists of MNIST digits blended with random color patches from the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) dataset. To generate a MNIST-M dataset, first download the BSDS500 dataset and run the `create_mnistm.py` script:

## Results

| Method | Target acc (paper) | Target acc (this repo w/ 30 epochs) |
| ------ | ------------------ | ----------------------------------- |
| DSN |  0.832 |  0.821 |

### Feature Maps Visualization

![Alt text](images/DNS_MNIST_MNISTM.png?raw=true "Domain_Adaptation")

### Decoder Reconstructed Images
![Alt text](images/source.png?raw=true )
![Alt text](images/reconstracted_source.png?raw=true )
<br/>
![Alt text](images/target.png?raw=true )
![Alt text](images/reconstracted_target.png?raw=true )

## Contribution

It would be great to add other experiments to this repository. Feel free to make a PR if you decide to recreate other results from the papers or new experiments entirely.

## Contact
In case you faced any issues or found any bugs feel free to contact me at "anh21@mail.aub.edu"
