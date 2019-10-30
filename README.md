# Simplified-DSN

This is a simplified implementation of the popular "[Domain Separation Neural Network](https://arxiv.org/abs/1608.06019) in Tensorflow. This work is just a humble attempt to simplify the official implementation see "https://github.com/tensorflow/models/tree/master/research/domain_adaptation". All main functions including encoders and the shared decoder was borrowed from the DSN official implementation. Big part of this implementation was inspired by the Domain Adaptation Neural Network official implementatio see "https://github.com/pumpikano"

Tested with TensorFlow=1.14.0 and Python 3.6.
## Model
[https://github.com/AmirHussein96/Simplified-DSN/blob/master/images/DSN.png|alt=octocat]

## MNIST Experiments

The `DSN.ipynb` notebook implements the MNIST->MNISTM experiments mentioned in the paper. The domain classifier with reversal layer (DANN) is used here as a domain discrepancy loss since it was reported to be better than $MMD^2$. by For more information about DANN check "http://jmlr.org/papers/volume17/15-239/15-239.pdf"

## Running the Experiment

Build MNIST-M dataset: MNIST-M dataset consists of MNIST digits blended with random color patches from the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) dataset. To generate a MNIST-M dataset, first download the BSDS500 dataset and run the `create_mnistm.py` script:



