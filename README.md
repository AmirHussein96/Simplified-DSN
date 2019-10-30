# Simplified-DSN

This is a simplified implementation of the popular "[Domain Separation Neural Network](https://arxiv.org/abs/1608.06019) in Tensorflow. This work is just a humble attempt to simplify the official implementation see "https://github.com/tensorflow/models/tree/master/research/domain_adaptation". 

Tested with TensorFlow=1.14.0 and Python 3.6.

## MNIST Experiments

The `DSN.ipynb` notebook implements the MNIST experiments mentioned in the paper. The domain classifier with reversal layer is used here as a domain discrepancy loss since it was reported to be better than $MMD^2$. 

## Running the Experiment

Build MNIST-M dataset: MNIST-M dataset consists of MNIST digits blended with random color patches from the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) dataset. To generate a MNIST-M dataset, first download the BSDS500 dataset and run the `create_mnistm.py` script:
```bash
curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
python create_mnistm.py
```


