# Semi-blind Large Kernel Spatially Varying Deblurring

## Dataset

The datasets are stored in HuggingFace and they can be accessed @harshana95.
I have used a synthetic dataset for all training scripts and this dataset can be updated 
by changing the ```datasets.train.name``` parameter on the config file (```.yml```)


## Usage

### Initialization

* Update ```.comet.config``` file with your API key
* Download any pretrained models to ```checkpoints``` folder if required

### Baseline 1: [Learning degradation](https://github.com/dasongli1/Learning_degradation)
To run the training scripts of the existing code from the repo with
our dataset, use the following python scripts.

  * ```python train_basicsr.py```
  * ```python train_basicsr_v2.py```

To train the same model from  the repo using the code adapted to i2lab use the following python script.

  * ```python train_i2lab_basicsr.py```


### PSF generation

You can use ```generate_psfs_synthetic.py``` and ```generate_dataset_synthetic.py``` to generate your own dataset.