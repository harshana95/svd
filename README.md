# Semi-blind Large Kernel Spatially Varying Deblurring

## Dataset

The datasets are stored in HuggingFace and they can be accessed @harshana95.
I have used a synthetic dataset for all training scripts and this dataset can be updated 
by changing the ```datasets.train.name``` parameter on the config file (```.yml```)


## Usage

### Initialization

* Update ```.comet.config``` file with your API key
* Download any pretrained models to ```checkpoints``` folder if required
* Use ```accelerate``` instead of ```python``` to launch the scripts if your're using more than 1 GPU

### Baseline 1: [Learning degradation](https://github.com/dasongli1/Learning_degradation)
To run the training scripts of the existing code from the repo with
our dataset, use the following python scripts.

  * ```python train_basicsr.py```
  * ```python train_basicsr_v2.py```

To train the same model from  the repo using the code adapted to i2lab use the following python script.

  * ```python train_i2lab_basicsr.py```

### Baseline 2: [NAFNet](https://github.com/megvii-research/NAFNet)

### Baseline 3: [Restormer](https://github.com/swz30/Restormer)

### PSF generation

You can use ```generate_psfs_synthetic.py``` and ```generate_dataset_synthetic.py``` to generate your own dataset.
