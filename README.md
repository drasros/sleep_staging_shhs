# Installation

The main dependencies are tensorflow/tensorflow-gpu, numpy, scipy, scikit-learn, matplotlib, pyedflib, tqdm. 

The recommended way to install them is to use conda and import the attached environment:
```
conda env create -f=environment.yml
```

# Obtaining data

Please follow instructions from the [SHHS website](https://sleepdata.org/datasets/shhs). 

# Train

* Modify CONFIG settings in train.py

* Run train.py and specify your options. To reproduce the model described in the paper, use:

```
python train.py -training_batches 300000 -learning_rates 0.00003 -featuremap_sizes 128 128 128 128 128 128 256 256 256 256 256 256 -filter_sizes 7 7 7 7 7 7 7 5 5 5 3 3 -strides 2 2 2 2 2 2 2 2 2 2 2 2 --eps_before=2 --balance_classes=False --conv_type=std --batch_norm=False --filter=False --hiddenlayer_size=256
```

# Visualize

To visualize synthetic inputs as described in the paper:

* Modify CONFIG settings in visualize.py

* Run:
```
python visualize.py
```
