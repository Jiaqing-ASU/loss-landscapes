# Visualizations for Loss

## Install Instructions

This part of the code has been edited and modified from the original [loss-landscapes](https://github.com/marcellodebernardi/loss-landscapes), so you need to manually install the Python package to successfully run the tests in this project.

```shell
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m build
cd dist
pip install <tar.gz file>
```

## Running Instructions

In order to compare the different effects of different corruption inputs, we divide the training and visualization of the model into two parts. First, we train the model and save it in the current folder.

```shell
python3 train_mnist.py
```

Then we can see the `model_initial.pt` and `model_final.pt` have been generated under the current folder.

To do the visualization, run the following code:

```shell
python3 plot_mnist.py --dataset=<corruption dataset>
```

The corruption dataset options are `brightness`, `canny_edges`, `dotted_line`, `fog`, `glass_blur`, `identity`, `impulse_noise`, `motion_blur`, `rotate`, `scale`, `shear`, `shot_noise`, `spatter`, `stripe`, `translate`, `zigzag`. The default setting is `brightness`.