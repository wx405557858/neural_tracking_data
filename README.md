# Marker Tracking with Neural Networks

## Getting Started

**System**: Ubuntu, OSX

**Install packages**

```
pip install tensorflow-gpu opencv-python
```

**Download models**

```
bash ./models/download_model.sh
```

**Download test GelSight video**

```
bash ./data/download_video.sh
```

**Inference**

```
python example_tracking_video.py
```

<!--
<img src='https://drive.google.com/file/d/1DN3zOfz4xx4wgTNtGNRruHs9mUEewjer/view?usp=sharing' align="right" width=384>-->

**Note**: The model is originally implemented with TensorFlow to be compatible with [Coral](https://coral.ai/products/accelerator), using TPU as USB Accelerator for Raspberry Pi on-device computation. Please feel free to switch the model to other frameworks, like PyTorch, for your purpose.



## Examples

* `python example_tracking_sim.py`

* `python example_tracking_video.py`


## Datasets

`python generate_data.py`


## Train

`python train.py -p test`



## Citation
If you use this code for your research, please cite our paper: [Gelsight Wedge: Measuring High-Resolution 3D Contact Geometry with a Compact Robot Finger](https://arxiv.org/pdf/2106.08851.pdf):

```
@inproceedings{wang2021gelsight,
  title={Gelsight wedge: Measuring high-resolution 3d contact geometry with a compact robot finger},
  author={Wang, Shaoxiong and She, Yu and Romero, Branden and Adelson, Edward},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6468--6475},
  year={2021},
  organization={IEEE}
}
```