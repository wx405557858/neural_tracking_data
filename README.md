# Marker Tracking with Neural Networks

## Getting Started

<img src="https://github.com/wx405557858/neural_tracking/blob/media/imgs/output_example.gif" align="right" width=384>

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

**Marker tracking for GelSight video**

```
python example_tracking_video.py
```

**Note**: The model is originally implemented with TensorFlow to be compatible with [Coral](https://coral.ai/products/accelerator), using TPU as USB Accelerator for Raspberry Pi on-device computation. Please feel free to switch the model to other frameworks, like PyTorch, for your purpose.



## Examples
 
* `python example_tracking_sim.py`:

* `python example_tracking_sim_generic.py`:

* `python example_tracking_video.py`:

<img src="https://github.com/wx405557858/neural_tracking/blob/media/imgs/output_example.gif" width=384>

## Train

* `python train.py`

* `python train_generic.py`



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