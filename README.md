# BGKOctoMap-CRF

This repository extends the continuous [Bayesian Generalized Kernel Inference occcupancy map](https://github.com/RobustFieldAutonomyLab/la3dm) to semantic mapping by performing an additinal a dense [Conditional Random Field](http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/) update on semantic voxels. This repo is a part of the code release from [BKIOctoMap](https://github.com/ganlumomo/BKISemanticMapping), serving as a baseline implementation on public datasets. It uses code from [BGKoctomap](https://github.com/RobustFieldAutonomyLab/la3dm) and [semantic_3d_mapping](https://github.com/shichaoy/semantic_3d_mapping).  

Visualization with `rviz` on Kitti seq15:
![Kitti15](https://github.com/zeroAska/BGKOctoMap-CRF/raw/crf/config/datasets/visualization_kitti_15.png)

Performance on Kitti as tested on the [BKIOctoMap](https://arxiv.org/abs/1909.04631) paper:
![Kitti Test Result](https://github.com/zeroAska/BGKOctoMap-CRF/raw/crf/config/datasets/kitti_seq15_seq05_result.png)


## Getting Started

### Dependencies

We tested BGKOctoMap with ROS Melodic. Dependencies include:
```
ros-melodic-desktop-full
octomap_ros
openmp
```


### Building with catkin

The repository is set up to work with catkin, so to get started you can clone the repository into your catkin workspace `src` folder and compile with `catkin_make`:

```bash
my_catkin_workspace/src$ git clone https://github.com/zeroAska/BGKOctoMap-CRF.git
my_catkin_workspace/src$ cd BGKOctoMap-CRF
my_catkin_workspace/src/BGKOctoMap-CRF$ mv dense_crf ../
my_catkin_workspace/src/BGKOctoMap-CRF$ cd ../../
my_catkin_workspace$ source devel/setup.bash
my_catkin_workspace$ catkin_make
my_catkin_workspace$ source devel/setup.bash
```

## Running the Demo

We provide ros launchfiles for kitti seq 05 and kitti seq 15. The format of the dataset follows [semantic_3d_mapping](https://github.com/shichaoy/semantic_3d_mapping). The zip files of kitti seq 15 can be downloaded [here](https://drive.google.com/file/d/1dIHRrsA7rZSRJ6M9Uz_75ZxcHHY96Gmb/view?usp=sharing). Put the files into `data/` folder, e.g. `data/data_kitti_15/`. The launch files is `launch/kitti_node.launch`. The config file is in `config/datasets/kitti_15.yaml`. 

If you want to turn on the visualization with `rviz`, set the `visualize` flag in the last line of `config/datasets/kitti_15.yaml` to true. This will increase the memory usage and reduce running speed.

To run the demo on kitti 15:

```bash
$ roslaunch la3dm kitti_node.launch
```

which by default will run the full BGKOctoMap-CRF method. 

## Relevant Works and Publications

This repository serves as a baseline of the [BKIOctoMap](https://github.com/ganlumomo/BKISemanticMapping).  It uses code from [BGKOctoMap](https://github.com/RobustFieldAutonomyLab/la3dm), [densecrf](http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/)  and  [semantic_3d_mapping](https://github.com/shichaoy/semantic_3d_mapping). If you find this repo useful, please cite:
```
@ARTICLE{gan2019bayesian,
author={L. {Gan} and R. {Zhang} and J. W. {Grizzle} and R. M. {Eustice} and M. {Ghaffari}},
journal={IEEE Robotics and Automation Letters},
title={Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping},
year={2020},
volume={5},
number={2},
pages={790-797},
keywords={Mapping;semantic scene understanding;range sensing;RGB-D perception},
doi={10.1109/LRA.2020.2965390},
ISSN={2377-3774},
month={April},}
```


