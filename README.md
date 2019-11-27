# BGKOctoMap-CRF

This repository performs a CRF semantic voxel map on top of the continuous Bayesian Generalized Kernel Inference occcupancy map, built on top of [BGKOctoMap](https://github.com/RobustFieldAutonomyLab/la3dm), [densecrf](http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/)  and  [semantic_3d_mapping](https://github.com/shichaoy/semantic_3d_mapping). The implementation is intended for the replication of the methods on a few datasets. 


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

We provide ros launchfiles for kitti seq 05 and kitti seq 15. The format of the dataset follows [semantic_3d_mapping](https://github.com/shichaoy/semantic_3d_mapping). The zip files of kitti seq 15 can be downloaded [here](https://drive.google.com/file/d/1dIHRrsA7rZSRJ6M9Uz_75ZxcHHY96Gmb/view?usp=sharing). Put the files into `data/` folder, e.g. `data/data_kitti_15/`. The launch files is `launch/kitti_node.launch`. The config file is in `config/kitti_15.yaml`.

To run the demo on kitti 15:

```bash
$ roslaunch la3dm kitti_node.launch
```

which by default will run the full BGKOctoMap-CRF method. 

## Relevant Works and Publications

This repository serves as a baseline of the [BKIOctoMap](https://github.com/ganlumomo/BKISemanticMapping).  It uses code from [BGKOctoMap](https://github.com/RobustFieldAutonomyLab/la3dm), [densecrf](http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/)  and  [semantic_3d_mapping](https://github.com/shichaoy/semantic_3d_mapping).


