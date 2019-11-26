#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "bgkoctomap.h"
#include "markerarray_pub.h"
#include "kitti_util.h"
#include "PointSegmentedDistribution.hpp"


int main(int argc, char **argv) {

  typedef pcl::PointCloud<pcl::PointXYZRGB> PCLPointCloudRGB;
  
    ros::init(argc, argv, "kitti_node");
    ros::NodeHandle nh("~");
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2 > ("new_points", 1);
    std::string map_topic("/occupied_cells_vis_array");
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    float prior = 1.0f;
    float var_thresh = 1.0f;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    double resolution = 0.1;
    int num_class = 2;
    double free_resolution = 0.5;
    double ds_resolution = 0.1;
    int scan_num = 0;
    double max_range = -1;
    
    // KITTI 05
    std::string dir;
    std::string left_img_prefix;
    std::string depth_img_prefix;
    std::string label_bin_prefix;
    std::string camera_pose_file;
    std::string evaluation_list_file;
    std::string reproj_img_prefix;
    std::string sp_bin_prefix;
    int image_width = 1226;
    int image_height = 370;
    float focal_x = 707.0912;
    float focal_y = 707.0912;
    float center_x = 601.8873;
    float center_y = 183.1104;
    float depth_scaling = 2000;
    int scan_start = 0;
    bool reproject = false;
    bool visualize = false;

    nh.param<int>("block_depth", block_depth, block_depth);
    nh.param<double>("sf2", sf2, sf2);
    nh.param<double>("ell", ell, ell);
    nh.param<float>("prior", prior, prior);
    nh.param<float>("var_thresh", var_thresh, var_thresh);
    nh.param<double>("free_thresh", free_thresh, free_thresh);
    nh.param<double>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<double>("resolution", resolution, resolution);
    nh.param<int>("num_class", num_class, num_class);
    nh.param<double>("free_resolution", free_resolution, free_resolution);
    nh.param<double>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<int>("scan_start",scan_start, scan_start);
    nh.param<int>("scan_num", scan_num, scan_num);
    nh.param<double>("max_range", max_range, max_range);

    // KITTI
    nh.param<std::string>("dir", dir, dir);
    nh.param<std::string>("left_img_prefix", left_img_prefix, left_img_prefix);
    nh.param<std::string>("depth_img_prefix", depth_img_prefix, depth_img_prefix);
    nh.param<std::string>("label_bin_prefix", label_bin_prefix, label_bin_prefix);
    nh.param<std::string>("superpixel_bin_prefix", sp_bin_prefix, sp_bin_prefix);
    nh.param<std::string>("camera_pose_file", camera_pose_file, camera_pose_file);
    nh.param<std::string>("evaluation_list_file", evaluation_list_file, evaluation_list_file);
    nh.param<std::string>("reproj_img_prefix", reproj_img_prefix, reproj_img_prefix);
    nh.param<int>("image_width", image_width, image_width);
    nh.param<int>("image_height", image_height, image_height);
    nh.param<float>("focal_x", focal_x, focal_x);
    nh.param<float>("focal_y", focal_y, focal_y);
    nh.param<float>("center_x", center_x, center_x);
    nh.param<float>("center_y", center_y, center_y);
    nh.param<float>("depth_scaling", depth_scaling, depth_scaling);
    nh.param<bool>("reproject", reproject, reproject);
    nh.param<bool>("visualize", visualize, visualize);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
      "block_depth: " << block_depth << std::endl <<
      "sf2: " << sf2 << std::endl <<
      "ell: " << ell << std::endl <<
      "prior: " << prior << std::endl <<
      "var_thresh: " << var_thresh << std::endl <<
      "free_thresh: " << free_thresh << std::endl <<
      "occupied_thresh: " << occupied_thresh << std::endl <<
      "resolution: " << resolution << std::endl <<
      "num_class: " << num_class << std::endl <<
      "free_resolution: " << free_resolution << std::endl <<
      "ds_resolution: " << ds_resolution << std::endl <<
      "scan_sum: " << scan_num << std::endl <<
      "max_range: " << max_range << std::endl <<

      "KITTI:" << std::endl <<
      "dir: " << dir << std::endl <<
      "left_img_prefix: " << left_img_prefix << std::endl <<
      "depth_img_prefix: " << depth_img_prefix << std::endl <<
      "label_bin_prefix: " << label_bin_prefix << std::endl <<
      "camera_pose_file: " << camera_pose_file << std::endl <<
      "evaluation_list_file: " << evaluation_list_file << std::endl <<
      "reproj_img_prefix: " << reproj_img_prefix << std::endl <<
      "image_width: " << image_width << std::endl <<
      "image_height: " << image_height << std::endl <<
      "focal_x: " << focal_x << std::endl <<
      "focal_y: " << focal_y << std::endl <<
      "center_x: " << center_x << std::endl <<
      "center_y: " << center_y << std::endl <<
      "depth_scaling: " << depth_scaling << std::endl <<
      "reproject: " << reproject << std::endl <<
      "visualize" << visualize
      );

    num_class = NUM_CLASSES;
    KITTIData kitti_data(image_width, image_height, focal_x, focal_y, center_x, center_y, depth_scaling, num_class);
    std::string camera_pose_name(dir + "/" + camera_pose_file);
    if (!kitti_data.read_camera_poses(camera_pose_name))
      return 0;
    if (reproject) {
      std::string evaluation_list_name(dir + "/" + evaluation_list_file);
      if (!kitti_data.read_evaluation_list(evaluation_list_name))
        return 0;
      std::string reproj_img_folder(dir + "/" + reproj_img_prefix + "/");
      kitti_data.set_up_reprojection(reproj_img_folder);
    }

    ///////// Build Map /////////////////////
    la3dm::BGKOctoMap map(resolution, block_depth, num_class, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
    la3dm::MarkerArrayPub m_pub(nh, map_topic, 0.1f);
    ros::Time start = ros::Time::now();
    for (int scan_id = scan_start; scan_id <= scan_num; ++scan_id) {
      //la3dm::PCLPointCloud cloud;
      pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASSES>> cloud;
      la3dm::point3f origin;
        
      char scan_id_c[256];
      sprintf(scan_id_c, "%06d", scan_id);
      std::string scan_id_s(scan_id_c);
      std::string color_img_name(dir+"/" + left_img_prefix + "/"+scan_id_s + ".png");
      std::string depth_img_name(dir + "/" + depth_img_prefix + "/" + scan_id_s + ".png");
      std::string label_bin_name(dir + "/" + label_bin_prefix + "/" + scan_id_s + ".bin");
      std::string sp_bin_name(dir + "/" + sp_bin_prefix + "/" + scan_id_s + ".bin");

      cv::Mat color_img = cv::imread(color_img_name, cv::IMREAD_COLOR);
      cv::Mat depth_img = cv::imread(depth_img_name, cv::IMREAD_ANYDEPTH );
      kitti_data.read_label_prob_bin(label_bin_name);
      std::unordered_map<int, la3dm::point3f> uv1d_to_map3d;
      kitti_data.process_depth_img(scan_id, color_img, depth_img, cloud, uv1d_to_map3d, origin, reproject);

      std::vector<SuperPixel *> sp;
      kitti_data.read_superpixel(sp_bin_name, sp);
      
      map.insert_semantic_pointcloud(cloud, sp, uv1d_to_map3d, origin, resolution, free_resolution, max_range);
      ROS_INFO_STREAM("Scan " << scan_id << " done");
     
      if (reproject)
        kitti_data.reproject_imgs(scan_id, map);

      if (visualize) {
        m_pub.clear_map(resolution);
        int counter = 0;
        int total = 0;

        sensor_msgs::PointCloud2 msg;
        pcl::PointCloud<pcl::PointXYZRGB> rgb;
        PointSeg_to_PointXYZRGB(cloud, rgb);
        pcl::toROSMsg(rgb, msg );
        msg.header.frame_id = "/map";
        msg.header.stamp = ros::Time::now();
        pub.publish(msg);
        

        int unknows = 0;
        int frees = 0;
        int occupies = 0;
        int pruned = 0;
        for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
          total++;
          if (it.get_node().get_state() == la3dm::State::OCCUPIED) occupies ++;
          if (it.get_node().get_state() == la3dm::State::UNKNOWN ) unknows ++;
          if (it.get_node().get_state() == la3dm::State::FREE) frees ++;
          if (it.get_node().get_state() == la3dm::State::PRUNED ) pruned ++;
          
          
          if (it.get_node().get_state() != la3dm::State::FREE ) {
            la3dm::point3f p = it.get_loc();
            la3dm::Block * block = map.search( la3dm::block_to_hash_key(p));
            //auto & node = it.get_node();
            auto & node = block->search(p);
            //if (node.get_semantics().get_counter() < 2 )
            //  continue;
            m_pub.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), node.get_label(), 1);
            //if (counter == 0) {
            //std::cout<<" the first point at "<<p.x()<<", "<<p.y()<<", "<<p.z()<<" has label "<<node.get_label()<<" with distribtujion "<<node.get_semantics().get_feature().transpose()<<std::endl;

              //}

            counter++;
          }
        }
       
        static bool query_center = false;
        if (!query_center) {
          la3dm::point3f p (3.05, 20.25, 0.25);
          la3dm::Block * block = map.search( la3dm::block_to_hash_key(p));
          if (block ) {
            //auto & node = it.get_node();
            auto & q_center = block->search(p);

            printf(" query point at 3.05, 20.25, 0.25: label is %d, ", q_center.get_label());
            print_state(q_center.get_state());
            
          }
        }
          
       

        printf("occupies: %d, frees: %d, pruned: %d, unknowns: %d\n", occupies, frees, pruned, unknows);
        
        std::cout<<"Total number of cells "<<total<<", useufl "<<counter<<std::endl;
        m_pub.publish();
      }
    }
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("Mapping finished in " << (end - start).toSec() << "s");
        
    //ros::spin();
    return 0;
}
