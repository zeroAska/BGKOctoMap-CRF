#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <pcl/filters/voxel_grid.h>
#include "bgkoctomap.h"
#include "bgkoctree_node.h"
#include "bgkinference.h"
#include "densecrf.h"
using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>


#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace la3dm {

    BGKOctoMap::BGKOctoMap() : BGKOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        1.0, // sf2
                                        1.0, // ell
                                        0.3f, // free_thresh
                                        0.7f, // occupied_thresh
                                        1.0f, // var_thresh
                                        1.0f, // prior_A
                                        1.0f // prior_B
                                    ) { }

    BGKOctoMap::BGKOctoMap(float resolution,
                        unsigned short block_depth,
                        float sf2,
                        float ell,
                        float free_thresh,
                        float occupied_thresh,
                        float var_thresh,
                        float prior_A,
                        float prior_B)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);

        OcTree::max_depth = block_depth;

        OcTreeNode::sf2 = sf2;
        OcTreeNode::ell = ell;
        OcTreeNode::free_thresh = free_thresh;
        OcTreeNode::occupied_thresh = occupied_thresh;
        OcTreeNode::var_thresh = var_thresh;
        OcTreeNode::prior_A = prior_A;
        OcTreeNode::prior_B = prior_B;
    }

    BGKOctoMap::~BGKOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void BGKOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        OcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKOctoMap::insert_training_data(const GPPointCloud &xy) {
        if (xy.empty())
            return;

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, BGK3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            }
            BGK3f *bgk = new BGK3f(OcTreeNode::sf2, OcTreeNode::ell);
            bgk->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }
#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            Block *block;
#ifdef OPENMP
#pragma omp critical
#endif
            {
                block = search(key);
                if (block == nullptr)
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk == bgk_arr.end())
                    continue;

                vector<float> m, var;
                bgk->second->predict(xs, m, var);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    node.update(m[j], var[j]);
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif
        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            auto block = block_arr.find(key);
            if (block == block_arr.end())
                continue;
            block->second->prune();
        }
#ifdef DEBUG
        Debug_Msg("Pruning done");
#endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void BGKOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, BGK3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            }
            BGK3f *bgk = new BGK3f(OcTreeNode::sf2, OcTreeNode::ell);
            bgk->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }
#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk == bgk_arr.end())
                    continue;

                vector<float> ybar, kbar;
                bgk->second->predict(xs, ybar, kbar);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    auto node_loc = block->get_loc(leaf_it);
                    if (node_loc.x() == 7.45 && node_loc.y() == 10.15 && node_loc.z() == 1.15) {
                        std::cout << "updating the node " << ybar[j] << " " << kbar[j] << std::endl;
                    }

                    // Only need to update if kernel density total kernel density est > 0
                    if (kbar[j] > 0.0)
                        node.update(ybar[j], kbar[j]);
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif
        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        /////////////////////////////////////////////////
// #ifdef OPENMP
// #pragma omp parallel for
// #endif
//         for (int i = 0; i < test_blocks.size(); ++i) {
//             BlockHashKey key = test_blocks[i];
//             auto block = block_arr.find(key);
//             if (block == block_arr.end())
//                 continue;
//             block->second->prune();
//         }
// #ifdef DEBUG
//         Debug_Msg("Pruning done");
// #endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }
  void BGKOctoMap::insert_semantic_pointcloud(const PCLSemanticPointCloud & cloud,
                                              const std::vector<SuperPixel *> & super_pixels_2d,
                                              const std::unordered_map<int, point3f> & uv1d_to_map3d,
                                              const point3f &origin, float ds_resolution,
                                              float free_res,
                                              float max_range) {
    pcl::PointCloud<pcl::PointXYZ> cloud_xyz;
    pcl::PointSeg_to_PointXYZ<NUM_CLASSES>(cloud, cloud_xyz);
    insert_pointcloud(cloud_xyz, origin, ds_resolution, free_res, max_range);
    for (int i = 0; i < cloud.size(); i++) {
      auto &p = cloud[i];
      Block * b = search(block_to_hash_key(point3f(p.x, p.y, p.z)));
      if (b == nullptr) continue;
      Eigen::Vector3f rgb;
      rgb << (float)p.r, (float)(p.g), (float)(p.b);
      Eigen::VectorXf semantic = Eigen::Map<const Eigen::VectorXf> (p.label_distribution, NUM_CLASSES);
      b->update_color_semantics(p.x, p.y, p.z, rgb, semantic);

      if (i == 0) {
           std::cout<<("[insertSemanticPointCloud] point 0:  insert semantic ")<<semantic.transpose()<<std::endl<<"after update node, the semantic becomes " << std::endl ;
           b->print_node_color_semantics(p.x,p.y,p.z);
      }
      
    }

    bool is_crf =true;
    if (is_crf)
      dense_crf(super_pixels_2d, uv1d_to_map3d );
  }

  Eigen::MatrixXf BGKOctoMap::high_order_crf(DenseCRF3D & crf_grid_3d,
                                  const std::vector<SuperPixel *> & super_pixels_2d ,
                                  const std::unordered_map<int, point3f> & uv1d_to_map3d,
                                  std::unordered_map<Occupancy *, int> & node_to_crf_ind,
                                  Eigen::Ref<Eigen::MatrixXf> unary_mat,
                                  Eigen::Ref<Eigen::MatrixXf> rgb_mat,
                                  Eigen::Ref<Eigen::MatrixXf> pose_mat
                                  ) {

    std::vector<std::shared_ptr<SuperPixel> > grid_3d_superpixels;
    std::unordered_map< std::shared_ptr<SuperPixel>, std::vector<Occupancy *> > sp3d_to_attached_nodes;
    for (int i = 0; i < super_pixels_2d.size(); i++) {
      std::vector<Occupancy *> sp_crf_grid_inds;
      int sp_crf_grid_num = 0;
      for (int j=0; j<super_pixels_2d[i]->pixel_indexes.size();j++){
        //int ux= super_pixels_2d[i]->pixel_indexes[j] % im_width;  // pixel coordinate in one superpixel, horizontal
        //int uy= super_pixels_2d[i]->pixel_indexes[j] / im_width;  // vertical
			  
        //if ( ((ux>im_width-1) || (ux<0)) || ((uy>im_height-1) || (uy<0)) ){
        //  std::cout<<"pixel index out of image "<<super_pixels_2d[i]->pixel_indexes[j]<<std::endl;
        //  break;
        /// }

        if (uv1d_to_map3d.find( super_pixels_2d[i]->pixel_indexes[j] )== uv1d_to_map3d.end()  )
          continue;
        auto p = uv1d_to_map3d.find( super_pixels_2d[i]->pixel_indexes[j] )->second;
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) 
          continue;
        OcTreeNode & node =  block->search(p) ;
        if (node.get_state() == State::FREE ) continue;
        sp_crf_grid_num++;
        sp_crf_grid_inds.push_back(&node);
        
      }
      if (sp_crf_grid_num>1){
        std::shared_ptr<SuperPixel> sp_3d( new SuperPixel(sp_crf_grid_num) );
        for (int k=0;k<sp_crf_grid_num;k++){
          sp_3d->current_ind=k;
          //sp_3d->pixel_indexes[k]=sp_crf_grid_inds[k];  // 3d superpixel's indexes stores CRF node index
          if (sp3d_to_attached_nodes.find(sp_3d) == sp3d_to_attached_nodes.end()  ) {
            std::vector< Occupancy*> v;
            sp3d_to_attached_nodes[sp_3d] = v;
          }
          sp3d_to_attached_nodes[sp_3d].push_back(sp_crf_grid_inds[k]);
        }
        grid_3d_superpixels.push_back(sp_3d);
      }
      
    }

    std::cout<<"grid_3d_superpixel size iis "<<grid_3d_superpixels.size()<<std::endl;
    int total_sp_grid_num=0;
    for (int i=0;i<grid_3d_superpixels.size();i++)
      total_sp_grid_num += grid_3d_superpixels[i]->pixel_indexes.size();

    bool use_hierachical = true;
    if (use_hierachical) {
      crf_grid_3d.hierarchical_high_order = true;
      Eigen::MatrixXf unary_mat_sp(NUM_CLASSES , grid_3d_superpixels.size());
      Eigen::MatrixXf rgb_mat_sp(3,   grid_3d_superpixels.size());
      MatrixXf pose_mat_sp(3,  grid_3d_superpixels.size());
      for (int i=0;i<grid_3d_superpixels.size();i++)
      {
        Vector_XXf unary_temp_sum=Vector_XXf::Zero();
        Vector3f pose_temp_sum=Vector3f::Zero();
        Vector3f rgb_temp_sum=Vector3f::Zero();
        int super_size = sp3d_to_attached_nodes[grid_3d_superpixels[i]].size();
        //int super_size=grid_3d_superpixels[i]->pixel_indexes.size();
        for (auto  n  : sp3d_to_attached_nodes[grid_3d_superpixels[i]]  )
        {
          int px_ind = node_to_crf_ind[n ];
          unary_temp_sum += unary_mat.col(px_ind);
          pose_temp_sum += pose_mat.col(px_ind);
          rgb_temp_sum += rgb_mat.col(px_ind);
        }
        unary_mat_sp.col(i)= unary_temp_sum / (float) super_size;
        pose_mat_sp.col(i)= pose_temp_sum / (float) super_size;
        rgb_mat_sp.col(i)= rgb_temp_sum / (float) super_size;
        if (i == 0) {
          std::cout<<"crf sp input : unary(:,0) \n"<<unary_mat_sp.col(i)
                   <<"\npoes_mat_sp(:, 0) \n"<<pose_mat_sp.col(i)
                   <<"\nrgb_mat_sp(:,0) \n"<<rgb_mat_sp.col(i)
                   <<std::endl;
          
        }
      }
      // reason superpixel's and grid's label together
      // the 3 and 4 th pairwise energy is superpixel's.   smooth_xy_stddev   appear_xy_stddev
      crf_grid_3d.setUnaryEnergy2(unary_mat_sp);
      //      auto pottsG(std::shared_ptr<PottsCompatibility>(new PottsCompatibility(3)) );
      //auto pottsB(std::shared_ptr<PottsCompatibility>(new PottsCompatibility(8)) );
      crf_grid_3d.addPairwiseGaussian( 0.1, 0.1, 0.1, pose_mat_sp,   //smooth_xy_stddev
                                       new PottsCompatibility(3));
      crf_grid_3d.addPairwiseBilateral( 0.1, 0.1, 0.1, 8,8,8,
                                        pose_mat_sp,rgb_mat_sp, new PottsCompatibility(8));

      std::vector<SuperPixel *> all_3d_superpixels;
      for (int j = 0; j < grid_3d_superpixels.size(); j++) {
        all_3d_superpixels.push_back(grid_3d_superpixels[j].get());
      }
      
      crf_grid_3d.all_3d_superpixels_=all_3d_superpixels;

      Eigen::MatrixXf result = crf_grid_3d.inference(5 );
      return result;
    } else {
      Eigen::MatrixXf result = crf_grid_3d.inference(5 );
      return result;
      
    }
    
  }

  void BGKOctoMap::dense_crf(const std::vector<SuperPixel *> & super_pixels_2d,
                             const std::unordered_map<int, point3f> & uv1d_to_map3d
                             ) {
    int crf_grid_num_guess=0; //cloud.size();
    std::vector<Occupancy *> crf_ind_to_node;
    std::unordered_map<Occupancy * , int> node_to_crf_ind;
    std::vector<point3f> poses;
    for (auto leaf = begin_leaf(); leaf != end_leaf(); leaf++) {
      if (leaf.get_node().get_state() == State::OCCUPIED) {
        
        point3f p = leaf.get_loc();
        la3dm::Block * block = search( block_to_hash_key(p));
        //auto & node = it.get_node();
        OcTreeNode & node = block->search(p);
        //        if ( node.get_semantics().get_counter() > 1) {
          crf_ind_to_node.push_back(&node);
          node_to_crf_ind[&node] = crf_grid_num_guess ;
          poses.push_back(point3f(p.x(),p.y(),p.z()));
          crf_grid_num_guess++;
          //}
      }
    }

    std::cout<<"crf_grid_num_guess "<<crf_grid_num_guess<<std::endl;

    Eigen::MatrixXf unary_mat(NUM_CLASSES , crf_grid_num_guess);
    Eigen::MatrixXf rgb_mat(3, crf_grid_num_guess);
    Eigen::MatrixXf pose_mat(3,  crf_grid_num_guess);
    //std::unordered_map<int, int> memInd_To_crfInd, crfInd_To_memInd;

    int counter = 0;
    //    for (auto leaf = begin_leaf(); leaf!= end_leaf(); leaf++  ) {
    for (counter = 0 ; counter<crf_grid_num_guess; counter++  ) {
      Occupancy * node = crf_ind_to_node[counter];
      //if (node->get_state() != State::OCCUPIED) continue;
      //if (node->get_semantics().get_counter() < 2) continue;
      Eigen::VectorXf semantic = node->get_semantics().get_feature();

      semantic = (semantic / semantic.sum()).eval();
      unary_mat.col(counter) = -semantic.array().log() ;
      //if (counter == 0)

      rgb_mat.col(counter) = node->get_color().get_feature();
      //if (node.get_color().get_feature().sum() > 1) {
      //std::cout<<"unary mat before taking the log: "<<unary_mat.col(counter).transpose()<<std::endl;
      //std::cout<<"rgb mat is "<<node->get_color().get_feature().transpose()<<std::endl<<std::endl;
      //}
      auto p = poses[counter];
      pose_mat.col(counter) << p.x(), p.y(), p.z();
    } 

    // set up crf. The params are taken from Yang's code
    std::cout<<"Construct crf " <<", counter is "<<counter<<std::endl;;
    DenseCRF3D crf_grid_3d(crf_grid_num_guess , NUM_CLASSES );
    std::cout<<"Construct crf finish"<<std::endl;
    crf_grid_3d.setUnaryEnergy(unary_mat);
    //  auto pottsG(std::shared_ptr<PottsCompatibility>(new PottsCompatibility(3)) );
    //auto pottsB(std::shared_ptr<PottsCompatibility>(new PottsCompatibility(8)) );
    crf_grid_3d.addPairwiseGaussian( 0.2 , 0.2 , 0.2 , pose_mat ,
                                     new PottsCompatibility(3));
    crf_grid_3d.addPairwiseBilateral(0.2,0.2,0.2, 8,8,8, pose_mat, rgb_mat, new PottsCompatibility(8));
    Eigen::MatrixXf crf_grid_output_prob;

    bool use_high_order = true;
    int crf_iterations = 5; 
    if (use_high_order) {
      crf_grid_3d.set_ho(true );
      crf_grid_output_prob = high_order_crf(crf_grid_3d, super_pixels_2d, uv1d_to_map3d,
                                            node_to_crf_ind,
                                            unary_mat, rgb_mat, pose_mat
                                            );
    }  else{
      crf_grid_output_prob = crf_grid_3d.inference(crf_iterations);
    }

    //assert(crf_grid_output_prob.cols() == crf_grid_num_guess );
    std::cout<<"Num of cols in crf output is "<<crf_grid_output_prob.cols()<<", num of cols in unary mat is "<<unary_mat.cols()<<std::endl;
    for (int i = 0; i < unary_mat.cols(); i++) {
      auto &node = *crf_ind_to_node[i];
      auto new_label = crf_grid_output_prob.col(i);
      for (int label_id=0;label_id< new_label.rows();label_id++) {
        if ( new_label(label_id) < 1e-8) // don't want prob to be to small.
          new_label (label_id) = 1e-8;
      }
      if (i == 0) std::cout<<"before norm: crf output at i = 0 is "<<new_label.transpose()<<std::endl;
     new_label = (new_label / new_label.sum() ).eval();
      if (i == 0) std::cout<<"after norm: crf output at i = 0 is "<<new_label.transpose()<<std::endl;
      assert( std::abs(1.0- new_label.sum()) < 0.00001  );
      Eigen::VectorXf color_new;
      node.update(color_new, new_label, true);
    }
    std::cout<<"CRF finish\n";
  }

    void BGKOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPPointCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(it->second->get_center(), 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    void BGKOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {
        PCLPointCloud sampled_hits;
        downsample(cloud, sampled_hits, ds_resolution);

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        xy.clear();
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);
            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            xy.emplace_back(p, 1.0f);

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);

            frees.push_back(PCLPointType(origin.x(), origin.y(), origin.z()));
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
                frees.push_back(PCLPointType(p->x(), p->y(), p->z()));
                frees.width++;
            }
        }

        PCLPointCloud sampled_frees;    
        downsample(frees, sampled_frees, ds_resolution);

        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.emplace_back(point3f(it->x, it->y, it->z), 0.0f);
        }
    }

    void BGKOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
        if (ds_resolution < 0) {
            out = in;
            return;
        }

        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(out);
    }

    void BGKOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution) const {
        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = free_resolution;
        while (d < l) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d += free_resolution;
        }
        if (l > free_resolution)
            frees.emplace_back(x0 + nx * (l - free_resolution), y0 + ny * (l - free_resolution), z0 + nz * (l - free_resolution));
    }

    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void BGKOctoMap::bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x());
            y.push_back(it->first.y());
            z.push_back(it->first.z());
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void BGKOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int BGKOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         GPPointCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int BGKOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int BGKOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         GPPointCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKOctoMap::search_callback, static_cast<void *>(&out));
    }

    int BGKOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKOctoMap::count_callback, NULL);
    }

    bool BGKOctoMap::count_callback(GPPointType *p, void *arg) {
        return false;
    }

    bool BGKOctoMap::search_callback(GPPointType *p, void *arg) {
        GPPointCloud *out = static_cast<GPPointCloud *>(arg);
        out->push_back(*p);
        return true;
    }


    int BGKOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int BGKOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         GPPointCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *BGKOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    OcTreeNode BGKOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
            return OcTreeNode();
        } else {
            return OcTreeNode(block->search(p));
        }
    }

    OcTreeNode BGKOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
