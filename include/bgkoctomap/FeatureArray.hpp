#include <iostream>
#include <Eigen/Dense>

namespace la3dm {
  class FeatureArray {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    FeatureArray(int num_channel, bool need_normalizing=false):
      dimension (num_channel),
      need_normalizing(need_normalizing){
      
      feature.resize(num_channel);
      feature.setZero();
      for (int i = 0; i < num_channel; i++)
        feature(i) = 1.0 / num_channel;
      if (need_normalizing == false)
        feature = (feature * 255.0).eval();
      //feature.colwise() += 1.0 / num_channel ;
      counter = 1;

    };

    FeatureArray(const FeatureArray & other):
      dimension(other.dimension),
      counter(other.counter),
      need_normalizing(other.need_normalizing)
    {
      feature  = other.feature;
      
    }

    FeatureArray &operator=(const FeatureArray &other) {
      dimension = other.dimension;
      counter = other.counter;
      need_normalizing = other.need_normalizing;
      feature = other.feature;
      return *this;
    }

    
    int add_observation_by_averaging(const Eigen::VectorXf & f) {
      if (f.size() == feature.size()){
        Eigen::VectorXf to_add = f;
        if (need_normalizing && f.sum() != 0)
          to_add = (to_add / f.sum()).eval();
        feature = ((feature * counter + f)/(counter+1)).eval();
        counter++;
        if (need_normalizing && feature.sum() != 0)
          feature = (feature / feature.sum()).eval();
        return 0;
      } else {
        return -1;
      }
      
    }

    int get_counter() const {return counter;}

    int overwrite_feature(const Eigen::VectorXf & f) {
      if (f.size() != feature.size())
        return -1;

      feature = f;
      if (need_normalizing && feature.sum()!= 0 )
        feature = (feature / feature.sum()) .eval();
      return 0;
    }

    int dimension;
    Eigen::VectorXf get_feature()  const {return feature;}
    
  private:
    Eigen::VectorXf feature;
    int counter;
    bool need_normalizing;
  };
  
}
