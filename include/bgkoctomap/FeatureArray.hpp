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
      counter = 0;

    };

    FeatureArray(const FeatureArray & other):
      dimension(other.dimension),
      counter(other.counter),
      need_normalizing(other.need_normalizing)
    {
      feature  = other.feature;
      
    }
    
    int add_observation_by_averaging(const Eigen::VectorXf & f) {
      if (f.size() == feature.size()){
        feature = ((feature * counter + f)/(counter+1)).eval();
        counter++;
        if (need_normalizing && feature.sum() != 0)
          feature = (feature / feature.sum()).eval();
        return 0;
      } else {
        return -1;
      }
      
    }

    int overwrite_feature(const Eigen::VectorXf & f) {
      if (f.size() != feature.size())
        return -1;

      feature = f;
      if (need_normalizing && feature.sum()!= 0 )
        feature = (feature / feature.sum()) .eval();
      return 0;
    }

    const int dimension;
    Eigen::VectorXf get_feature()  const {return feature;}
    
  private:
    Eigen::VectorXf feature;
    int counter;
    bool need_normalizing;
  };
  
}
