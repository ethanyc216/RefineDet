#include <algorithm>
#include <vector>

#include "caffe/layers/fsp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FSPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype * bottom_data_0 = bottom[0]->gpu_data();
  const Dtype * bottom_data_1 = bottom[1]->gpu_data();
  Dtype scale = 1.0 / (Dtype)(bottom[0]->height() * bottom[0]->width());
  
  for (int i = 0; i < bottom[0]->num(); ++i){
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), 
        bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),
        scale, bottom_data_0 + bottom[0]->offset(i), 
        bottom_data_1 + bottom[1]->offset(i), (Dtype)0.,
        top[0]->mutable_gpu_data() + top[0]->offset(i));
  }
}

template <typename Dtype>
void FSPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype scale = 1.0 / (Dtype)(bottom[0]->height() * bottom[0]->width());
  /**
   * bottom[0]' gradient = top_diff * B 
   */
  if (propagate_down[0]) {
    for(int i =0; i < bottom[0]->num(); ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
          bottom[0]->channels(), bottom[1]->height() * bottom[1]->width(),
          bottom[1]->channels(), scale, top[0]->gpu_diff() + top[0]->offset(i),
          bottom[1]->gpu_data() + bottom[1]->offset(i), (Dtype)0.,
          bottom[0]->mutable_gpu_diff() + bottom[0]->offset(i));
    }
  }

  /**
   * bottom[1]' gradient = trans(top_diff) * A
   */
  if (propagate_down[1]) {
    for(int i =0; i < bottom[1]->num(); ++i) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,bottom[1]->channels(), 
          bottom[0]->height() * bottom[0]->width(), bottom[0]->channels(), 
          scale, top[0]->gpu_diff() + top[0]->offset(i),
          bottom[0]->gpu_data() + bottom[0]->offset(i), (Dtype)0.,
          bottom[1]->mutable_gpu_diff() + bottom[1]->offset(i));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FSPLayer);
}  // namespace caffe
