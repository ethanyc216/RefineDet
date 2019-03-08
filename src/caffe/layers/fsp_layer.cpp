#include <algorithm>
#include <vector>

#include "caffe/layers/fsp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FSPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num_axes() == 4 && bottom[1]->num_axes() == 4) 
      << "bottom[0] and bottom[1] must have 4 axes";

  CHECK(bottom[0]->num() == bottom[1]->num())
      << "bottom[0]->num() must equal to bottom[1]->num()";
  int fea_map_size1 = bottom[0]->height() * bottom[0]->width();
  int fea_map_size2 = bottom[1]->height() * bottom[1]->width();
  CHECK(fea_map_size1 == fea_map_size2) << 
    "fea_map_size1 must equal to fea_map_size2 " << fea_map_size1 <<
    " != " << fea_map_size2;
}

template <typename Dtype>
void FSPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(bottom[0]->shape());
  top_shape[1] = 1;
  top_shape[2] = bottom[0]->channels();
  top_shape[3] = bottom[1]->channels();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void FSPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype * bottom_data_0 = bottom[0]->cpu_data();
  const Dtype * bottom_data_1 = bottom[1]->cpu_data();
  Dtype scale = 1.0 / (Dtype)(bottom[0]->height() * bottom[0]->width());
  /**
   * define A = bottom[0] feature
   * define B = bottom[1] feature
   * FSP Matrix C = A * trans(B)
   */
  for (int i = 0; i < bottom[0]->num(); ++i){
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), 
        bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),
        scale, bottom_data_0 + bottom[0]->offset(i), 
        bottom_data_1 + bottom[1]->offset(i), (Dtype)0.,
        top[0]->mutable_cpu_data() + top[0]->offset(i));
  }
}

template <typename Dtype>
void FSPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype scale = 1.0 / (Dtype)(bottom[0]->height() * bottom[0]->width());
  /**
   * bottom[0]' gradient = top_diff * B 
   */
  if (propagate_down[0]) {
    for(int i =0; i < bottom[0]->num(); ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
          bottom[0]->channels(), bottom[1]->height() * bottom[1]->width(),
          bottom[1]->channels(), scale, top[0]->cpu_diff() + top[0]->offset(i),
          bottom[1]->cpu_data() + bottom[1]->offset(i), (Dtype)0.,
          bottom[0]->mutable_cpu_diff() + bottom[0]->offset(i));
    }
  }

  /**
   * bottom[1]' gradient = trans(top_diff) * A
   */
  if (propagate_down[1]) {
    for(int i =0; i < bottom[1]->num(); ++i) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,bottom[1]->channels(), 
          bottom[0]->height() * bottom[0]->width(), bottom[0]->channels(), 
          scale, top[0]->cpu_diff() + top[0]->offset(i),
          bottom[0]->cpu_data() + bottom[0]->offset(i), (Dtype)0.,
          bottom[1]->mutable_cpu_diff() + bottom[1]->offset(i));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(FSPLayer);
#endif

INSTANTIATE_CLASS(FSPLayer);
REGISTER_LAYER_CLASS(FSP);
}  // namespace caffe
