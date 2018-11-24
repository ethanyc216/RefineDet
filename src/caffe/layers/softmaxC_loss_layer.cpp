#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/softmaxC_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxCWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top) {
// void SoftmaxCWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  int group = this->layer_param_.group_id();
  group_id_ = group;
  LOG(INFO) << "SoftmaxCWithLossLayer:LayerSetUp";
  LOG(INFO) << "bottom size " << bottom.size();
}

template <typename Dtype>
void SoftmaxCWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
// void SoftmaxCWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxCWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
// void SoftmaxCWithLossLayer<Dtype>::Forward_cpu(
//     const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  CHECK_EQ(1, spatial_dim);
  int kG = bottom[1]->channels();
  Dtype loss = 0;
  int total = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int l = static_cast<int>(label[i * kG + group_id_]);
      if (l == -1) {
        // skip
      } else {
        total++;
        Dtype dd = log(std::max(prob_data[i * dim + l * spatial_dim + j], Dtype(FLT_MIN)));
        if (isnan(dd)) {
          printf("dd is nan\n");
        }
        // if (dd <= -10) {
        //     printf("prob is very small\n");
        // }
        loss -= dd;
      }
    }
  }
  // printf("softmax forward, total %d, num %d, spatial_dim %d\n", total, num, spatial_dim);
  top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  // (*top)[0]->mutable_cpu_data()[0] = loss / total;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxCWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom) {
// void SoftmaxCWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down,
//     vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int kG = bottom[1]->channels();

    int total = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        int l = static_cast<int>(label[i * kG + group_id_]);
        if (l == -1) {
          // ignore and not propagate back any gradient
          for (int k = 0; k < dim; ++k) {
            bottom_diff[i * dim + k] = 0;
          }
        } else {
          ++total;
          bottom_diff[i * dim + l * spatial_dim + j] -= 1;
        }
      }
    }
    // printf("backward, total %d, num %d, spatial_dim %d\n", total, num, spatial_dim);

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
    // caffe_scal(prob_.count(), loss_weight / total, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxCWithLossLayer);
#endif

// INSTANTIATE_CLASS(SoftmaxCWithLossLayer);
INSTANTIATE_CLASS(SoftmaxCWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxCWithLoss);

}  // namespace caffe
