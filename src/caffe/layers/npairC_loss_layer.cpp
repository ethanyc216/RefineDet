#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/npairC_loss_layer.hpp"
#include <time.h>

namespace caffe {

template <typename Dtype>
void NPairCLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  gamma_ = 0.001;

  int group = this->layer_param_.group_id();
  int kG_ = bottom[1]->channels();
  group_id_ = group;
  kG = kG_;
  LOG(INFO) << "NPairCLossLayer:LayerSetUp";
  LOG(INFO) << "bottom size " << bottom.size();
}

template <typename Dtype>
void NPairCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1,1,1,1);
  top[1]->Reshape(1,1,1,1);

  const int batch_size = bottom[0]->num();
  W_mat_.Reshape(1,1,batch_size,batch_size);
}

template <typename Dtype>
void NPairCLossLayer<Dtype>::CalSquEucDist(const int qry_num, const int ref_num, const int dim,
    const Dtype* _Q, const Dtype* _R, Dtype* _Dist) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, qry_num, ref_num, dim, gamma_, _Q, _R, 0.0, _Dist);
}

template <typename Dtype>
Dtype NPairCLossLayer<Dtype>::CalLossAndWeightMatrix(const int batch_size, const Dtype* labels,
    map<Dtype, vector<int> > reference_labels_idx, Dtype* _W) {
  Dtype loss = 0;
  
  for (int qry_idx=0; qry_idx<batch_size; qry_idx++) {
    Dtype qry_label = labels[qry_idx * kG + group_id_];
    vector<int> ref_idx = reference_labels_idx[static_cast<int>(qry_label)];
    CHECK_EQ(ref_idx.size()-1, 1) << "num of pos must be 1";

  int pos_idx = -1;
  for (int t=0; t<ref_idx.size(); t++) {
    if (ref_idx[t] != qry_idx) {
      pos_idx = ref_idx[t];
    break;
    }
  }
  CHECK(pos_idx >= 0) << "pos_idx is invalid";

    Dtype single_loss = CalLossAndWeightRow(qry_idx, pos_idx, batch_size, _W+qry_idx*batch_size);
    loss += single_loss;
  }

  loss /= (Dtype)batch_size;
  return loss;
}

template <typename Dtype>
Dtype NPairCLossLayer<Dtype>::CalLossAndWeightRow(int qry_idx, int pos_idx, const int batch_size,
    Dtype* _Wq) {
  Dtype max_distance_diff = -1e10;
  for (int ref_idx=0; ref_idx<batch_size; ref_idx++) {
    if (qry_idx == ref_idx || pos_idx == ref_idx)
    continue;
  int neg_idx = ref_idx;
  Dtype distance_diff = _Wq[neg_idx]-_Wq[pos_idx];
  if (distance_diff > max_distance_diff)
    max_distance_diff = distance_diff;
  }
  // for test
  max_distance_diff = 0;

  // compute exponential and phi
  vector<double> exp_dd(batch_size, 0);
  Dtype between_phi = 0;
  for (int ref_idx=0; ref_idx<batch_size; ref_idx++) {
  if (ref_idx == qry_idx) continue;
    if (ref_idx != pos_idx) {
    int neg_idx = ref_idx;
    Dtype distance_diff = _Wq[neg_idx]-_Wq[pos_idx];
    exp_dd[neg_idx] = exp(distance_diff-max_distance_diff);
    between_phi += exp_dd[neg_idx];
    }
  }

  // calc loss
  Dtype loss = 0;
  if (-max_distance_diff >= 16)
  loss = log(1+exp(max_distance_diff)*between_phi);
  else
    loss = log(exp(-max_distance_diff)+between_phi)+max_distance_diff;

  // recalc weight matrix
  Dtype common_factor = 0;
  if (-max_distance_diff >= 16)
    common_factor = (Dtype)(exp(max_distance_diff)/(1+exp(max_distance_diff)*between_phi)/batch_size);
  else
    common_factor = (Dtype)(1.0/batch_size/(exp(-max_distance_diff)+between_phi));
  for (int ref_idx=0; ref_idx<batch_size; ref_idx++) {
  if (ref_idx == qry_idx)
    _Wq[qry_idx] = 0;
    else if (ref_idx == pos_idx)
    _Wq[pos_idx] = (Dtype)(common_factor*between_phi*gamma_*(-1.0));
  else {
    int neg_idx = ref_idx;
    _Wq[neg_idx] = (Dtype)(common_factor*exp_dd[neg_idx]*gamma_);
  }
  }

  return loss;
}

template <typename Dtype>
Dtype NPairCLossLayer<Dtype>::CalPrecision(const int qry_num, const int ref_num,
    map<Dtype,vector<int> >reference_labels_idx, const Dtype* _W, const Dtype* labels, int topk) {
  Dtype precision = 0;
  for(int qry_i = 0; qry_i<qry_num; qry_i++){
    Dtype qry_label = labels[qry_i * kG + group_id_];
    Dtype precision_one = 0;
    vector<Dtype> buf(ref_num, 0);
    for(int ref_i = 0; ref_i < ref_num; ref_i++){
      buf[ref_i] = -_W[qry_i * ref_num + ref_i];
    }
    std::sort(buf.begin(), buf.end());
    CHECK_LT(topk, ref_num) << "ERROR:topk >= ref_num";
    Dtype threshold = buf[topk - 1];
    for(int k = 0; k < reference_labels_idx[static_cast<int>(qry_label)].size(); k++){
      int ref_idx = reference_labels_idx[static_cast<int>(qry_label)][k];
      if(-_W[qry_i * ref_num + ref_idx] <= threshold){
        precision_one += 1;
      }
    }
    // fix same value problem at topk
    if (precision_one > topk) {
      precision_one = topk;
    }
    precision += precision_one /topk;
  }
  precision = precision / qry_num;
  return precision;
}

template <typename Dtype>
void NPairCLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  map<Dtype, vector<int> > reference_labels_idx;

  int batch_size = bottom[0]->num();
  int feat_dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  CHECK_EQ(1, spatial_dim);
  // int kG = bottom[1]->channels();

  const Dtype* _Q = bottom[0]->cpu_data(); 
  const Dtype* labels = bottom[1]->cpu_data();
  Dtype* _Q_diff = bottom[0]->mutable_cpu_diff();

  Dtype* _W = W_mat_.mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    int label = static_cast<int>(labels[i * kG + group_id_]);
    reference_labels_idx[label].push_back(i);
  }

  // calc weight matrix
  CalSquEucDist(batch_size, batch_size, feat_dim, _Q, _Q, _W);
  // calc precision
  Dtype precision = CalPrecision(batch_size, batch_size, reference_labels_idx, _W, labels, 10);
  // calc loss
  Dtype loss = CalLossAndWeightMatrix(batch_size, labels, reference_labels_idx, _W);

  top[0]->mutable_cpu_data()[0] = loss;
  top[1]->mutable_cpu_data()[0] = precision;
}

template <typename Dtype>
void NPairCLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int batch_size = bottom[0]->num();
    int feat_dim = bottom[0]->count() / bottom[0]->num();
    Dtype* _W = W_mat_.mutable_cpu_data();
    const Dtype* _Q = bottom[0]->cpu_data();
    Dtype* _Q_diff = bottom[0]->mutable_cpu_diff();

    // calc gradient
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, feat_dim, batch_size, 1.0, _W, _Q, 0.0, _Q_diff);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, batch_size, feat_dim, batch_size, 1.0, _W, _Q, 1.0, _Q_diff);

    // regularize
    caffe_axpy(bottom[0]->count(), (Dtype)(gamma_*0.001/batch_size), _Q, _Q_diff);

    const Dtype alpha = top[0]->cpu_diff()[0];
  caffe_scal(bottom[0]->count(), alpha, bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(NPairCLossLayer);
#endif

INSTANTIATE_CLASS(NPairCLossLayer);
REGISTER_LAYER_CLASS(NPairCLoss);

}  // namespace caffe
