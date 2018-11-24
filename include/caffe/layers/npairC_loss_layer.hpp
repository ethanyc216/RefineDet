#ifndef CAFFE_NPAIRC_LOSS_LAYER_HPP_
#define CAFFE_NPAIRC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class NPairCLossLayer : public LossLayer<Dtype> {
 public:
  explicit NPairCLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_fea_(), W_mat_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "NPairCLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CalSquEucDist(const int qry_num, const int ref_num, const int dim,
      const Dtype* _Q, const Dtype* _R, Dtype* _Dist);
  virtual Dtype CalPrecision(const int qry_num, const int ref_num,
      map<Dtype,vector<int> > reference_labels_idx, const Dtype* _W, const Dtype* labels, int topk);

  virtual Dtype CalLossAndWeightMatrix(const int batch_size, const Dtype* labels,
      map<Dtype, vector<int> > reference_labels_idx, Dtype* _W);
  virtual Dtype CalLossAndWeightRow(int qry_idx, int pos_idx, const int batch_size, Dtype* _Wq);

  Blob<Dtype> diff_fea_;
  Blob<Dtype> W_mat_;
  Dtype gamma_;
  int group_id_;
  int kG;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
