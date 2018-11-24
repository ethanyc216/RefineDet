#ifndef CAFFE_NPAIR_DATA_SSL_LAYER_HPP_
#define CAFFE_NPAIR_DATA_SSL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class NPairDataSSLLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
    explicit NPairDataSSLLayer(const LayerParameter& param)
            : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~NPairDataSSLLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "NPairDataSSL"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

  protected:
    virtual void load_batch(Batch<Dtype>* batch);
    vector<pair<string, int> > RandLeafImages(int leaf_label, int rand_image_num);
    void ShuffleSecLabels();
    vector<int> ShuffleLeafsWithSecLabels(int sec_label_num);
    void ShuffleImages(int leaf_label);

    shared_ptr<Caffe::RNG> prefetch_rng_;

    map<int, set<int> > sec_label_leaf_map_;
    map<int, vector<string> > leaf_image_map_;
    map<int, int> leaf_cursor_map_;
    vector<int> sec_labels_;
    int leaf_cursor_;

    int mini_batch_index_;
    bool restarted_;
    Blob<Dtype> mega_data_;
};


}  // namespace caffe

#endif  // CAFFE_NPAIR_DATA_SSL_LAYER_HPP_
