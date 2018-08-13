#ifndef CAFFE_NPAIR_DATA_LAYER_HPP_
#define CAFFE_NPAIR_DATA_LAYER_HPP_

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
class NPairDataLayer : public BasePrefetchingDataLayer<Dtype> {
  public:
    explicit NPairDataLayer(const LayerParameter& param)
            : BasePrefetchingDataLayer<Dtype>(param) {}
    virtual ~NPairDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "NPairData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

  protected:
    virtual void load_batch(Batch<Dtype>* batch);
    vector<pair<string, int> > RandLeafImages(int leaf_label, int rand_image_num);
    void ShuffleLeafs();
    void ShuffleImages(int leaf_label);

    shared_ptr<Caffe::RNG> prefetch_rng_;

    map<int, vector<string> > leaf_image_map_;
    map<int, int> leaf_cursor_map_;
    vector<int> leafs_;
    int leaf_cursor_;

    int mini_batch_index_;
    bool restarted_;
    Blob<Dtype> mega_data_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
