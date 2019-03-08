#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <sstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <time.h>

#include "caffe/layer.hpp"
#include "caffe/layers/npair_data_with_aspect_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#define INVALID_VALUE -99999

namespace caffe {

template <typename Dtype>
vector<pair<string, int> > NPairDataWithAspectLayer<Dtype>::RandLeafImages(int leaf_label, int rand_image_num) {
  vector<pair<string, int> > images;
  for (int k=0; k<rand_image_num; k++) {
    images.push_back(pair<string, int>(leaf_image_map_[leaf_label][leaf_cursor_map_[leaf_label]], leaf_label));
    leaf_cursor_map_[leaf_label] ++;
    if (leaf_cursor_map_[leaf_label] >= leaf_image_map_[leaf_label].size()) {
      if (this->layer_param_.image_data_param().shuffle())
        ShuffleImages(leaf_label);
      leaf_cursor_map_[leaf_label] = 0;
    }
  }
  return images;
}

template <typename Dtype>
NPairDataWithAspectLayer<Dtype>::~NPairDataWithAspectLayer<Dtype>() {
  // this->JoinPrefetchThread();
  this->StopInternalThread();
}

template <typename Dtype>
void NPairDataWithAspectLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
    "new_height and new_width to be set at the same time.";
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const string& source = this->layer_param_.image_data_param().source();

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  CHECK(infile.good())
    << "Could not open image list (filename: \""+ source + "\")";

  const int batch_size = this->layer_param_.image_data_param().batch_size();

  // Read the file with filenames and labels
  string line;
  string filename;
  int leaf;
  set<int> leafs_set;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    leaf = INVALID_VALUE;
    iss >> filename >> leaf;
    if (leaf == INVALID_VALUE || leaf == -1)
      continue;

    leafs_set.insert(leaf);

    if (leaf_image_map_.find(leaf) == leaf_image_map_.end()) {
      vector<string> images;
      leaf_image_map_[leaf] = images;
    }
    leaf_image_map_[leaf].push_back(filename);

    if (leaf_cursor_map_.find(leaf) == leaf_cursor_map_.end())
      leaf_cursor_map_[leaf] = 0;
  }
  leafs_.assign(leafs_set.begin(), leafs_set.end());

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLeafs();
  }
  leaf_cursor_ = 0;

  int channels = is_color ? 3 : 1;
  vector<int> top_shape;
  top_shape.push_back(batch_size);
  top_shape.push_back(channels);

  const int crop_size = this->layer_param_.transform_param().crop_size();
  if (crop_size > 0) {
    top_shape.push_back(crop_size);
    top_shape.push_back(crop_size);
    top[0]->Reshape(top_shape);
    // this->prefetch_data_.Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }
    top_shape[0] = 1;
    this->transformed_data_.Reshape(top_shape);

    // top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    // this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    // this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top_shape.push_back(new_height);
    top_shape.push_back(new_width);
    top[0]->Reshape(top_shape);
    // this->prefetch_data_.Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }
    top_shape[0] = 1;
    this->transformed_data_.Reshape(top_shape);

    // top[0]->Reshape(batch_size, channels, new_height, new_width);
    // this->prefetch_data_.Reshape(batch_size, channels, new_height, new_width);
    // this->transformed_data_.Reshape(1, channels, new_height, new_width);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();

  // label: total_query_num equals to label shape.
  // top[1]->Reshape(batch_size, 1, 1, 1);
  // this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
  }

  LOG(INFO) << "Data Setup OK!!!";
}

template<typename Dtype>
void NPairDataWithAspectLayer<Dtype>::ShuffleLeafs() {
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(leafs_.begin(), leafs_.end(), prefetch_rng);
}

template<typename Dtype>
void NPairDataWithAspectLayer<Dtype>::ShuffleImages(int leaf_label) {
  caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(leaf_image_map_[leaf_label].begin(), leaf_image_map_[leaf_label].end(), prefetch_rng);
}

template <typename Dtype>
void NPairDataWithAspectLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  Datum datum;
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  int rand_image_num = 2;

  int item_id = 0;
  while (true) {
    bool is_finish = false;

    int leaf_label = leafs_[leaf_cursor_];
    leaf_cursor_ ++;

    // randome select samples from certain class
    vector<pair<string, int> > leaf_sample_images = RandLeafImages(leaf_label, rand_image_num);

    for (int i=0; i<leaf_sample_images.size(); i++) {
      const string& filename = leaf_sample_images[i].first;
      int label = leaf_sample_images[i].second;

      // load image
      const float max_aspect_ratio = 2;
      const float min_aspect_ratio = 0.19;
      cv::Mat cv_img;
      bool read_status = ReadImageToCVMat(root_folder + '/' + filename,
          new_height, new_width, is_color,
          max_aspect_ratio, min_aspect_ratio, cv_img);
      CHECK(read_status);
      //cv::Mat cv_img = ReadImageToCVMat(root_folder + '/' + filename, new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << filename;

      // transform feature
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      prefetch_label[item_id] = label;

      item_id ++;
      if (item_id >= batch_size) {
        is_finish = true;
        break;
      }
    }

    if (is_finish) {
      if (image_data_param.shuffle())
        ShuffleLeafs();
      leaf_cursor_ = 0;
      break;
    }
  }
}

INSTANTIATE_CLASS(NPairDataWithAspectLayer);
REGISTER_LAYER_CLASS(NPairDataWithAspect);

}  // namespace caffe
