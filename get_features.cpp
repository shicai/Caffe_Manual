/***
 usage:
 get_features.exe feat.prototxt H:\Models\Caffe\bvlc_reference_caffenet.caffemodel 6 
 conv1,fc7,prob,argmax conv1.dat,fc7.dat,prob.dat,argmax.dat GPU 0 

 for feat.prototxt, see the following example:
 name: "CaffeNet"
 state {
    phase: TEST
 }
 layer {
     name: "data"
     type: "ImageData"
     top: "data"
     top: "label"
     transform_param {
         mirror: false
         crop_size: 227
         mean_file: "imagenet_mean.binaryproto"
     }
     image_data_param {
         source: "file_list.txt"
         batch_size: 1
         new_height: 256
         new_width: 256
     }
 }
 layer {
     name: "conv1"
     type: "Convolution"
     bottom: "data"
     top: "conv1"
     convolution_param {
         num_output: 96
         kernel_size: 11
         stride: 4
     }
 }
 #################################################################################
 ######some lines are ignored here for simplicity, complete them by yourself######
 #################################################################################
 layer {
     name: "fc8"
     type: "InnerProduct"
     bottom: "fc7"
     top: "fc8"
     inner_product_param {
        num_output: 1000
     }
 }
 layer {
     name: "prob"
     type: "Softmax"
     bottom: "fc8"
     top: "prob"
 }
 layer {
     name: "argmax"
     type: "ArgMax"
     bottom: "prob"
     top: "argmax"
     argmax_param {
        top_k: 1
     }
 }

 for file_list.txt, see the following example:
     H:\Data\ILSVRC2012\n01440764\n01440764_18.JPEG 0
     H:\Data\ILSVRC2012\n01440764\n01440764_297.JPEG 0
     H:\Data\ILSVRC2012\n01443537\n01443537_395.JPEG 1
     H:\Data\ILSVRC2012\n01443537\n01443537_693.JPEG 1
     H:\Data\ILSVRC2012\n01518878\n01518878_103.JPEG 9
     H:\Data\ILSVRC2012\n01518878\n01518878_477.JPEG 9

 How to load features in Matlab? use the following function, see:
 prob = sc_load('prob.dat');

 function data = sc_load(filename, type)
     if ~exist('type', 'var') || isempty(type)
        type = 'single';
     end

     fid = fopen(filename, 'r');    
     rows = fread(fid, 1, type);
     cols = fread(fid, 1, type);
     data = fread(fid, rows * cols, type);    
     fclose(fid);

     data = reshape(data, rows, cols);
     switch type
     case 'int32'
        data = int32(data);
     case 'single'
        data = single(data);
     end
 end

***/

#include <string>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
using boost::shared_ptr;
using std::string;
using namespace caffe; 

#define MAX_FEAT_NUM 16

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        LOG(ERROR) << "get_features proto_file model_file iterations blob_name1[,name2] save_name1[,name2]"
            << "[CPU/GPU] [Device ID]";
        return 1;
    }

    Phase phase = TEST;
    if (argc >= 7 && strcmp(argv[6], "GPU") == 0) 
    {
        Caffe::set_mode(Caffe::GPU);
        int device_id = 0;
        if (argc == 8)
        {
            device_id = atoi(argv[7]);
        }
        Caffe::SetDevice(device_id);
        LOG(ERROR) << "Using GPU #" << device_id;
    } else {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }

    boost::shared_ptr<Net<float> > feature_net;
    feature_net.reset(new Net<float>(argv[1], phase));
    feature_net->CopyTrainedLayersFrom(argv[2]);

    int total_iter = atoi(argv[3]);
    LOG(ERROR) << "Running " << total_iter << " iterations.";

    std::string feature_blob_names(argv[4]);
    std::vector<std::string> blob_names;
    boost::split(blob_names, feature_blob_names, boost::is_any_of(","));

    std::string save_file_names(argv[5]);
    std::vector<std::string> file_names;
    boost::split(file_names, save_file_names, boost::is_any_of(","));
    CHECK_EQ(blob_names.size(), file_names.size()) <<
        " the number of feature blob names and save file names must be equal";

    size_t num_features = blob_names.size();
    for (size_t i = 0; i < num_features; i++) 
    {
        CHECK(feature_net->has_blob(blob_names[i]))
            << "Unknown feature blob name " << blob_names[i] << " in the network";
    }

    FILE *fp[MAX_FEAT_NUM];
    for (size_t i = 0; i < num_features; i++)
    {
        fp[i] = fopen(file_names[i].c_str(), "wb");
    }

    for (int i = 0; i < total_iter; ++i)
    {
        feature_net->ForwardPrefilled();
        for (int j = 0; j < num_features; ++j) 
        {
            const boost::shared_ptr<Blob<float> > feature_blob = feature_net->blob_by_name(blob_names[j]);
            float num_imgs = feature_blob->num() * total_iter;
            float feat_dim = feature_blob->count() / feature_blob->num();
            const float* data_ptr = (const float *) feature_blob->cpu_data();

            if (i == 0)
            {                
                fwrite(&feat_dim, sizeof(float), 1, fp[j]);
                fwrite(&num_imgs, sizeof(float), 1, fp[j]);        
            }
            fwrite(data_ptr, sizeof(float), feature_blob->count(), fp[j]);
        }
    }

    for (size_t i = 0; i < num_features; i++)
    {
        fclose(fp[i]);
    }
    return 0;
}
