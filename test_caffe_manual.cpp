#include <string>
#include <vector>
#include "caffe/caffe.hpp"

using namespace caffe;

unsigned int get_layer_index(boost::shared_ptr< Net<float> > & net, char *query_layer_name)
{
    std::string str_query(query_layer_name);    
    vector< string > const & layer_names = net->layer_names();
    for( unsigned int i = 0; i != layer_names.size(); ++i ) 
    { 
        if( str_query == layer_names[i] ) 
        { 
            return i;
        } 
    }
    LOG(FATAL) << "Unknown layer name: " << str_query;
}

unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
    std::string str_query(query_blob_name);    
    vector< string > const & blob_names = net->blob_names();
    for( unsigned int i = 0; i != blob_names.size(); ++i ) 
    { 
        if( str_query == blob_names[i] ) 
        { 
            return i;
        } 
    }
    LOG(FATAL) << "Unknown blob name: " << str_query;
}

void get_blob_features(boost::shared_ptr< Net<float> > & net, float *data_ptr, char* layer_name)
{
    unsigned int id = get_layer_index(net, layer_name);
    const vector<Blob<float>*>& output_blobs = net->top_vecs()[id];
    for (unsigned int i = 0; i < output_blobs.size(); ++i) 
    {
        switch (Caffe::mode()) {
        case Caffe::CPU:
            memcpy(data_ptr, output_blobs[i]->cpu_data(),
                sizeof(float) * output_blobs[i]->count());
            break;
        case Caffe::GPU:
            cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
                sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
            break;
        default:
            LOG(FATAL) << "Unknown Caffe mode.";
        }
    }
}

#define PRINT_SHAPE1(x) \
    std::cout << (x).num() << "\t" << (x).channels() << "\t" << (x).height() << "\t" << (x).width() << "\n"; 
#define PRINT_SHAPE2(x) \
    std::cout << (x)->num() << "\t" << (x)->channels() << "\t" << (x)->height() << "\t" << (x)->width() << "\n"; 
#define PRINT_DATA(x) \
    std::cout << (x)[0] << "\t" << (x)[1] << "\n";

int main(int argc, char** argv) {
  
    char *proto = "H:\\Models\\Caffe\\deploy.prototxt";
    char *model = "H:\\Models\\Caffe\\bvlc_reference_caffenet.caffemodel";
    char *mean_file = "H:\\Models\\Caffe\\imagenet_mean.binaryproto";
    Phase phase = TEST;
    //Caffe::set_mode(Caffe::CPU);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    // processing mean
    Blob<float> image_mean;
    BlobProto blob_proto;
    const float *mean_ptr;
    bool succeed = ReadProtoFromBinaryFile(mean_file, &blob_proto);
    if (succeed) 
    {
        std::cout << "read image mean succeeded" << std::endl;
        image_mean.FromProto(blob_proto);
        mean_ptr = (const float *) image_mean.cpu_data();
        unsigned int num_pixel = image_mean.count();
        std::cout << num_pixel << "\n";
        PRINT_SHAPE1(image_mean);
        PRINT_DATA(mean_ptr);
    }
    else
    {
        LOG(FATAL) << "read image mean failed";
    }

    // load net model
    boost::shared_ptr<Net<float> > net(new caffe::Net<float>(proto, phase));
    net->CopyTrainedLayersFrom(model);

    const std::vector<boost::shared_ptr<Layer<float>  >> layers = net->layers();
    std::vector<boost::shared_ptr<Blob<float>  >> net_blobs = net->blobs();
    std::vector<string> layer_names = net->layer_names();
    std::vector<string> blob_names = net->blob_names();
    boost::shared_ptr<Layer<float> > layer;
    boost::shared_ptr<Blob<float> > blob;
    
    // show input blob size
    Blob<float>* input_blobs = net->input_blobs()[0];
    std::cout << "\nInput blob size:\n";
    PRINT_SHAPE2(input_blobs);
    
    // processing blobs of each layer, namely, weights and bias
    const float *mem_ptr;
    CHECK(layers.size() == layer_names.size());
    std::cout << "\n#Layers: " << layers.size() << std::endl;
    std::vector<boost::shared_ptr<Blob<float>  >> layer_blobs;
    for (int i = 0; i < layers.size(); ++i)
    {
        layer_blobs = layers[i]->blobs();
        std::cout << "\n[" << i+1 << "] layer name: " << layer_names[i] << ", type: " << layers[i]->type() << std::endl;
        std::cout << "#Blobs: " << layer_blobs.size() << std::endl;
        for (int j = 0; j < layer_blobs.size(); ++j)
        {
            blob = layer_blobs[j];
            PRINT_SHAPE2(blob);
            mem_ptr = (const float *) blob->cpu_data();
            PRINT_DATA(mem_ptr);
        }
    }

    // get weights and bias from layer name
    char *query_layer_name = "conv1";
    const float *weight_ptr, *bias_ptr;
    unsigned int layer_id = get_layer_index(net, query_layer_name);
    layer = net->layers()[layer_id];
    std::vector<boost::shared_ptr<Blob<float>  >> blobs = layer->blobs();
    if (blobs.size() > 0)
    {
        std::cout << "\nweights and bias from layer: " << query_layer_name << "\n";
        weight_ptr = (const float *) blobs[0]->cpu_data();
        PRINT_DATA(weight_ptr);
        bias_ptr = (const float *) blobs[1]->cpu_data();
        PRINT_DATA(bias_ptr);
    }

    // modify weights from layer name
    blob = net->layers()[layer_id]->blobs()[0];
    unsigned int data_size = blob->count();
    float *data_ptr = new float[data_size];
    caffe_copy(blob->count(), weight_ptr, data_ptr);
    float *w_ptr = NULL;
    data_ptr[0] = 1.1111f;
    data_ptr[1] = 2.2222f;
    switch (Caffe::mode())
    {
    case Caffe::CPU:
        w_ptr = blob->mutable_cpu_data();
        break;
    case Caffe::GPU:
        w_ptr = blob->mutable_gpu_data();
        break;
    default:
        LOG(FATAL) << "Unknown Caffe mode";
    }
    caffe_copy(blob->count(), data_ptr, w_ptr);
    weight_ptr = (const float *) blob->cpu_data();
    delete [] data_ptr;
    std::cout << "\nnew weights and bias from layer: " << query_layer_name << "\n";
    PRINT_DATA(weight_ptr);

    // get features from name
    char *query_blob_name = "conv1"; /* data, conv1, pool1, norm1, fc6, prob, etc */
    unsigned int blob_id = get_blob_index(net, query_blob_name);
    blob = net->blobs()[blob_id];
    unsigned int num_data = blob->count(); /* NCHW=10x96x55x55 */
    mem_ptr = (const float *) blob->cpu_data();    
    std::cout << "\n#Features: " << num_data << "\n";
    PRINT_DATA(mem_ptr);

    char* weights_file = "bvlc_reference_caffenet_new.caffemodel";
    NetParameter net_param;
    net->ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, weights_file);

    std::cout << "END" << std::endl;
    return 0;
}
