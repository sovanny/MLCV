#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;

// This function reads data and responses from the file <filename>
static bool
read_num_class_data( const string& filename, int var_count,
                     Mat* _data, Mat* _responses )
{
    const int M = 1024;
    char buf[M+2];

    Mat el_ptr(1, var_count, CV_32F);
    int i;
    vector<int> responses;

    _data->release();
    _responses->release();

    FILE* f = fopen( filename.c_str(), "rt" );
    if( !f )
    {
        cout << "Could not read the database " << filename << endl;
        return false;
    }

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        responses.push_back((int)buf[0]);
        ptr = buf+2;
        for( i = 0; i < var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", &el_ptr.at<float>(i), &n );
            ptr += n + 1;
        }
        if( i < var_count )
            break;
        _data->push_back(el_ptr);
    }
    fclose(f);
    Mat(responses).copyTo(*_responses);

    cout << "The database " << filename << " is loaded.\n";

    return true;
}

inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}


static void test_and_save_classifier(const Ptr<StatModel>& model,
                                     const Mat& data, const Mat& responses,
                                     int ntrain_samples, int rdelta,
                                     const string& filename_to_save)
{
    int i, nsamples_all = data.rows;
    double train_hr = 0, test_hr = 0;

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        Mat sample = data.row(i);
        cout << sample<< endl;
        // sample = the resize of you hand. a Mat. (called img2 I think)
        // this two lines are needed for the predicition in facedetect.
        //convert to char
        float r = model->predict( sample );
        r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= nsamples_all - ntrain_samples;
    train_hr = ntrain_samples > 0 ? train_hr/ntrain_samples : 1.;

    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    if( !filename_to_save.empty() )
    {
        model->save( filename_to_save );
    }
}


static bool
build_mlp_classifier( const string& data_filename,
                      const string& filename_to_save,
                      const string& filename_to_load )
{
    const int class_count = 26;
    Mat data;
    Mat responses;

    //the data and respose will be filled up by this function
    bool ok = read_num_class_data( data_filename, 256, &data, &responses );
    if( !ok )
        return ok;

    //declaring the models. you need this line for facedetect
    Ptr<ANN_MLP> model;

    int nsamples_all = data.rows;
    //80% will be the training set
    int ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load MLP classifier
    if( !filename_to_load.empty() )
    {
      // use this line to load the model to do prediction in facedetect_new
        model = load_classifier<ANN_MLP>(filename_to_load);
        if( model.empty() )
            return false;
        ntrain_samples = 0;
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // MLP does not support categorical variables by explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of <class_count> components for training and,
        // therefore, MLP will give us a vector of "probabilities" at the
        // prediction stage
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Mat train_data = data.rowRange(0, ntrain_samples);
        //note: train respose = classcount 0-26
        Mat train_responses = Mat::zeros( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        //the 80 first lines will be used for training. that is why you have to shuffle the file
        // before doing the training
        cout << "Unrolling the responses...\n";
        for( int i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = responses.at<int>(i) - 'A';
            train_responses.at<float>(i, cls_label) = 1.f;
        }

        // 2. train classifier
        //data.cols = the features. the numbe rof features which will be 256
        // you can change to (10, nothing, class_count)
        int layer_sz[] = { data.cols, 10, class_count };
        int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
        Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );

#if 1
        int method = ANN_MLP::BACKPROP;
        double method_param = 0.001;
        int max_iter = 300;
#else
        int method = ANN_MLP::RPROP;
        double method_param = 0.1;
        int max_iter = 1000;
#endif

        Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

        cout << "Training the classifier (may take a few minutes)...\n";
        model = ANN_MLP::create();
        model->setLayerSizes(layer_sizes);
        model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
        model->setTermCriteria(TC(max_iter,0));
        model->setTrainMethod(method, method_param);
        model->train(tdata);
        cout << endl;
    }
// if a filename_to_save is provided, the network will be provided. the BACKUP NETWORK
    test_and_save_classifier(model, data, responses, ntrain_samples, 'A', filename_to_save);
    return true;
}

int main( int argc, char *argv[] )
{
    string filename_to_save = "backup_network";
    string filename_to_load = "";
    string data_filename = "letter.txt";
    int method = 2; // use MLP

    build_mlp_classifier( data_filename, filename_to_save, filename_to_load );

    return 0;
}
