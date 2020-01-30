/**
 * 应为我在kaagle上下载的数据集的test集没有标签，应此暂时不使用test集合。。
 * 我现在从train集合中分出5000个集合用于test
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cblas.h>
using namespace std;

const string train_set_path = "/Users/ningkanglin/Documents/code/practice/deep_learning/logical/dataset/train";
const string prefix         = ".jpg";
const string cat_font       = "cat.";
const string dog_font       = "dog.";

vector<float> w(4096, 0);
float b = 0;

/**
 * 通过vector存取train和test集和。。。
*/
//vector<Mat> trainset;
//vector<Mat> testset;

cv::Mat trainset[20000];
cv::Mat testset[5000];

vector<float> train_y(20000, 0);
vector<float> test_y(5000, 0);

/**
 * vector A save the result of sigmod function
*/
vector<float> A(20000, 0);

struct Grads{
    vector<float> dw;
    float db;
};


/**
 * 该函数读取数据集文件夹中的图片，存入数组中。。
*/
void fill_the_train_set(cv::Mat * trainset, vector<float> & train_y){
    int count = 0;
    for(int i = 0; i < 12000; i++){
        string catname = train_set_path + "/" + cat_font + to_string(i) + prefix;
        string dogname = train_set_path + "/" + dog_font + to_string(i) + prefix;
        trainset[count] = cv::imread(catname);
        train_y[count] = 1;
        count++;
        trainset[count] = cv::imread(dogname);
        train_y[count] = 0;
        count++;
        if(0 == (i % 100)){
            cout << "now adding trainset " << i << endl;
        }
    }
}

/**
 * 读取测试集合数据
*/
void fill_the_test_set(cv::Mat * trainset, vector<float> & test_y){
    int count = 0;
    for(int i = 0; i < 5000; i++){
        string catname = train_set_path + "/" + cat_font + to_string(i + 10000) + prefix;
        string dogname = train_set_path + "/" + dog_font + to_string(i + 10000) + prefix;
        testset[count] = cv::imread(catname);
        test_y[count]  = 1;
        count++;
        testset[count] = cv::imread(dogname);
        test_y[count]  = 0;
        count++;
        if(0 == (i % 100)){
            cout << "now adding testset " << i << endl;
        }
    }
}

float sigmod(float x){
    float s = 1/(1 + exp(-x));
    return s;
}

/**
 * So now I think I shoud scan every photo and put it's piex to a two demision array...
 * before call this function we should init the grads
*/

float propagate(vector<float> &w, float b, cv::Mat train[], vector<float> symbol, Grads &grads){
    vector<float> log_temp_alpha;
    vector<float> log_temp_beta;
    vector<float> symbol_beta;                                  // 1 减去symbol中的每个元素
    vector<float> symbol_alpha;                                 // A-Y.....
    float cost;
    // cost  = -(np.dot(symbol, np.log(A.T)) + np.dot(np.log(1-A), (1-symbol).T))/m;
    // now let symbol_beta = 1 - symbol
    for(int i = 0; i < 20000; i++){
        symbol_beta.push_back(1-symbol[i]);
    }
    // caculate the vector A, whose element a = w*x + b;
    // from the equation we will use log(1-A),and log(A). So I let log_temp_alpha = log(A), log_temp_beta = log(1 - A).
    for(int i = 0; i < 20000; i++){
        // first get the sigmod function result...
        A[i] = sigmod(cblas_sdot(4096, w.data(), 1, train[i].ptr<float>(0), 1));
        // second get the log(A) and log(1 - A) and (A - Y)
        log_temp_alpha.push_back(log(A[i]));
        log_temp_beta.push_back(log(1 - A[i]));
        symbol_alpha.push_back(A[i] - symbol[i]);
    }
    cost = -(cblas_sdot(20000, A.data(), 1, symbol.data(), 1) + cblas_sdot(20000, log_temp_beta.data(), 1, symbol_beta.data(), 1))/20000;
    // now caculate the dw and db
    // dw = \sum^m_i x^{i} (a^i - y^i) / m;
    // db = \sum^m_i (a^i - y^i) /m;
    vector<float> dx_tem(20000, 0);
    // each cycle get one vector and caculate the result....
    for(int i = 0; i < 4096; i++){
        for(int j = 0; j < 20000; j++){
            // opencv mat object's function () get the point ...
            dx_tem[j] = *train[i].ptr<float>(i);
        }
        grads.dw[i] = cblas_sdot(20000, dx_tem.data(), 1, symbol_alpha.data(), 1);
    }
    grads.db = cblas_sasum(20000, symbol_alpha.data(), 1);
    return cost;
}

void optimize(vector<float> w, float b, cv::Mat train[], vector<float> y, int num_iterations, float learning_rate){
    Grads grads;
    grads.db = 0;
    for(int j = 0; j < 20000; j++){
        grads.dw.push_back(0);
    }
    for(int i = 0; i < num_iterations; i++){
        float cost = propagate(w,b,train,y,grads);
        b = b - learning_rate * grads.db;
        // w = w - dw * learn_rate
        cblas_saxpy(4096, learning_rate, grads.dw.data(), 1, w.data(), 1);
        if(0 == i % 100){
            cout << "Cost after iteration " << i << " : " << cost << endl;
        }
    }
}

bool predict(cv::Mat x, vector<float> w, float b){
    float temp = cblas_sdot(4096, x.ptr<float>(0), 1, w.data(), 1);
    float A = sigmod(temp);
    if(A > 0.5){
        cout << "This picture has a cat!!!" << endl;
        return true;
    }else{
        cout << "This picture doesn't have cat!!!" << endl;
        return false;
    }
}

int main(int argc, const char * argv[]){
    int num_iterations;
    float learning_rate;
    cout << "Now please input the iterations times and learning_rate(iteration times > 10000 and 0 < learning_rate < 1):" << endl;
    cin >> num_iterations >> learning_rate;
    fill_the_train_set(trainset, train_y);
    fill_the_test_set(testset, test_y);
    if((learning_rate < 1 && learning_rate > 0) && (num_iterations > 10000)){
        optimize(w, b, trainset, train_y, num_iterations, learning_rate);
    };
    return 0;
}