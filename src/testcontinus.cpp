#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

const string train_set_path = "/Users/ningkanglin/Documents/code/practice/deep_learning/logical/dataset/train";
const string prefix         = ".jpg";
const string font           = "cat.";

void test_whether_is_constunious(){
    for(int i = 0; i < 12500; i++){
        string filename = train_set_path + "/" + font + to_string(i) + prefix;
        Mat srcimg = imread(filename);
        if(srcimg.empty()){
            cerr << "picture is empty" << endl;
            exit(-1);
        }
        if((0 == i % 10)&(srcimg.isContinuous())){
            cout << "The picture " << i << " is continuous..." << endl;
        }else if( 0 == i % 10){
            cout << "The picture " << i << " is not continuous!!!" << endl;
        }
    }
}

int main(int argc, char * argv[]){
    test_whether_is_constunious();
    return 0;
}