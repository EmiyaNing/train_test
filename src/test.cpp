#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

const string train_set_path = "/Users/ningkanglin/Documents/code/practice/deep_learning/logical/dataset/train";
const string prefix         = ".jpg";
const string cat_font       = "cat.";
const string dog_font       = "dog.";

vector<int> w(4096, 0);
vector<int> b(4096, 0);

void through_transet(){
    /**
     * throught the cat image
    */
    for(int i = 0; i < 12500; i++){
        string filename = train_set_path + "/" + cat_font + to_string(i) + prefix;
        Mat srcimg = imread(filename);
        if(srcimg.empty()){
            cerr << "picture is empty" << endl;
            exit(-1);
        }
        Mat tempimg;
        Mat dstimg;
        resize(srcimg, tempimg, Size(64,64));
        cvtColor(tempimg, dstimg, COLOR_BGR2GRAY);
        imwrite(filename, dstimg);
        if(0 == i % 100){
            cout << "now is throught the cat picture " << i << endl;
        }
    }
    /**
     * through the dog image
    */
   for(int i = 0; i < 12500; i++){
       string filename = train_set_path + "/" + dog_font + to_string(i) + prefix;
       Mat srcimg = imread(filename);
       if(srcimg.empty()){
           cerr << "picture is empty" << endl;
           exit(-1); 
       }
       Mat tempimg;
       Mat dstimg;
       resize(srcimg, tempimg, Size(64,64));
       cvtColor(tempimg, dstimg, COLOR_BGR2GRAY);
       imwrite(filename, dstimg);
       if(0 == i % 100){
           cout << "now is throught the dog picture" << i << endl;
       }
   }
}




int main(int argc, char* argv[]){
    cout << "start to through the train set" << endl;
    through_transet();
    return 0;
}