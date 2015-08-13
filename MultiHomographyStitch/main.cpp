/*
 * 
 * @author Carl Ren
 * Created Aug. 13, 2015
 * 
 * */
#include <string>
#include<stdexcept>
#include <opencv2/opencv.hpp>

#include"MultiHomographyStitch.h"

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
std::vector <cv::Mat3b>
loadVideo (const std::string& input_filename, bool need_rotate=false)
{
    std::cerr << "Loading video '" << input_filename << "'\n";
    cv::VideoCapture video_capture (input_filename);
    std::vector <cv::Mat3b> frames;
    
    //size_t i = 0;
    while (video_capture.grab ())
    {
        cv::Mat3b frame;
        video_capture.retrieve (frame);
        
        if(need_rotate)
        {
            cv::transpose(frame.clone(),frame);
        }
            frames.push_back (frame.clone ());

    }
    if (frames.empty ())
    {
        throw std::runtime_error ("The input frames are empty");
    }
    std::cerr << "Loaded " << frames.size () << " frames." << std::endl;
    
    return (frames);
}
////////////////////////////////////////////////////////////////////////////////


int main(int argc, char** argv)
{
//    string img_name1 = "/Users/carlren/Code/VisionTools/MultiHomographyStitch/data/desk5.jpg";
//    string img_name2 = "/Users/carlren/Code/VisionTools/MultiHomographyStitch/data/desk4.jpg";

//    Mat img1 = imread(img_name1);
//    Mat img2 = imread(img_name2);
    
//    string video_name = "/Users/carlren/Code/VisionTools/MultiHomographyStitch/data/table.mov";
    string video_name = "/Users/carlren/Data/in/car5/fyuse_raw.mp4";
    vector<Mat3b> frames = loadVideo(video_name);

    Mat& img1 = frames[0];
    Mat& img2 = frames[10];
    
    
    MultiHomographyStitch my_stitch;
    my_stitch.processImagePair(img1,img2);
    
}
