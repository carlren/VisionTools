#include <string>

#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/token_functions.hpp>

#include <opencv2/opencv.hpp>
#include"prob_model.h"

#include"MotionDetector.h"

using namespace cv;
using namespace std;
using namespace fyusion;

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

void RunMotionDetection(std::string& video_name, bool need_rotate){
    
    std::vector<cv::Mat3b> frames = loadVideo(video_name, need_rotate);
    int w = frames[0].cols/3;
    int h = frames[0].rows/3;    
    Size tSize(w,h);
  
    Mat smallFrame(tSize.height,tSize.width,CV_8UC3);
    Mat motionFrame(tSize.height,tSize.width,CV_8UC3);
    
    bool show_frames = true;
    bool need_refresh =true;
    int frame_id = 1;
    
    MotionDetector* motion_detector = new MotionDetector(need_rotate);
    
//    resize(frames[frame_id], smallFrame, tSize);
//    motion_detector->updateDualSGM(smallFrame);
    
    namedWindow("origin",WINDOW_AUTOSIZE|WINDOW_OPENGL);
    namedWindow("motion",WINDOW_AUTOSIZE|WINDOW_OPENGL);
    namedWindow("sgm_mean",WINDOW_AUTOSIZE|WINDOW_OPENGL);
    namedWindow("warped_mean",WINDOW_AUTOSIZE|WINDOW_OPENGL);
    
//    namedWindow("err_before");
//    namedWindow("err_after");
    
     moveWindow("origin",100,100);
     moveWindow("motion",100,1000);
     moveWindow("sgm_mean", 500,100);
     moveWindow("warped_mean", 500,300);
     
//     moveWindow("err_before",500,700);
//     moveWindow("err_after",500,1000);
     
    while(show_frames){
        
        if(need_refresh)
        {
            resize(frames[frame_id], smallFrame, tSize);
                        
            size_t tic = getTickCount();
            motion_detector->detectMotion(smallFrame,motionFrame,Matx33f::eye());
            size_t toc = getTickCount() - tic;
            cout<<frame_id<<"-detection time:" << (double) toc * 1000 / getTickFrequency() << " ms" <<endl;
                        
            need_refresh = false;
        }
        
        imshow("origin", smallFrame);
        imshow("motion",motionFrame);
        
        
        char key = cv::waitKey(1);
        
        switch (key) {
            case 27: case 'q':
                show_frames = false;
                break;
            case 'x':
                ++frame_id;
                if (frame_id>=frames.size()-1) {
                    frame_id = frames.size()-2;
                }
                else need_refresh = true;
                break;
            case 'z':
                --frame_id;
                if (frame_id<=1) {
                    frame_id=1;
                }
                else need_refresh = true;
                break;
            default:
                break;
        }
    }
    
}
/////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    try {
    
        boost::program_options::options_description description ("MotionDetector");
        
        description.add_options()
        ("help","Desplay this help message")
        ("video", boost::program_options::value<std::string>(), "the name of the video")
        ("need_rotate", "whether the video need to rotated");
        
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(description).run(), vm);
        boost::program_options::notify(vm);
        
        std::string video_name;
        bool need_rotate = false;
        
        if (vm.count("help")) {
            std::cerr<<description<<std::endl;
        }
        if(vm.count("need_rotate")){
            need_rotate = true;
        }
        
        if (vm.count("video")) {
            video_name = vm["video"].as<std::string>();
            std::cerr<<"processing single video from:" << video_name << std::endl;
            
            RunMotionDetection(video_name,need_rotate);
        }
        else
        {
            std::cerr<<description<<std::endl;
        }
        
    } catch (std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
