//
//  main.cpp
//  pwpLib
//
//  Created by Carl Ren on 03/05/2015.
//  Copyright (c) 2015 Carl Ren. All rights reserved.
//
#include "pwp/PWPTracker.hpp"
#include <iostream>


using namespace std;
using namespace cv;

////////////////////////////////////////////////////////////////////////////////
struct MouseState
{
    int x;
    int y;
    
    int event;
};

////////////////////////////////////////////////////////////////////////////////
static MouseState mouse_state;

////////////////////////////////////////////////////////////////////////////////
void mouseCallback (int event, int x, int y, int flags, void* userdata)
{
    mouse_state.x = x;
    mouse_state.y = y;
    mouse_state.event = event;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    if (argc!=2){
        
        cerr<<"useage: ./PWP <video name>"<<endl;
        return -1;
    }
        
    
    VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        cerr << "no video cap!\n"<<endl;
        return -1;
    }
    
    
    PWPTracker *mytracker = new PWPTracker();
    namedWindow("frame", WINDOW_AUTOSIZE | CV_GUI_NORMAL);
    
    Mat oldFrame;
    Mat frame;
	Mat targetFram;
    Size s(640, 480);
    cap.read(oldFrame);
    
    if (!cap.read(oldFrame)) {
        cout<<"cant grab frame"<<endl;
        return -1;
    }
    
    resize(oldFrame, frame, s);
    
    cv::imshow ("frame", frame);
    cv::setMouseCallback ("frame", mouseCallback, NULL);

    cv::Point upper_left_point (0,0);
    cv::Point lower_right_point (0,0);
    bool bounding_box_set = false;
    while (!bounding_box_set)
    {
        cv::Mat tmp_frame = frame.clone ();
        
        cv::circle (tmp_frame, cv::Point (mouse_state.x, mouse_state.y), 5, CV_RGB (0, 0, 255), 1);
        
        if (mouse_state.event == cv::EVENT_LBUTTONDOWN)
        {
            upper_left_point.x = mouse_state.x;
            upper_left_point.y = mouse_state.y;
        }
        if (mouse_state.event == cv::EVENT_RBUTTONDOWN)
        {
            lower_right_point.x = mouse_state.x;
            lower_right_point.y = mouse_state.y;
        }
        
        cv::line (tmp_frame, upper_left_point, cv::Point (upper_left_point.x, lower_right_point.y), CV_RGB (0, 255, 0), 3);
        cv::line (tmp_frame, upper_left_point, cv::Point (lower_right_point.x, upper_left_point.y), CV_RGB (0, 255, 0), 3);
        cv::line (tmp_frame, lower_right_point, cv::Point (upper_left_point.x, lower_right_point.y), CV_RGB (0, 128, 0), 3);
        cv::line (tmp_frame, lower_right_point, cv::Point (lower_right_point.x, upper_left_point.y), CV_RGB (0, 128, 0), 3);
        
        cv::imshow ("frame", tmp_frame);
        char key = cv::waitKey (10);
        
        if (key == 'p')
            bounding_box_set = true;
    }
    
    
    int count = 0;
     
	vector<Eigen::Vector2f> corners(4);
    
    while (cap.read(oldFrame))
    {
        resize(oldFrame, frame, s);
        
        if (!mytracker->HasTarget())
        {
            PWPTracker::PWPBoundingBox bb(upper_left_point.x, upper_left_point.y, lower_right_point.x, lower_right_point.y); // hand
            mytracker->AddTarget(frame, bb);
        }
        
		mytracker->Process(frame);

		mytracker->DrawTargeLSOverlay(frame);

		mytracker->GetBoundingBox(corners);
		cv::line(frame, cvPoint(corners[0][0], corners[0][1]), cvPoint(corners[1][0], corners[1][1]), cv::Scalar(0, 0, 255), 2);
		cv::line(frame, cvPoint(corners[1][0], corners[1][1]), cvPoint(corners[2][0], corners[2][1]), cv::Scalar(0, 0, 255), 2);
		cv::line(frame, cvPoint(corners[2][0], corners[2][1]), cvPoint(corners[3][0], corners[3][1]), cv::Scalar(0, 0, 255), 2);
		cv::line(frame, cvPoint(corners[3][0], corners[3][1]), cvPoint(corners[0][0], corners[0][1]), cv::Scalar(0, 0, 255), 2);
        
        imshow("frame", frame);
        
        //char outname[200];
        //sprintf_s(outname, "e:/pwp/movie/%04i.png", count);
        //imwrite(outname, frame);
        
        count++;
        
        if (cvWaitKey(10) == 27)
            break;
    }
    
    cv::destroyAllWindows();
    return 0;
}
