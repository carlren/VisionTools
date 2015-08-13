/*
 * 
 * @author Carl Ren
 * Created Aug. 13, 2015
 * 
 * */

#pragma once
#include <opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <vector>

class MultiHomographyStitch{
    
private:
    
    cv::Size internal_size_;
    cv::Size original_size_;
    cv::Mat ref_img_;
    cv::Mat cur_img_;
    cv::Mat motion_labeling_on_ref_img_;
    
    std::vector<cv::Matx33f> homographies_;
    
    std::vector<cv::Point2f> keypoint_cur_;
    std::vector<cv::Point2f> keypoint_ref_;
    
    std::vector<std::vector<int> > inlier_list_;
    std::vector<cv::Matx33f> homography_list_;
    
private:
    
    void computeHomographyLayersRANSAC();
    void assgnLayerLable();
    
    void visualizeMultipleHomography();
    
    
    
public:
    
    MultiHomographyStitch();
    ~MultiHomographyStitch();
    
    void processImagePair(const cv::Mat& in_img_ref, const cv::Mat& in_img_cur);
    
    
};
