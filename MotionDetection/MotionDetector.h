#pragma once

#include <opencv2/opencv.hpp>

#define DSGM_VAR_INIT (20.0*20.0)


class MotionDetector{
    
    struct SGM{
        
        cv::Vec3f  mean;
        cv::Vec3f var;
        float age;
        
        SGM():mean(cv::Vec3f(0,0,0)),var(cv::Vec3f(DSGM_VAR_INIT,DSGM_VAR_INIT,DSGM_VAR_INIT)),age(0){}
        
        bool classify(cv::Vec3b val, float theta = 1.0f) const {
            
            return
            (pow(val[0] - mean[0],(int)2) <= theta * var[0]) &&
            (pow(val[1] - mean[1],(int)2) <= theta * var[1]) &&
            (pow(val[2] - mean[2],(int)2) <= theta * var[2]);        
            
        }
        
        bool classify(cv::Vec3f val, float theta = 1.0f) const {
            
            return
            (pow(val[0] - mean[0],(int)2) <= theta * var[0]) &&
            (pow(val[1] - mean[1],(int)2) <= theta * var[1]) &&
            (pow(val[2] - mean[2],(int)2) <= theta * var[2]);        
            
        }
        
        void clear()
        {
            mean = cv::Vec3f(0,0,0);
            var = cv::Vec3f(DSGM_VAR_INIT,DSGM_VAR_INIT,DSGM_VAR_INIT);
            age = 0;
        }
    };
    
    struct DualSGM{
        
        SGM models[2];
        int active_idx;
        
        SGM& currentModel(){
            return models[active_idx];
        }
        
        const SGM& currentModel() const{
            return models[active_idx];
        }
        
        SGM& inactiveModel(){
            return models[!active_idx];
        }
        
        const SGM& inactiveModel() const{
            return models[!active_idx];
        }
        
        void copyFrom(const DualSGM& rhs){
            models[0] = rhs.models[0];
            models[1] = rhs.models[1];
            active_idx = rhs.active_idx;
        }
        
        void swapModel(){
            active_idx = !active_idx;
        }
        
        DualSGM():active_idx(0){}
        
        void clear()
        {
            models[0].clear();
            models[1].clear();
            active_idx = 0;
        }
    };
        
    typedef cv::Matx33f Hom3x3;
    
private:
    
    bool model_initialized_;
    bool pre_image_loaded;
    bool motion_compensated_;
    
    float sigma_blend_;
    float sigma_classify_;
    float sigma_decay_;
    float lambda_decay_;
    
    float max_age_;
    
    cv::Size original_size_;
    cv::Size target_size_;
    cv::Size grid_layout_;
    cv::Size patch_size_;
    
    int total_grid_no_;
    
    cv::Mat gray_image_;
    cv::Mat small_rgb_image_;
    cv::Mat small_blur_image_;
    cv::Mat small_motion_mask_;
    
    cv::Mat pre_small_gray_image_;
    cv::Mat cur_small_gray_image_;
    
    DualSGM *background_models_;
    DualSGM *warped_background_models_;
    
    
    void prepareData(const cv::Mat& in_img);
    
    // this homography directly warp prev_frame to cur_frame, with target_size
    cv::Matx33f computeBackgroundMotion(const cv::Mat& pre_img, const cv::Mat& cur_img);
    
    void updateDualSGM(const cv::Mat& in_img, const DualSGM* in_dsgm, DualSGM *out_dsgm, cv::Mat* out_motion_mask = NULL);
   
    void compensateMotion(const DualSGM* in_dsgm, DualSGM* out_dsgm, cv::Matx33f pose);
       
   void updateBackgroundModel(const cv::Mat& in_img, const Hom3x3 pose, cv::Mat* out_motion_mask = NULL);
   
public:
    MotionDetector(bool need_rotate=false);
    ~MotionDetector();
    
    void detectMotion(const cv::Mat& in_img, cv::Mat& out_motion_mask, Hom3x3 pose);
    
    
};


