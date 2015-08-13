#include "MotionDetector.h"
#include <iostream>
#include<math.h>

using namespace  std;
using namespace cv;


MotionDetector::~MotionDetector(){}

MotionDetector::MotionDetector(bool need_rotate){
    
    target_size_.width = 320;
    target_size_.height = 180;
    
    patch_size_.width = 1;
    patch_size_.height = 1;
    
    model_initialized_ = false;
    pre_image_loaded = false;
    motion_compensated_ = false;
    
    max_age_ = 30;
    sigma_blend_ = 2.0f;
    sigma_classify_ = 4.0f;
    sigma_decay_ = 50*50;
    lambda_decay_ = 0.001;
    
    
    if(need_rotate) swap(target_size_.width, target_size_.height);

    grid_layout_.width = target_size_.width / patch_size_.width;
    grid_layout_.height = target_size_.height / patch_size_.height;
    
    total_grid_no_ = grid_layout_.width * grid_layout_.height;
    
    background_models_ = new DualSGM[total_grid_no_];
    warped_background_models_ = new DualSGM[total_grid_no_];
    
    for (int i=0;i<total_grid_no_;i++){
        background_models_[i].clear();
        warped_background_models_[i].clear();
    }
    
    small_motion_mask_.create(target_size_,CV_8UC1);

}

// done
void MotionDetector::prepareData(const Mat &in_img){
    
    resize(in_img,small_rgb_image_,target_size_);
    cvtColor(small_rgb_image_,cur_small_gray_image_,CV_BGR2GRAY);
    GaussianBlur(small_rgb_image_, small_blur_image_,Size(7,7),0);
    medianBlur(small_blur_image_,small_blur_image_,5);
    
      if (!pre_image_loaded){
        original_size_.width = in_img.cols;
        original_size_.height = in_img.rows;
        cur_small_gray_image_.copyTo(pre_small_gray_image_);
        pre_image_loaded = true;
    }
}

// done
//#define DEBUG_BG_MOTION
Matx33f MotionDetector::computeBackgroundMotion(const Mat &pre_img, const Mat &cur_img){

        cv::Mat n;
     
        std::vector<cv::Point2f>   pre_points;
        std::vector<cv::Point2f>   cur_points;
        std::vector<uchar> point_status;
        std::vector<float>         point_error;
        
        const int maxCorners = 300;
        const float qualityLevel = 0.01;
        const float minDistance = 5;
        const int blockSize = 4;
        const bool useHarrisDetector = false;
        const double k = 0.04;
        
        cv::goodFeaturesToTrack(
                    pre_img,
                    pre_points, 
                    maxCorners, 
                    qualityLevel, 
                    minDistance, 
                    noArray(),
                    blockSize, 
                    useHarrisDetector, 
                    k);
    
        if(pre_points.size() >= 1) {
            
            cv::calcOpticalFlowPyrLK(
                        pre_img, 
                        cur_img, 
                        pre_points, 
                        cur_points, 
                        point_status, 
                        point_error, 
                        Size(20,20), 
                        5);
    
            vector <Point2f> prev_corner2, cur_corner2;
            n = cur_img.clone();
    
            // weed out bad matches
            for(size_t i=0; i < point_status.size(); i++) {
                if(point_status[i]) {
                    prev_corner2.push_back(pre_points[i]);
                    cur_corner2.push_back(cur_points[i]);
                }
            }
            
            cv::Mat H = cv::findHomography(
                        prev_corner2,
                        cur_corner2, 
                        CV_RANSAC,
                        1);
            
#ifdef DEBUG_BG_MOTION            
            warpPerspective(cur_img, n, H, target_size_, INTER_LINEAR | WARP_INVERSE_MAP);
            
            Mat show_frame;
            cvtColor(n,show_frame,CV_GRAY2BGR);
            for (size_t i=0;i<cur_corner2.size();i++){
                cv::arrowedLine(show_frame,cur_corner2[i], prev_corner2[i],Scalar(0,255,0));
            }
            
            Mat err_before;//(n.rows, n.cols,CV_8UC1);
            Mat err_after;//(n.rows,n.cols,CV_8UC1);
            
            absdiff(pre_img,cur_img,err_before);
            absdiff(pre_img,n,err_after);
           
            imshow("err_before",err_before);
            imshow("err_after",err_after);
            
//            moveWindow("err_before",100,200);
//            moveWindow("err_after",400,200);
            
            imshow("tracking", show_frame);
#endif                        
            
            return Matx33f(H);
        } else {
            printf("NO matching points");
            return Matx33f::eye();
        }
 
        
}

#define MIX_DSGM_WARP
void MotionDetector::compensateMotion(const DualSGM *in_dsgm, DualSGM *out_dsgm, Matx33f pose){

    for (int i=0; i<total_grid_no_; i++)  out_dsgm[i].clear();
    
    for (int r=0, count = 0;r<grid_layout_.height;r++){
        for (int c=0;c<grid_layout_.width;c++, count++){
            
            DualSGM& current_dsgm_out = out_dsgm[count];
            
            float X = patch_size_.width * c + patch_size_.width * 0.5;
            float Y = patch_size_.height * r + patch_size_.height * 0.5;
            
            float normalizer = pose(2,0) * X + pose(2,1) * Y + pose(2,2);
            float X_new = (pose(0,0) * X + pose(0,1) * Y + pose(0,2)) / normalizer;
            float Y_new = (pose(1,0) * X + pose(1,1) * Y + pose(1,2)) / normalizer;
            
            float c_new = X_new / patch_size_.width;
            float r_new = Y_new / patch_size_.height;
            
            int c_idx = floor(c_new);
            int r_idx = floor(r_new);
            
            float dr = r_new - ((float)r_idx + 0.5);
            float dc = c_new - ((float)c_idx + 0.5);
            
            float area_h;
            float area_v;
            float area_hv;
            float area_self;
            
            Vec3f tmp_var[2] = {Vec3f(0,0,0),Vec3f(0,0,0)};
            Vec3f tmp_mean[2] = {Vec3f(0,0,0),Vec3f(0,0,0)};
            float tmp_age[2] = {0,0};
            float total_area = 0;      
            
            
            ///////////////////////////////////////////////////////////
            //
            //
            // first we warp the mean
            //
            //
            ///////////////////////////////////////////////////////////            
            {
      
#ifdef MIX_DSGM_WARP
                ///////////////////////////////////////////////////////////
                // horizontal neighbor
                ///////////////////////////////////////////////////////////
                if (dc!=0){
                    
                    int c_idx_new = c_idx;
                    int r_idx_new = r_idx;
                    c_idx_new += dc > 0 ? 1 : -1;
                    
                    if (c_idx_new >=0 && c_idx_new < grid_layout_.width && r_idx_new >=0 && r_idx_new < grid_layout_.height){ 
                        const DualSGM& selected_dsgm = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new];
                        area_h = abs(dc) * (1.0 - abs(dr));
                        tmp_mean[0] += area_h * selected_dsgm.currentModel().mean;
                        tmp_mean[1] += area_h * selected_dsgm.inactiveModel().mean;
                        tmp_age[0] += area_h * selected_dsgm.currentModel().age;
                        tmp_age[1] += area_h * selected_dsgm.inactiveModel().age;
                        total_area += area_h;
                    }    
                }
                
                ///////////////////////////////////////////////////////////
                // vertical neighbor
                ///////////////////////////////////////////////////////////
                if (dr!=0){
                    
                    int c_idx_new = c_idx;
                    int r_idx_new = r_idx;
                    r_idx_new += dr > 0 ? 1 : -1;
                    
                    if (c_idx_new >=0 && c_idx_new < grid_layout_.width && r_idx_new >=0 && r_idx_new < grid_layout_.height){ 
                        const DualSGM& selected_dsgm = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new];
                        area_v = abs(dr) * (1.0 - abs(dc));
                        tmp_mean[0] += area_v * selected_dsgm.currentModel().mean;
                        tmp_mean[1] += area_v * selected_dsgm.inactiveModel().mean;
                        tmp_age[0] += area_v * selected_dsgm.currentModel().age;
                        tmp_age[1] += area_v * selected_dsgm.inactiveModel().age;
                        total_area += area_v;
                    }
                }
                
                ///////////////////////////////////////////////////////////
                // vertical+horizontal neighbor
                ///////////////////////////////////////////////////////////
                if (dr!=0 && dc!=0){
                    
                    int c_idx_new = c_idx;
                    int r_idx_new = r_idx;
                    c_idx_new += dc > 0 ? 1 : -1;
                    r_idx_new += dr > 0 ? 1 : -1;
                    
                    if (c_idx_new >=0 && c_idx_new < grid_layout_.width && r_idx_new >=0 && r_idx_new < grid_layout_.height){ 
                        const DualSGM& selected_dsgm = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new];
                        area_hv = abs(dr) * abs(dc);
                        tmp_mean[0] += area_hv * selected_dsgm.currentModel().mean;
                        tmp_mean[1] += area_hv * selected_dsgm.inactiveModel().mean;
                        tmp_age[0] += area_hv * selected_dsgm.currentModel().age;
                        tmp_age[1] += area_hv * selected_dsgm.inactiveModel().age;
                        total_area += area_hv;
                    }
                }
#endif                
                ///////////////////////////////////////////////////////////
                // self
                ///////////////////////////////////////////////////////////
                
                if (c_idx >=0 && c_idx < grid_layout_.width && r_idx >=0 && r_idx < grid_layout_.height){ 
                    const DualSGM& selected_dsgm = in_dsgm[r_idx * grid_layout_.width + c_idx];
                    area_self = (1.0 - abs(dr)) * (1.0 - abs(dc));
                    tmp_mean[0] += area_self * selected_dsgm.currentModel().mean;
                    tmp_mean[1] += area_self * selected_dsgm.inactiveModel().mean;
                    tmp_age[0] += area_self * selected_dsgm.currentModel().age;
                    tmp_age[1] += area_self * selected_dsgm.inactiveModel().age;
                    total_area += area_self;
                    
                }
                
                if (total_area>0)
                {
                    tmp_mean[0] /= total_area;
                    tmp_mean[1] /= total_area;
                    tmp_age[0] /= total_area;
                    tmp_age[1] /= total_area;
                }
                
                
            }
            
            
            ///////////////////////////////////////////////////////////
            //
            //
            // then we warp the variance
            //
            //
            ///////////////////////////////////////////////////////////
            {
                
#ifdef MIX_DSGM_WARP                
                ///////////////////////////////////////////////////////////
                // horizontal neighbor
                ///////////////////////////////////////////////////////////
                if (dc!=0){
                    
                    int c_idx_new = c_idx;
                    int r_idx_new = r_idx;
                    c_idx_new += dc > 0 ? 1 : -1;
                    
                    if (c_idx_new >=0 && c_idx_new < grid_layout_.width && r_idx_new >=0 && r_idx_new < grid_layout_.height){ 

                        const SGM& selected_active = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new].currentModel();
                        const Vec3f diff_active = selected_active.mean - tmp_mean[0];                        
                        tmp_var[0][0] +=area_h * (selected_active.var[0] + pow(diff_active[0],(int)2));
                        tmp_var[0][1] +=area_h * (selected_active.var[1] + pow(diff_active[1],(int)2));
                        tmp_var[0][2] +=area_h * (selected_active.var[2] + pow(diff_active[2],(int)2));
                        
                        const SGM& selected_inactive = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new].inactiveModel();
                        const Vec3f diff_inactive = selected_inactive.mean - tmp_mean[1]; 
                        tmp_var[1][0] +=area_h * (selected_inactive.var[0] + pow(diff_inactive[0],(int)2));
                        tmp_var[1][1] +=area_h * (selected_inactive.var[1] + pow(diff_inactive[1],(int)2));
                        tmp_var[1][2] +=area_h * (selected_inactive.var[2] + pow(diff_inactive[2],(int)2));
                        
                    }    
                }
                
                ///////////////////////////////////////////////////////////
                // vertical neighbor
                ///////////////////////////////////////////////////////////
                if (dr!=0){
                    
                    int c_idx_new = c_idx;
                    int r_idx_new = r_idx;
                    r_idx_new += dr > 0 ? 1 : -1;
                    
                    if (c_idx_new >=0 && c_idx_new < grid_layout_.width && r_idx_new >=0 && r_idx_new < grid_layout_.height){ 

                        const SGM& selected_active = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new].currentModel();
                        const Vec3f diff_active = selected_active.mean - tmp_mean[0];                        
                        tmp_var[0][0] +=area_v * (selected_active.var[0] + pow(diff_active[0],(int)2));
                        tmp_var[0][1] +=area_v * (selected_active.var[1] + pow(diff_active[1],(int)2));
                        tmp_var[0][2] +=area_v * (selected_active.var[2] + pow(diff_active[2],(int)2));
                        
                        const SGM& selected_inactive = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new].inactiveModel();
                        const Vec3f diff_inactive = selected_inactive.mean - tmp_mean[1]; 
                        tmp_var[1][0] +=area_v * (selected_inactive.var[0] + pow(diff_inactive[0],(int)2));
                        tmp_var[1][1] +=area_v * (selected_inactive.var[1] + pow(diff_inactive[1],(int)2));
                        tmp_var[1][2] +=area_v * (selected_inactive.var[2] + pow(diff_inactive[2],(int)2));
                        
                    }
                }
                
                ///////////////////////////////////////////////////////////
                // vertical+horizontal neighbor
                ///////////////////////////////////////////////////////////
                if (dr!=0 && dc!=0){
                    
                    int c_idx_new = c_idx;
                    int r_idx_new = r_idx;
                    c_idx_new += dc > 0 ? 1 : -1;
                    r_idx_new += dr > 0 ? 1 : -1;
                    
                    if (c_idx_new >=0 && c_idx_new < grid_layout_.width && r_idx_new >=0 && r_idx_new < grid_layout_.height){ 
                        
                        const SGM& selected_active = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new].currentModel();
                        const Vec3f diff_active = selected_active.mean - tmp_mean[0];                        
                        tmp_var[0][0] +=area_hv * (selected_active.var[0] + pow(diff_active[0],(int)2));
                        tmp_var[0][1] +=area_hv * (selected_active.var[1] + pow(diff_active[1],(int)2));
                        tmp_var[0][2] +=area_hv * (selected_active.var[2] + pow(diff_active[2],(int)2));
                        
                        const SGM& selected_inactive = in_dsgm[r_idx_new * grid_layout_.width + c_idx_new].inactiveModel();
                        const Vec3f diff_inactive = selected_inactive.mean - tmp_mean[1]; 
                        tmp_var[1][0] +=area_hv * (selected_inactive.var[0] + pow(diff_inactive[0],(int)2));
                        tmp_var[1][1] +=area_hv * (selected_inactive.var[1] + pow(diff_inactive[1],(int)2));
                        tmp_var[1][2] +=area_hv * (selected_inactive.var[2] + pow(diff_inactive[2],(int)2));

                    }
                }
#endif                
                ///////////////////////////////////////////////////////////
                // self
                ///////////////////////////////////////////////////////////
                
                if (c_idx >=0 && c_idx < grid_layout_.width && r_idx >=0 && r_idx < grid_layout_.height){ 

                    const SGM& selected_active = in_dsgm[r_idx * grid_layout_.width + c_idx].currentModel();
                    const Vec3f diff_active = selected_active.mean - tmp_mean[0];                        
                    tmp_var[0][0] +=area_self * (selected_active.var[0] + pow(diff_active[0],(int)2));
                    tmp_var[0][1] +=area_self * (selected_active.var[1] + pow(diff_active[1],(int)2));
                    tmp_var[0][2] +=area_self * (selected_active.var[2] + pow(diff_active[2],(int)2));
                    
                    const SGM& selected_inactive = in_dsgm[r_idx * grid_layout_.width + c_idx].inactiveModel();
                    const Vec3f diff_inactive = selected_inactive.mean - tmp_mean[1]; 
                    tmp_var[1][0] +=area_self * (selected_inactive.var[0] + pow(diff_inactive[0],(int)2));
                    tmp_var[1][1] +=area_self * (selected_inactive.var[1] + pow(diff_inactive[1],(int)2));
                    tmp_var[1][2] +=area_self * (selected_inactive.var[2] + pow(diff_inactive[2],(int)2));
                    
                }
                
                if (total_area>0)
                {
                    tmp_var[0] /= total_area;
                    tmp_var[1] /= total_area;
                }
                
            }
            
            tmp_var[0][0] = MAX(tmp_var[0][0],DSGM_VAR_INIT);
            tmp_var[0][1] = MAX(tmp_var[0][1],DSGM_VAR_INIT);
            tmp_var[0][2] = MAX(tmp_var[0][2],DSGM_VAR_INIT);
            
            tmp_var[1][0] = MAX(tmp_var[1][0],DSGM_VAR_INIT);
            tmp_var[1][1] = MAX(tmp_var[1][1],DSGM_VAR_INIT);
            tmp_var[1][2] = MAX(tmp_var[1][2],DSGM_VAR_INIT);
            
            // writing tmp stuffs back to the model
            
            current_dsgm_out.currentModel().mean = tmp_mean[0];
            current_dsgm_out.inactiveModel().mean = tmp_mean[1];
            
            current_dsgm_out.currentModel().age = tmp_age[0];
            current_dsgm_out.inactiveModel().age = tmp_age[1];
                        
            
            if (c_idx <1 || c_idx >= grid_layout_.width-1 || r_idx <1 || r_idx >= grid_layout_.height-1){
                
                current_dsgm_out.currentModel().var = Vec3f(DSGM_VAR_INIT,DSGM_VAR_INIT,DSGM_VAR_INIT);
                current_dsgm_out.inactiveModel().var = Vec3f(DSGM_VAR_INIT,DSGM_VAR_INIT,DSGM_VAR_INIT);
                                
                current_dsgm_out.currentModel().age = 0;
                current_dsgm_out.inactiveModel().age = 0;                
                
            }else{
                
                current_dsgm_out.currentModel().var = tmp_var[0];
                current_dsgm_out.inactiveModel().var = tmp_var[1];

//                float decay_factor = exp(-lambda_decay_ * MAX(0,tmp_var[0]-sigma_decay_));
//                current_dsgm_out.currentModel().age =  MIN(decay_factor * current_dsgm_out.currentModel().age, max_age_) ;
                
//                decay_factor = exp(-lambda_decay_ * MAX(0,tmp_var[1]-sigma_decay_));
//                current_dsgm_out.inactiveModel().age =  MIN(decay_factor * current_dsgm_out.inactiveModel().age, max_age_) ;
            }
            
            if (current_dsgm_out.inactiveModel().age>current_dsgm_out.currentModel().age)
                current_dsgm_out.swapModel();
        }
    }
    
}


void MotionDetector::updateDualSGM(const Mat &in_img, const DualSGM *in_dsgm, DualSGM *out_dsgm, Mat *out_motion_mask){
    
    for (int r=0, count = 0;r<grid_layout_.height;r++){
        for (int c=0;c<grid_layout_.width;c++, count++){
            
            const DualSGM& current_dsgm_in = in_dsgm[count];
            DualSGM& current_dsgm_out = out_dsgm[count];
            
            
            const int y_base = r * patch_size_.height;
            const int x_base = c * patch_size_.width;
        
            Vec3f mean_tmp = 0;
            float no_points = 0;
            
            // compute mean of current image block
            for (int i = 0; i<patch_size_.height; i++){
                for(int j = 0; j<patch_size_.width; j++){
                    mean_tmp += in_img.at<Vec3b>(y_base+i,x_base+j);
                    no_points++;
                }
            }
            mean_tmp /= no_points;
            
            current_dsgm_out.copyFrom(current_dsgm_in);
            
            const SGM& active_sgm_in = current_dsgm_in.currentModel();
            const SGM& inactive_sgm_in = current_dsgm_in.inactiveModel();
                        
            SGM& active_sgm_out = current_dsgm_out.currentModel();
            SGM& inactive_sgm_out = current_dsgm_out.inactiveModel();
                    
            if (active_sgm_in.age == 0 && inactive_sgm_in.age == 0){
                active_sgm_out.mean = mean_tmp;
                active_sgm_out.var = Vec3f(DSGM_VAR_INIT,DSGM_VAR_INIT,DSGM_VAR_INIT);
                active_sgm_out.age = 1;
                continue;
            }
            
            SGM* sgm_to_update = &active_sgm_out;
            
            /////////////////////////////////////////////
            //
            // update mean fist here
            // with the current mean, decide with model to update
            //
            /////////////////////////////////////////////
            
            // current mean fits active model
            if (active_sgm_in.classify(mean_tmp,sigma_blend_))
            {
                active_sgm_out.age = active_sgm_in.age + 1;
                float weight = 1.0 / active_sgm_out.age;
                active_sgm_out.mean = (1-weight) * active_sgm_in.mean + weight * mean_tmp;
                sgm_to_update = &active_sgm_out;
            } 
            // current mean fits inactive model
            else if (inactive_sgm_in.classify(mean_tmp,sigma_blend_))
            {
                inactive_sgm_out.age = inactive_sgm_in.age + 1;
                float weight = 1.0 / inactive_sgm_out.age;
                inactive_sgm_out.mean = (1-weight) * inactive_sgm_in.mean + weight * mean_tmp;
                sgm_to_update = &inactive_sgm_out;
            }
            // current mean doesn't fit anything, re-initialzie inactive model
            else 
            {
                inactive_sgm_out.mean = mean_tmp;
                inactive_sgm_out.age = 1;
                sgm_to_update = &inactive_sgm_out;
            }

            
            /////////////////////////////////////////////
            //
            // then use the updated mean to update variance
            //
            /////////////////////////////////////////////
            
            Vec3f var_tmp = Vec3f(DSGM_VAR_INIT,DSGM_VAR_INIT,DSGM_VAR_INIT);      
            for (int i = 0; i<patch_size_.height; i++){
                for(int j = 0; j<patch_size_.width; j++){
                    Vec3f diff =  Vec3f(in_img.at<Vec3b>(y_base+i,x_base+j)) - sgm_to_update->mean;
                    var_tmp[0] = MAX(pow(diff[0], (int)2),var_tmp[0]);
                    var_tmp[1] = MAX(pow(diff[1], (int)2),var_tmp[1]);
                    var_tmp[2] = MAX(pow(diff[2], (int)2),var_tmp[2]);
                }
            }
            
            if(sgm_to_update->age == 1){
                sgm_to_update->var = var_tmp;
            }else{
                float weight = 1.0f /  sgm_to_update->age;
                sgm_to_update->var = (1-weight) * sgm_to_update->var + weight * var_tmp;
                sgm_to_update->age = MIN(sgm_to_update->age, max_age_);    
            }
            
            if (active_sgm_out.age < inactive_sgm_out.age) {
                current_dsgm_out.swapModel();
                //cout<<"swap!"<<endl;
            }
            
        }    
    }
    
}

#define SHOW_DSGM_MODEL
void MotionDetector::updateBackgroundModel(const Mat &in_img, const Hom3x3 pose, Mat* out_motion_mask)
{
    // preprocess image by smoothing
    prepareData(in_img);
    Matx33f warp = computeBackgroundMotion(pre_small_gray_image_,cur_small_gray_image_); 
    compensateMotion(background_models_, warped_background_models_,warp.inv());
    updateDualSGM(small_blur_image_,warped_background_models_,background_models_, out_motion_mask);
    
    cur_small_gray_image_.copyTo(pre_small_gray_image_);
    
#ifdef SHOW_DSGM_MODEL    
//// debug   gaussian means
    Mat sgm_image(grid_layout_,CV_8UC3);
    for (int r=0, count = 0;r<grid_layout_.height;r++){
        for (int c=0;c<grid_layout_.width;c++, count++){
            DualSGM& bm = background_models_[count];
            SGM& current_model = bm.currentModel();
            sgm_image.at<Vec3b>(r,c) = current_model.mean;
        }
    }
    imshow("sgm_mean", sgm_image);
    
    Mat sgm_warp_image(grid_layout_,CV_8UC3);
    for (int r=0, count = 0;r<grid_layout_.height;r++){
        for (int c=0;c<grid_layout_.width;c++, count++){
            DualSGM& bm = background_models_[count];
            SGM& current_model = bm.inactiveModel();
            sgm_warp_image.at<Vec3b>(r,c) = current_model.mean;
        }
    }
    imshow("warped_mean", sgm_warp_image);
#endif
    
}



void MotionDetector::detectMotion(const Mat &in_img, Mat &out_motion_mask, Hom3x3 pose){
    
    updateBackgroundModel(in_img,pose);

    resize(in_img,out_motion_mask,target_size_);
    
    small_motion_mask_ = Scalar(0);
    for (int r=0, idx=0;r<grid_layout_.height;r++){
        for (int c=0;c<grid_layout_.width;c++,idx++){
        
            const SGM& current_sgm = background_models_[idx].currentModel();
            
            const int y_base = r * patch_size_.height;
            const int x_base = c * patch_size_.width;
            
            for (int i = 0; i<patch_size_.height; i++){
                for(int j = 0; j<patch_size_.width; j++){
                    const Vec3b val = small_blur_image_.at<Vec3b>(y_base+i,x_base+j);
                    
                    if (!current_sgm.classify(val,sigma_classify_))
                        out_motion_mask.at<Vec3b>(y_base+i,x_base+j) = Vec3b(0,0,255);
                    
                }
            }
            
        }
    }
    
}
