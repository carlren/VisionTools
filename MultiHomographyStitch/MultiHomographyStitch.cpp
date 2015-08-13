/*
 * 
 * @author Carl Ren
 * Created Aug. 13, 2015
 * 
 * */
#include"MultiHomographyStitch.h"

using namespace std;
using namespace cv;

MultiHomographyStitch::MultiHomographyStitch(){

    internal_size_.width = 640;
    internal_size_.height = 360;
    
    motion_labeling_on_ref_img_.create(internal_size_,CV_8UC3);
    
}

MultiHomographyStitch::~MultiHomographyStitch(){}

void MultiHomographyStitch::computeHomographyLayersRANSAC()
{
    vector<Point2f>   pre_points;
    vector<Point2f>   cur_points;
    vector<uchar>       point_status;
    vector<float>         point_error;
    
    // settings for harris corner and KLT tracker
    const int maxCorners = 1000;
    const float qualityLevel = 0.01;
    const float minDistance = 3;
    const int blockSize = 4;
    const bool useHarrisDetector = false;
    const double k = 0.04;
    
    // settings for RANSAC
    const int RASAC_threshold =1;
    const int MAX_no_layers = 5;
    
    inlier_list_.clear();
    homography_list_.clear();
    
    /*
     * 
     * get features and track them
     * 
     * */
    
    cv::goodFeaturesToTrack(
                ref_img_,
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
                    ref_img_, 
                    cur_img_, 
                    pre_points, 
                    cur_points, 
                    point_status, 
                    point_error, 
                    Size(40,40),
                    5);

        // get rid of out bad matches
        for(size_t i=0; i < point_status.size(); i++) {
            if(point_status[i]) {
                keypoint_ref_.push_back(pre_points[i]);
                keypoint_cur_.push_back(cur_points[i]);
            }
        }
        
        /*
         * 
         * RANSAC style finding multiple homography layers 
         * 
         * */

        vector<int> corres_list;
        for (int i=0;i<keypoint_cur_.size();i++) corres_list.push_back(i);

        vector<Point2f> pt_left_ref = keypoint_ref_;
        vector<Point2f> pt_left_cur = keypoint_cur_;
        
        for (int i=0;i<MAX_no_layers;i++){
        
        if (corres_list.size()<10) break;
            
        Matx33f H = cv::findHomography(
                    pt_left_ref,
                    pt_left_cur, 
                    CV_RANSAC,
                    RASAC_threshold );
        
         vector<int> inlier;
         
        for (int j=0;j<pt_left_ref.size();j++){
            const Point2f& from = pt_left_ref[j];
            const Point2f& to = pt_left_cur[j];
            
            const float rx = from.x * H(0,0) + from.y * H(0,1) +H(0,2);
            const float ry = from.x * H(1,0) + from.y * H(1,1) +H(1,2);
            const float rz = from.x * H(2,0) + from.y * H(2,1) +H(2,2);
            
            const float dx = rx / rz - to.x;
            const float dy = ry / rz - to.y;
            const float sqr_error = dx * dx + dy * dy;
            
            if (sqr_error<RASAC_threshold*RASAC_threshold){
                inlier.push_back(corres_list[j]);
                
                corres_list.erase(corres_list.begin() + j);
                pt_left_ref.erase(pt_left_ref.begin() + j);
                pt_left_cur.erase(pt_left_cur.begin() + j);
            }
        }
        
        inlier_list_.push_back(inlier);
        homography_list_.push_back(H);
        }
    }
    
    for (int i=0;i<inlier_list_.size();i++)
    {
        cout<<inlier_list_[i].size()<<endl;
    }
}

void MultiHomographyStitch::assgnLayerLable(){
    
    Vec3b colors[5]={Vec3b(0,0,255),Vec3b(0,255,0),Vec3b(255,0,0),Vec3b(0,255,255),Vec3b(255,0,255)};
    motion_labeling_on_ref_img_ = Scalar(0,0,0);
    const int MAX_no_layers = 5;
    
    vector<Mat> warped_imges;
    
    for (int i=0;i<MAX_no_layers;i++){
        Mat tmp_img;
        warpPerspective(ref_img_,tmp_img,homography_list_[i],internal_size_);
        warped_imges.push_back(tmp_img);
    }
    

   
    Mat stitch_image(internal_size_,CV_8UC1);
    
    for (int y=0; y<cur_img_.rows;y++){
        for (int x=0;x<cur_img_.cols;x++){
            
            const uchar cur_val = cur_img_.at<uchar>(y,x);
            
            uchar min_diff = 255;
            int best_layer = -1;
            
            for (int i=0;i<MAX_no_layers;i++){
                
                const uchar warp_var = warped_imges[i].at<uchar>(y,x);
                uchar diff = abs(warp_var - cur_val);
                if(diff<min_diff){
                    best_layer = i;
                    min_diff = diff;
                }
            }
            
            if(best_layer>=0){
                motion_labeling_on_ref_img_.at<Vec3b>(y,x) = colors[best_layer];
                stitch_image.at<uchar>(y,x) = warped_imges[best_layer].at<uchar>(y,x);
            }

        }
    }
    
    imshow("lable", motion_labeling_on_ref_img_);
    imshow("stitch", stitch_image);
    
    imshow("cur", cur_img_);
    
        Mat diff_frame;
        absdiff(stitch_image,cur_img_,diff_frame);
        imshow("diff_new",diff_frame);
        
        absdiff(ref_img_,cur_img_,diff_frame);
        imshow("diff_old",diff_frame);
    
}

void MultiHomographyStitch::visualizeMultipleHomography()
{
    Mat ref_show;
    cvtColor(ref_img_,ref_show,CV_GRAY2BGR);

    Mat cur_show;
    cvtColor(cur_img_,cur_show,CV_GRAY2BGR);
    
    // red, green, blue, 
    Scalar colors[5]={Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0),Scalar(0,255,255),Scalar(255,0,255)};
    
    for (int i=0;i<inlier_list_.size();i++){
        for (int j=0;j<inlier_list_[i].size();j++){
            const Point2f& from = keypoint_ref_[inlier_list_[i][j]];
            const Point2f& to = keypoint_cur_[inlier_list_[i][j]];
            
            circle(ref_show,from,2,colors[i],2);
            circle(cur_show,to,2,colors[i],2);
            
        }
    }
    
    
    imshow("from",ref_show);
    imshow("to",cur_show);

    
}

void MultiHomographyStitch::processImagePair(const Mat &in_img_ref, const Mat &in_img_cur){

    original_size_.width = in_img_cur.cols;
    original_size_.height = in_img_cur.rows;
    resize(in_img_ref,ref_img_,internal_size_);
    resize(in_img_cur,cur_img_,internal_size_);
    
    cvtColor(ref_img_,ref_img_,CV_BGR2GRAY);
    cvtColor(cur_img_,cur_img_,CV_BGR2GRAY);
    
    computeHomographyLayersRANSAC();
    
    assgnLayerLable();
    
    visualizeMultipleHomography();
    waitKey(0);
}
