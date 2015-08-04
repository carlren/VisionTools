#include "relative_pose_solver.h"

#include <vector>
#include <iostream>
#include <fstream>
//#include"CalibStereo.h"

using namespace std;
using namespace Eigen;

Eigen::Quaterniond randQuaternion()
{
    Eigen::Vector4d v;
    float sum=0;
    for (int i=0;i<4;i++) {
        v(i) = rand();
        sum += v(i)*v(i);
    }
    
    for (int i=0;i<4;i++)
        v(i) /=sqrt(sum);
    
    return Eigen::Quaterniond(v);
}


int main()
{
    const int no_poses=200;
    
    Eigen::Quaterniond d_rotation(sqrt(0.1),sqrt(0.2),sqrt(0.3),sqrt(0.4));
    Eigen::Vector3d d_translation(4,5,6);
    Eigen::Isometry3d pose;
    pose.setIdentity();

    Eigen::Quaterniond relative_rotation(sqrt(0.4),sqrt(0.3),sqrt(0.2),sqrt(0.1));
    Eigen::Vector3d relative_translation(1,2,3);
    Eigen::Isometry3d relative_pose;
    relative_pose.setIdentity();
    relative_pose.translate(relative_translation);
    relative_pose.rotate(relative_rotation);
    
    vector<Eigen::Matrix4d> pose1,pose2;
    
//    vector<TooN::SE3<double> > olaf_pose1, olaf_pose2;
    
    for (int i=0;i<no_poses;i++)
    {
        Eigen::Matrix4d new_pose = relative_pose.matrix()*pose.matrix();
       
        pose1.push_back(pose.matrix());
        pose2.push_back(new_pose);
        
//        TooN::Matrix<3,3> o_R1, o_R2;
       
//        for(int r=0;r<3;r++) for(int c=0;c<3;c++) 
//        {
//            o_R1[r][c] = pose.matrix()(r,c);
//            o_R2[r][c] = new_pose(r,c);
//        }
//        TooN::SO3<double> o_SO3_1(o_R1);
//        TooN::SO3<double> o_SO3_2(o_R2);
        
        
//        TooN::Vector<3> o_T1, o_T2;
//        for (int r=0;r<3;r++)
//        {
//            o_T1[r] = pose.matrix()(r,3);
//            o_T2[r] = new_pose(r,3);
//        }
        
        
//        TooN::SE3<double> se3_1(o_SO3_1,o_T1);
//        TooN::SE3<double> se3_2(o_SO3_2,o_T2);
        
//        olaf_pose1.push_back(se3_1);
//        olaf_pose2.push_back(se3_2);
        
        pose.rotate(randQuaternion());
        pose.translate(d_translation);
        
    }
        
    Eigen::Isometry3d result_relative_pose = getRelativeTransformation(pose1, pose2);
//    TooN::SE3<double> olaf_result_pose =  CalibStereo::getRelativePose_differentFrames(olaf_pose1,olaf_pose2);
    
    cout<<"relative matrix:"<< endl << relative_pose.matrix() << endl<<endl;
    cout <<"my result solved:"<< endl <<result_relative_pose.matrix()<< endl<<endl;
//    cout <<"olaf result solved:"<< endl <<olaf_result_pose<< endl<<endl;
    
    ofstream ofs1("/home/carl/Work/Code/github/absolute_orientation/data/aligned.txt");
    ofstream ofs2("/home/carl/Work/Code/github/absolute_orientation/data/pose1.txt");
    ofstream ofs3("/home/carl/Work/Code/github/absolute_orientation/data/pose2.txt");
    
    vector<Matrix4d> warped_pose_list;
    
    for (int i=0;i<no_poses;i++){
        
        warped_pose_list.push_back(relative_pose.matrix() * pose1[i]);
        
        ofs1<<warped_pose_list[i].inverse().col(3).transpose()<<endl;
        ofs2<<pose1[i].inverse().col(3).transpose()<<endl;
        ofs3<<pose2[i].inverse().col(3).transpose()<<endl;
    }
    
    ofs1.close();
    ofs2.close();
    ofs3.close();
    
    return 0;
}

