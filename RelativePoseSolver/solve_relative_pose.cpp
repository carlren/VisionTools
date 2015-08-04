#include "relative_pose_solver.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace  std;
using namespace Eigen;



int main(int argc, char **argv){
        
    char name[200] = "/home/carl/Work/Code/github/absolute_orientation/data/output.txt";
    
    vector<Matrix4d> itm_pose_list;
    vector<Matrix4d> vicon_pose_list;
    
    vector<Matrix4d> warped_pose_list;
    
    Matrix4d fliping; fliping.setZero();
    fliping(0,0) = -1;fliping(1,1) = 1;fliping(2,2) = 1;fliping(3,3) = 1;
    cout<<fliping;
    
    
    ifstream ifs(name); string str;
    
    for (int i=0;i<200;i++)
    {
        if(ifs.eof()) break;
        
        Matrix4d tmpMatrix;        
        for (int i=0;i<4;i++)
        {
            Vector4d tmp_vec;
            ifs>>tmp_vec[0]>>str>>tmp_vec[1]>>str>>tmp_vec[2]>>str>>tmp_vec[3];
            tmpMatrix.row(i) = tmp_vec;
        }
        itm_pose_list.push_back(tmpMatrix);
        
        for (int i=0;i<4;i++)
        {
            Vector4d tmp_vec;
            ifs>>tmp_vec[0]>>str>>tmp_vec[1]>>str>>tmp_vec[2]>>str>>tmp_vec[3];
            tmpMatrix.row(i) = tmp_vec;
        }
        vicon_pose_list.push_back(fliping*tmpMatrix);
        ifs>>str;
    }
    ifs.close();
    
    Eigen::Isometry3d result_relative_pose = getRelativeTransformation(vicon_pose_list,itm_pose_list);
    cout <<"my result solved:"<< endl <<result_relative_pose.matrix()<< endl<<endl;
    
    ofstream ofs1("/home/carl/Work/Code/github/absolute_orientation/data/aligned.txt");
    ofstream ofs2("/home/carl/Work/Code/github/absolute_orientation/data/vicon.txt");
    ofstream ofs3("/home/carl/Work/Code/github/absolute_orientation/data/itm.txt");
    
    for (int i=0;i<200;i++){
        
        warped_pose_list.push_back(result_relative_pose.matrix()*vicon_pose_list[i]);
        
        ofs1<<warped_pose_list[i].inverse().col(3).transpose()<<endl;
        ofs2<<vicon_pose_list[i].inverse().col(3).transpose()<<endl;
        ofs3<<itm_pose_list[i].inverse().col(3).transpose()<<endl;
    }
    
    ofs1.close();
    ofs2.close();
    ofs3.close();
    
    return 0;
}
