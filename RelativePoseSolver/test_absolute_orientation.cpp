#include "absolute_orientation_horn.h"
#include "relative_pose_solver.h"
#include <iostream>

using namespace std;

int main()
{
    const int no_points=200;
    
    float tx, ty, tz;
    float theta=0, r=2;
    
    Eigen::Quaterniond random_rotation(sqrt(0.1),sqrt(0.2),sqrt(0.3),sqrt(0.4));
    Eigen::Vector3d random_translation(rand(),rand(),rand());
    Eigen::Isometry3d random_pose;
    random_pose.setIdentity();
    random_pose.rotate(random_rotation);
    random_pose.translate(random_translation);
    Eigen::Isometry3d result_pose;
    
    Eigen::MatrixXd pts1(3,no_points);
    Eigen::MatrixXd pts2(3,no_points);
    
    for (int i=0;i<no_points;i++)
    {
            tx = r*sin(theta);
            ty = r*cos(theta);
            tz = r;
            r += 0.1;
            
            Eigen::Vector3d pt(tx,ty,tz);
            pts1.col(i) = pt;
            pts2.col(i) = random_pose.rotation()*pt + random_pose.translation();
    }
    
    absolute_orientation_horn(pts1,pts2,&result_pose);
    
    cout<<"relative matrix:"<< endl << random_pose.matrix() << endl<<endl;
    cout <<"result solved:"<< endl << result_pose.matrix() << endl<<endl;
    
    float error_sum=0;
    for (int i=0;i<no_points;i++)
    {
        Eigen::Vector3d diff_pt = (result_pose.rotation() * pts1.col(i) + result_pose.translation() )- pts2.col(i);
        error_sum += sqrt(diff_pt.dot(diff_pt));
    }
    
    cout<<"average error: "<<error_sum/no_points<<endl;
    
    return 0;
}

