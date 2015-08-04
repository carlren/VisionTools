#ifndef RELATIVE_POSE_SOLVER_H
#define RELATIVE_POSE_SOLVER_H

#include <assert.h>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include<Eigen/Dense>

#include <vector>
#include <map>

#include <iostream>

static Eigen::Matrix3d orthonormalize(const Eigen::Matrix3d & A)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svdA(A, Eigen::ComputeFullU|Eigen::ComputeFullV);
    
//    std::cout<<"me U"<<std::endl;
//    std::cout<<-svdA.matrixU()<<std::endl<<std::endl;
//    std::cout<<"me VT"<<std::endl;
//    std::cout<<-(svdA.matrixV().transpose())<<std::endl<<std::endl;
//    std::cout<<"me diag"<<std::endl;
//    std::cout<<svdA.singularValues().transpose()<<std::endl<<std::endl;
    
	return (svdA.matrixU())*(svdA.matrixV().transpose());
}


static Eigen::Isometry3d getRelativeTransformation(
        const std::vector<Eigen::Matrix4d>& pose1,
        const std::vector<Eigen::Matrix4d>& pose2)
{
    assert(pose1.size()==pose2.size());
	assert(pose1.size()>0);
        
    Eigen::MatrixXd A(16*(pose1.size()*(pose1.size()-1) / 2), 16);
    A.setZero();
    
    int counter = 0;
    
    for (unsigned int i=0; i<pose1.size(); i++) 
        for (unsigned int j=i+1; j<pose1.size(); j++) {
		Eigen::Matrix4d A1 = pose2[i].inverse();
        Eigen::Matrix4d C1 = pose1[i];
        Eigen::Matrix4d A2 = pose2[j].inverse();
		Eigen::Matrix4d C2 = pose1[j];
                
		for (int ro=0; ro<4; ro++) for (int co=0; co<4; co++) {
			for (int r=0; r<4; r++) for (int c=0; c<4; c++) 
				A(counter,c+4*r) = A1(ro,r) * C1(c,co) - A2(ro,r) * C2(c,co);
			counter++;
		}
	}

    //std::cout<<"me A"<<std::endl<<A<<std::endl<<std::endl;    
    
    // find column of V with smalest singular value
    Eigen::JacobiSVD<Eigen::MatrixXd> svdA(A, Eigen::ComputeThinV);
    Eigen::VectorXd diag = svdA.singularValues();
   
   // std::cout<<"me diag:"<<std::endl<<svdA.singularValues().transpose()<<std::endl<<std::endl;
    
    //std::cout<<"me V:"<<std::endl<<svdA.matrixV()<<std::endl<<std::endl;
    //std::cout<<"me V:"<<std::endl<<svdA.matrixV()<<std::endl<<std::endl;
    
    int min = 0;
	for (int i=1; i<16; i++) if (diag[i]<diag[min]) min=i;
    Eigen::VectorXd v = svdA.matrixV().col(min); 
    
    //std::cout<<"me v"<<std::endl<<v.transpose()<<std::endl<<std::endl;    
    
    
    // extract R ant t from vector v
    Eigen::Matrix3d Rrel;
    
    Rrel.block<1,3>(0,0) = v.segment<3>(0);
    Rrel.block<1,3>(1,0) = v.segment<3>(4);
    Rrel.block<1,3>(2,0) = v.segment<3>(8);
    Eigen::Vector3d trel;
    trel[0] = v[3];
	trel[1] = v[7];
	trel[2] = v[11];

   // std::cout<<"me R"<<std::endl<<Rrel<<std::endl<<std::endl;    
    
    // fix the scaling
	double scale = (sqrt(Rrel.row(0).dot(Rrel.row(0))) + sqrt(Rrel.row(1).dot(Rrel.row(1))) + sqrt(Rrel.row(2).dot(Rrel.row(2))) + fabs(v[15]))/4;
	Rrel *= 1.0/scale;
	trel *= 1.0/scale;
    
    // orthonormalize and take care of negative scaling
	Rrel = orthonormalize(Rrel);
	if ((Rrel.row(0).cross(Rrel.row(1))).dot(Rrel.row(2)) < 0.0) {
		Rrel = -Rrel;
		trel = -trel;
	}
    
    //std::cout<<"me T"<<std::endl<<trel.transpose()<<std::endl<<std::endl;    
    
    Eigen::Isometry3d RSE3;
    RSE3.setIdentity();
    RSE3.translate(trel);
    RSE3.rotate(Rrel);
    
    return RSE3;  
}

#endif // RELATIVE_POSE_SOLVER_H
