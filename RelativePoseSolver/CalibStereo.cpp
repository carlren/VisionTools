#include "CalibStereo.h"
#include <TooN/SVD.h>
#include <TooN/SymEigen.h>
#include <assert.h>

//#define DEBUG_STEREO

static TooN::Matrix<3,3> orthonormalize(const TooN::Matrix<3,3> & A)
{
	TooN::SVD<3> svdA(A);
	return svdA.get_U()*svdA.get_VT();
}

TooN::SE3<double> CalibStereo::getRelativePose_commonFrame(const std::vector<TooN::SE3<double> > & pose1, const std::vector<TooN::SE3<double> > & pose2)
{
	assert(pose1.size()==pose2.size());

	TooN::Matrix<> A(pose1.size()*12, 12);
	TooN::Vector<> b(pose1.size()*12);

	for (int r = 0; r < A.num_rows(); r++) 
		for (int c = 0; c < A.num_cols(); c++) A(r,c) = 0.0;

	std::vector<TooN::SE3<double> >::const_iterator it1 = pose1.begin();
	std::vector<TooN::SE3<double> >::const_iterator it2 = pose2.begin();
	int counter = 0;
	for (; it1 != pose1.end(); ++it1, ++it2, ++counter) {
		A.slice(counter*12, 0, 3,3) = it1->get_rotation().get_matrix().T();
		A.slice(counter*12+3, 3, 3,3) = it1->get_rotation().get_matrix().T();
		A.slice(counter*12+6, 6, 3,3) = it1->get_rotation().get_matrix().T();
		A.slice(counter*12+9, 0, 1,3) = it1->get_translation().as_row();
		A(counter*12+9, 9) = 1.0;
		A.slice(counter*12+10, 3, 1,3) = it1->get_translation().as_row();
		A(counter*12+10, 10) = 1.0;
		A.slice(counter*12+11, 6, 1,3) = it1->get_translation().as_row();
		A(counter*12+11, 11) = 1.0;
		b.slice(counter*12, 3) = it2->get_rotation().get_matrix()[0];
		b.slice(counter*12+3, 3) = it2->get_rotation().get_matrix()[1];
		b.slice(counter*12+6, 3) = it2->get_rotation().get_matrix()[2];
		b.slice(counter*12+9, 3) = it2->get_translation();
	} 

	#ifdef DEBUG_STEREO
	std::cerr << "getRelativePose_commonFrame Matrix A:" << std::endl << A;
	std::cerr << "Vector b:" << std::endl << b << std::endl;
	#endif
	TooN::SVD<> svdA(A);
	TooN::Vector<12> v = svdA.backsub(b);

	TooN::Matrix<3,3> Rrel;
	TooN::Vector<3> trel;
	Rrel.slice<0,0,1,3>() = v.slice<0,3>().as_row();
	Rrel.slice<1,0,1,3>() = v.slice<3,3>().as_row();
	Rrel.slice<2,0,1,3>() = v.slice<6,3>().as_row();
	trel = v.slice<9,3>();

	return TooN::SE3<double>(orthonormalize(Rrel), trel);
}

TooN::SE3<double> CalibStereo::getRelativePose_commonFrame(const std::vector<TooN::SE3<double> > & pose1, const std::vector<TooN::SE3<double> > & pose2, const std::map<int,int> & synclist)
{
	std::vector<TooN::SE3<double> > poses1;
	std::vector<TooN::SE3<double> > poses2;
	for (std::map<int,int>::const_iterator it = synclist.begin();
	     it != synclist.end(); ++it) {
		poses1.push_back(pose1[it->first]);
		poses2.push_back(pose2[it->second]);
	}
	if (poses1.size()>0) return CalibStereo::getRelativePose_commonFrame(poses1, poses2);
	return TooN::SE3<double>();
}

TooN::SO3<double> CalibStereo::getRelativePose_differentFrames(
		const std::vector<TooN::SO3<double> > & pose1,
		const std::vector<TooN::SO3<double> > & pose2)
{
	assert(pose1.size()==pose2.size());
	assert(pose1.size()>0);
	TooN::Matrix<> A(9*(pose1.size()*(pose1.size()-1) / 2), 9);
	for (int r=0; r<A.num_rows(); r++) for (int c=0; c<A.num_cols(); c++) A(r,c) = 0.0;
	int counter = 0;
	for (unsigned int i=0; i<pose1.size(); i++) for (unsigned int j=i+1; j<pose1.size(); j++) {
		const TooN::Matrix< 3, 3 > & A1 = pose2[i].inverse().get_matrix();
		const TooN::Matrix< 3, 3 > & C1 = pose1[i].get_matrix();
		const TooN::Matrix< 3, 3 > & A2 = pose2[j].inverse().get_matrix();
		const TooN::Matrix< 3, 3 > & C2 = pose1[j].get_matrix();

		for (int ro=0; ro<3; ro++) for (int co=0; co<3; co++) {
			for (int r=0; r<3; r++) for (int c=0; c<3; c++) 
				A(counter,c+3*r) = A1(ro,r) * C1(c,co) - A2(ro,r) * C2(c,co);
			counter++;
		}
	}

	TooN::SVD<> svdA(A);
	TooN::Vector<9> diag = svdA.get_diagonal();
	int min = 0;
	for (int i=1; i<9; i++) if (diag[i]<diag[min]) min=i;
	TooN::Vector<9> v = svdA.get_VT()[min];

	TooN::Matrix<3,3> Rrel;
	Rrel.slice<0,0,1,3>() = v.slice<0,3>().as_row();
	Rrel.slice<1,0,1,3>() = v.slice<3,3>().as_row();
	Rrel.slice<2,0,1,3>() = v.slice<6,3>().as_row();

	Rrel = orthonormalize(Rrel);
	if ((Rrel[0] ^ Rrel[1])*Rrel[2] < 0.0) {
		Rrel = -Rrel;
	}
	return TooN::SO3<double>(Rrel);
}

TooN::SO3<double> CalibStereo::getRelativePose_differentFrames(const std::vector<TooN::SO3<double> > & pose1, const std::vector<TooN::SO3<double> > & pose2, const std::map<int,int> & synclist)
{
	std::vector<TooN::SO3<double> > poses1;
	std::vector<TooN::SO3<double> > poses2;
	for (std::map<int,int>::const_iterator it = synclist.begin();
	     it != synclist.end(); ++it) {
		poses1.push_back(pose1[it->first]);
		poses2.push_back(pose2[it->second]);
	}
	if (poses1.size()>0) return CalibStereo::getRelativePose_differentFrames(poses1, poses2);
	return TooN::SO3<double>();
}

static TooN::Matrix<4,4> SE3_get_matrix(const TooN::SE3<double> & se3)
{
	TooN::Matrix<4,4> ret;
	ret.slice<0,0,3,3>() = se3.get_rotation().get_matrix();
	ret.slice<0,3,3,1>() = se3.get_translation().as_col();
	ret.slice<3,0,1,4>() = TooN::makeVector(0,0,0,1).as_row();
	return ret;
}

TooN::SE3<double> CalibStereo::getRelativePose_differentFrames(
		const std::vector<TooN::SE3<double> > & pose1,
		const std::vector<TooN::SE3<double> > & pose2)
{
	assert(pose1.size()==pose2.size());
	assert(pose1.size()>0);
	TooN::Matrix<> A(16*(pose1.size()*(pose1.size()-1) / 2), 16);
	for (int r=0; r<A.num_rows(); r++) for (int c=0; c<A.num_cols(); c++) A(r,c) = 0.0;
	int counter = 0;
	for (unsigned int i=0; i<pose1.size(); i++) for (unsigned int j=i+1; j<pose1.size(); j++) {
		TooN::Matrix< 4, 4 > A1 = SE3_get_matrix(pose2[i].inverse());
		TooN::Matrix< 4, 4 > C1 = SE3_get_matrix(pose1[i]);
		TooN::Matrix< 4, 4 > A2 = SE3_get_matrix(pose2[j].inverse());
		TooN::Matrix< 4, 4 > C2 = SE3_get_matrix(pose1[j]);

		for (int ro=0; ro<4; ro++) for (int co=0; co<4; co++) {
			for (int r=0; r<4; r++) for (int c=0; c<4; c++) 
				A(counter,c+4*r) = A1(ro,r) * C1(c,co) - A2(ro,r) * C2(c,co);
			counter++;
		}
	}

	// find column of V with smalest singular value
	TooN::SVD<> svdA(A);
	TooN::Vector<16> diag = svdA.get_diagonal();
	int min = 0;
	for (int i=1; i<16; i++) if (diag[i]<diag[min]) min=i;
	TooN::Vector<16> v = svdA.get_VT()[min];

	// extract R ant t from vector v
	TooN::Matrix<3,3> Rrel;
	Rrel.slice<0,0,1,3>() = v.slice<0,3>().as_row();
	Rrel.slice<1,0,1,3>() = v.slice<4,3>().as_row();
	Rrel.slice<2,0,1,3>() = v.slice<8,3>().as_row();
	TooN::Vector<3> trel;
	trel[0] = v[3];
	trel[1] = v[7];
	trel[2] = v[11];

	// fix the scaling
	double scale = (sqrt(Rrel[0]*Rrel[0]) + sqrt(Rrel[1]*Rrel[1]) + sqrt(Rrel[2]*Rrel[2]) + fabs(v[15]))/4;
	Rrel *= 1.0/scale;
	trel *= 1.0/scale;

	// orthonormalize and take care of negative scaling
	Rrel = orthonormalize(Rrel);
	if ((Rrel[0] ^ Rrel[1])*Rrel[2] < 0.0) {
		Rrel = -Rrel;
		trel = -trel;
	}

	// create and return a SE3
	TooN::SO3<double> Rrel_so3(Rrel);
	return TooN::SE3<double>(Rrel_so3, trel);
}

TooN::SE3<double> CalibStereo::getRelativePose_differentFrames(const std::vector<TooN::SE3<double> > & pose1, const std::vector<TooN::SE3<double> > & pose2, const std::map<int,int> & synclist)
{
	std::vector<TooN::SE3<double> > poses1;
	std::vector<TooN::SE3<double> > poses2;
	for (std::map<int,int>::const_iterator it = synclist.begin();
	     it != synclist.end(); ++it) {
		poses1.push_back(pose1[it->first]);
		poses2.push_back(pose2[it->second]);
	}
	if (poses1.size()>0) return CalibStereo::getRelativePose_differentFrames(poses1, poses2);
	return TooN::SE3<double>();
}

TooN::SE3<double> CalibStereo::getRelativePose(const std::vector<TooN::Vector<3> > & pts1, const std::vector<TooN::Vector<3> > & pts2)
{
	assert(pts1.size()==pts2.size());
	assert(pts1.size()>0);

	// align center of mass
	TooN::Vector<3> center1 = TooN::Zeros, center2 = TooN::Zeros;
	for (unsigned int pt=0; pt<pts1.size(); pt++) {
		center1 += pts1[pt];
		center2 += pts2[pt];
	}
	center1 /= pts1.size();
	center2 /= pts1.size();

	// apply the translation
	// find the rotation
	TooN::Matrix<3,3> M = TooN::Zeros;
	for (unsigned int pt=0; pt<pts1.size(); pt++) {
		M += (pts1[pt] - center1).as_col()*(pts2[pt] - center2).as_row();
	}
	TooN::Matrix<4,4> N;
	N[0][0] = M[0][0] + M[1][1] + M[2][2];
	N[0][1] = M[1][2] - M[2][1];
	N[0][2] = M[2][0] - M[0][2];
	N[0][3] = M[0][1] - M[1][0];

	N[1][0] = N[0][1];
	N[1][1] = M[0][0] - M[1][1] - M[2][2];
	N[1][2] = M[0][1] + M[1][0];
	N[1][3] = M[0][2] + M[2][0];

	N[2][0] = N[0][2];
	N[2][1] = N[1][2];
	N[2][2] =-M[0][0] + M[1][1] - M[2][2];
	N[2][3] = M[1][2] + M[2][1];

	N[3][0] = N[0][3];
	N[3][1] = N[1][3];
	N[3][2] = N[2][3];
	N[3][3] =-M[0][0] - M[1][1] + M[2][2];

	TooN::SymEigen<4> eigN(N);
	int max_eig = 0;
	for (int i = 1; i < 4; ++i) if (eigN.get_evalues()[i]>eigN.get_evalues()[max_eig]) max_eig = i;
	TooN::Vector<4> q = eigN.get_evectors()[max_eig];

	TooN::SE3<double> ret;
	double fact = 2.0 * acos(q[0]) / sqrt(1.0-q[0]*q[0]);
	ret.get_rotation() = TooN::SO3<double>::exp(TooN::makeVector(q[1]*fact, q[2]*fact, q[3]*fact));
	ret.get_translation() = center2 - ret.get_rotation()*center1;
	return ret;
}

TooN::SE3<double> CalibStereo::getRelativePose(const std::vector<TooN::Vector<3> > & points1, const std::vector<TooN::Vector<3> > & points2, const std::map<int,int> & synclist)
{
	std::vector<TooN::Vector<3> > pts1;
	std::vector<TooN::Vector<3> > pts2;
	for (std::map<int,int>::const_iterator it = synclist.begin();
	     it != synclist.end(); ++it) {
		pts1.push_back(points1[it->first]);
		pts2.push_back(points2[it->second]);
	}
	if (pts1.size()>0) return CalibStereo::getRelativePose(pts1, pts2);
	return TooN::SE3<double>();
}

