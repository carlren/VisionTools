#ifndef _INCLUDED_calib_compute_CalibStereo_h_
#define _INCLUDED_calib_compute_CalibStereo_h_

#include <TooN/TooN.h>
#include <TooN/se3.h>

#include <vector>
#include <map>

/** This class is used as a namespace to contain various stereo/hand-eye/multi-
    camera calibration algorithms. The main goal in all cases is to compute a
    rigid transformation between the coordinate frames of different sensors.
*/
class CalibStereo {
	public:
	/** Given two sets of corresponding points expressed in two different
	    coordinate frames, calculate the relative pose between them.
	    In this variant, a general translation and rotation is considered.
	    The algorithm is in fact an implementation of Horn's "absolute
	    orientation" paper.

	    At least two points and their corresponding partners are needed as
	    input.
	*/
	static TooN::SE3<double> getRelativePose(
		const std::vector<TooN::Vector<3> > & points1,
		const std::vector<TooN::Vector<3> > & points2);

	/** Just like the above method, but instead of two sets of
	    corresponding points rather two general sets with a list of
	    "synchronized points" is passed as arguments.
	*/
	static TooN::SE3<double> getRelativePose(
		const std::vector<TooN::Vector<3> > & points1,
		const std::vector<TooN::Vector<3> > & points2,
		const std::map<int,int> & synclist);

	/** Given two sets of corresponding poses relative to the same
	    coordinate frame, calculate the relative rotation and translation
	    between them. As a possible application scenario, consider a set
	    of calibrated images with a common calibration pattern observed by
	    two rigidly aligned cameras. This method then allows to compute the
	    relative pose between the two cameras. In terms of SE3s, the poses
	    will be:
		pose2 = relative * pose1

	    A single pair of poses is sufficient to compute the relative pose.
	    This is a linear least squares problem, solved internally with SVD.
	*/
	static TooN::SE3<double> getRelativePose_commonFrame(
		const std::vector<TooN::SE3<double> > & pose1,
		const std::vector<TooN::SE3<double> > & pose2);

	/** Just like the above method, but instead of two sets of
	    corresponding poses rather two general sets with a list of
	    synchronized frames is passed as arguments.
	*/
	static TooN::SE3<double> getRelativePose_commonFrame(
		const std::vector<TooN::SE3<double> > & pose1,
		const std::vector<TooN::SE3<double> > & pose2,
		const std::map<int,int> & synclist);

	/** Given two sets of corresponding poses relative to two different
	    coordinate frames, calculate the relative rotation between them.
	    In this variant, only pure rotations are considered. Possible
	    application scenarios are hand-eye-calibration or camera rig
	    calibration.

	    In terms of SO3s, the overall transformation from the coordinate
	    frame of pose1 to the coordinate frame of pose2 will be:
		pose1 * relative * pose2^T

	    At least two pairs of corresponding poses (i.e. three different
	    poses) are needed as input and the axes of these relative rotations
	    must not be identical (ideally they are orthogonal). The solution
	    is then determined by solving a linear least squares problem with
	    SVD. Details on the algorithm are given in the SE3 variant of this
	    method.
	*/
	static TooN::SO3<double> getRelativePose_differentFrames(
		const std::vector<TooN::SO3<double> > & pose1,
		const std::vector<TooN::SO3<double> > & pose2);

	/** Just like the above method, but instead of two sets of
	    corresponding poses rather two general sets with a list of
	    synchronized frames is passed as arguments.
	*/
	static TooN::SO3<double> getRelativePose_differentFrames(
		const std::vector<TooN::SO3<double> > & pose1,
		const std::vector<TooN::SO3<double> > & pose2,
		const std::map<int,int> & synclist);

	/** Given two sets of corresponding poses relative to two different
	    coordinate frames, calculate the relative rotation between them.
	    Possible application scenarios are hand-eye-calibration or camera
	    rig calibration.

	    In terms of SE3s, the overall transformation from the coordinate
	    frame of pose1 to the coordinate frame of pose2 will be:
		pose1 * relative * pose2^-1

	    At least two pairs of corresponding poses (i.e. three different
	    poses) are needed as input and the axes of the relative rotations
	    of these poses must not be identical (ideally they are orthogonal).
	    The solution is then determined by solving a linear least squares
	    problem with SVD, as is explained below.

	    Consider the transformations \f$\mathbf{A}_i, \mathbf{C}_i\f$
	    from the coordinate frames of sensors 1 and 2 to the \f$i\f$-th
	    observed pose of each sensor. The concatenated transformation
	    \f$ \mathbf{A}_i \mathbf{T} \mathbf{C}_i^{-1} \f$ with the unknown
	    relative transformation \f$ \mathbf{T} \f$ then transforms the
	    coordinate frame of sensor 1 to the coordinate frame of sensor 2.
	    For each i and j we have:
	    \f[
		\mathbf{A}_i \mathbf{T} \mathbf{C}_i^{-1} = \mathbf{A}_j \mathbf{T} \mathbf{C}_j^{-1}
	    \f] \f[
		\mathbf{A}_i \mathbf{T} \mathbf{C}_i^{-1} - \mathbf{A}_j \mathbf{T} \mathbf{C}_j^{-1} = 0
	    \f]
	    This provides a linear equation system in the entries of
	    \f$ \mathbf{T} \f$. Reinterpreting \f$ \mathbf{T} \f$ as a vector,
	    we arrive at a homogeneous linear equation system with a system
	    matrix defined by the entries of \f$ \mathbf{A}_i, \mathbf{A}_j, \mathbf{C}_i, \mathbf{C}_j \f$.
	    We solve this equation system using SVD. That's it. The only
	    problem is to get the indices right in your implementation...
	*/
	static TooN::SE3<double> getRelativePose_differentFrames(
		const std::vector<TooN::SE3<double> > & pose1,
		const std::vector<TooN::SE3<double> > & pose2);

	/** Just like the above method, but instead of two sets of
	    corresponding poses rather two general sets with a list of
	    synchronized frames is passed as arguments.
	*/
	static TooN::SE3<double> getRelativePose_differentFrames(
		const std::vector<TooN::SE3<double> > & pose1,
		const std::vector<TooN::SE3<double> > & pose2,
		const std::map<int,int> & synclist);
};

#endif

