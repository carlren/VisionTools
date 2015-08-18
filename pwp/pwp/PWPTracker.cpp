#include "PWPTracker.hpp"
#include <iostream>
#include <fstream>

#ifndef _DELETE
	#define _DELETE(x) {if(x) {delete(x); x=NULL;}}
	#define _DELETEARRAY(x) {if(x) {delete [] (x); x=NULL;}}
#endif

#ifndef min
#define min(a,b) (a<b?a:b)
#endif

#ifndef max
#define max(a,b) (a>b?a:b)
#endif

using namespace std;

//---------------------------------------------------------------------------------

PWPTracker::PWPTracker() 
{
	// learning settings
	m_sSS.fAppearanceLearningRate	=	0.5;
	m_sSS.fShapeLearningRate		=	0.3;

	// openGL drawing setting
	m_sDS.bDiscretizeImage			=	false;
	m_sDS.bClassifyWholeImage		=	false;
	m_sDS.bDrawLevelSet				=	true;
	m_sDS.eOverlayType				=	OVERLAY_NONE;
	m_sDS.bDrawNarrowBand			=	true;
	m_sDS.n2DOverlaySize			=	150;
	m_sDS.n3DOverlaySize			=	150;
	m_sDS.bOverlay2D				=	true;
	m_sDS.bOverlay3D				=	true;
	m_sDS.bAnimateLevelSet			=	true;
	m_sDS.fRange					=	1000.0f;
	m_sDS.fTheta					=	DegToRad(-90.0);
	m_sDS.fPhi						=	DegToRad(0.0);
	
	// pwp advance settings
	m_sAS.fPriorBackground			=	0.6;
	m_sAS.bUseSegmentation			=	true;
	m_sAS.fEpsilon					=	0.3; // smoothness for heaviside
	m_sAS.fMu						=	0.2; // evolution step size, can't be too big
	m_sAS.fPsi						=	1.0;
	m_sAS.nShapeIterations			=	1;
	m_sAS.nTrackIterations			=	20;
	m_sAS.fDamping					=	2.0;
	m_sAS.fStepSize					=	0.25;
	m_sAS.bUseDriftCorrection		=	true;
	m_sAS.fNarrowBand				=	2.0;
	m_sAS.eModelType				=	MODEL_TSR;
	m_sAS.nBitsY					=	5;
	m_sAS.nBitsU					=	5;
	m_sAS.nBitsV					=	5;
	m_sAS.fSigma					=	1.0;
	m_sAS.eOpinionPoolType			=	OP_LINEAR;//	OP_LOGARITHMIC;
	//m_sAS.eOpinionPoolType			= 	OP_LOGARITHMIC;

	// others
	m_fLSAnimationTime				=	0.0;
	m_pcBackHist					=	NULL;
	m_pcDiscretizedImage			=	NULL;
	m_mClassifiedImage				=	NULL;
	m_pasImageInfo					=	NULL;
	m_bHasTarget					=	false;

}

//---------------------------------------------------------------------------------

PWPTracker::~PWPTracker()
{
	Clear();
}

//---------------------------------------------------------------------------------

bool PWPTracker::_InsideImage(int nW, int nH, int nX, int nY)
{
	return nX>=0 && nX<nW && nY>=0 && nY<nH;
}

//---------------------------------------------------------------------------------

Eigen::Matrix3f PWPTracker::_BuildWarpMatrix(Pose vp)
{
    Eigen::Matrix3f mW;

	if(m_sAS.eModelType == MODEL_T)
	{
		mW(0, 0) = 1;								mW(0, 1) = 0;								mW(0, 2) = vp[0];
		mW(1, 0) = 0;								mW(1, 1) = 1;								mW(1, 2) = vp[1];
		mW(2, 0) = 0;								mW(2, 1) = 0;								mW(2, 2) = 1;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_TS)
	{
		mW(0, 0) = 1 + vp[2];						mW(0, 1) = 0;								mW(0, 2) = vp[0];
		mW(1, 0) = 0;								mW(1, 1) = 1 + vp[2];						mW(1, 2) = vp[1];
		mW(2, 0) = 0;								mW(2, 1) = 0;								mW(2, 2) = 1;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_TSR)
	{
		mW(0, 0) = (1 + vp[2])*cos(vp[3]);			mW(0, 1) = -(1 + vp[2])*sin(vp[3]);			mW(0, 2) = vp[0];
		mW(1, 0) = (1 + vp[2])*sin(vp[3]);			mW(1, 1) = (1 + vp[2])*cos(vp[3]);			mW(1, 2) = vp[1];
		mW(2, 0) = 0;								mW(2, 1) = 0;								mW(2, 2) = 1;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_AFFINE)
	{
		mW(0, 0) = (1 + vp[2]);						mW(0, 1) = vp[4];							mW(0, 2) = vp[0];
		mW(1, 0) = vp[3];							mW(1, 1) = (1 + vp[5]);						mW(1, 2) = vp[1];
		mW(2, 0) = 0;								mW(2, 1) = 0;								mW(2, 2) = 1;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_HOMOGRAPHY)
	{
		mW(0, 0) = (1 + vp[2]);						mW(0, 1) = vp[4];							mW(0, 2) = vp[0];
		mW(1, 0) = vp[3];							mW(1, 1) = (1 + vp[5]);						mW(1, 2) = vp[1];
		mW(2, 0) = vp[6];							mW(2, 1) = vp[7];							mW(2, 2) = 1;
		return mW;
	}
	return mW;
}

//---------------------------------------------------------------------------------

Eigen::Matrix3f PWPTracker::_BuildInverseWarpMatrix(Pose vp)
{
	Eigen::Matrix3f mW;

	if(m_sAS.eModelType == MODEL_T)
	{
		mW(0,0) = 1.0;								mW(0,1) = 0.0;								mW(0,2) = -vp[0];
		mW(1,0) = 0.0;								mW(1,1) = 1.0;								mW(1,2) = -vp[1];
		mW(2,0) = 0.0;								mW(2,1) = 0.0;								mW(2,2) = 1.0;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_TS)
	{
		double fDenom = (1+vp[2]);
		mW(0,0) = 1.0/fDenom;						mW(0,1) = 0.0;								mW(0,2) = -vp[0]/fDenom;
		mW(1,0) = 0.0;								mW(1,1) = 1.0/fDenom;						mW(1,2) = -vp[1]/fDenom;
		mW(2,0) = 0.0;								mW(2,1) = 0.0;								mW(2,2) = 1.0;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_TSR)
	{
		double fDenom = (1+vp[2]);
		mW(0,0) = cos(vp[3])/fDenom;				mW(0,1) = sin(vp[3])/fDenom;				mW(0,2) = -(vp[0]*cos(vp[3])+vp[1]*sin(vp[3]))/fDenom;
		mW(1,0) = -sin(vp[3])/fDenom;				mW(1,1) = cos(vp[3])/fDenom;				mW(1,2) = (vp[0]*sin(vp[3])-vp[1]*cos(vp[3]))/fDenom;
		mW(2,0) = 0.0;								mW(2,1) = 0.0;								mW(2,2) = 1.0;
		return mW;
	}
	if(m_sAS.eModelType == MODEL_AFFINE)
	{
		double fDenom = (1+vp[5]+vp[2]+vp[2]*vp[5]-vp[3]*vp[4]);
		mW(0,0) = (1+vp[5])/fDenom;					mW(0,1) = -vp[4]/fDenom;					mW(0,2) = (-vp[0]+vp[1]*vp[4]-vp[0]*vp[5])/fDenom;
		mW(1,0) = -vp[3]/fDenom;					mW(1,1) = (1.0+vp[2])/fDenom;				mW(1,2) = (-vp[1]-vp[1]*vp[2]+vp[0]*vp[3])/fDenom;
		mW(2,0) = 0.0;								mW(2,1) = 0.0;								mW(2,2) = 1.0;
		return mW;
	}

	return mW;
}

//---------------------------------------------------------------------------------

void PWPTracker::Clear()		
{
	if(m_bHasTarget)
	{
		_DELETE(m_pcBackHist);
		_DELETEARRAY(m_pasImageInfo);
		_DELETE(m_sTarget.pcForeHist);
		_DELETEARRAY(m_sTarget.pasLevelSet);
		m_sTarget.vFGCorners.clear();
		m_sTarget.vBGCorners.clear();

		m_bHasTarget = false;
	}
}

//---------------------------------------------------------------------------------

void PWPTracker::_DiscretizeImage(cv::Mat& cImage)
{
	if (!m_pcDiscretizedImage)
        m_pcDiscretizedImage = new cv::Mat(cImage.rows,cImage.cols,CV_8UC3);
    if (m_pcDiscretizedImage->size!=cImage.size) {
        cv::resize(cImage, *m_pcDiscretizedImage, cv::Size(cImage.rows,cImage.cols) );
    }


	for (int i = 0; i < cImage.rows; i++)
    for (int j = 0; j < cImage.cols; j++)
	{
        cv::Vec3b &inpix = m_pcDiscretizedImage->at<cv::Vec3b>(i,j);
        cv::Vec3b &outpix = cImage.at<cv::Vec3b>(i,j);
        
        outpix[0] = (unsigned char) (inpix[0] >> (8 - m_sAS.nBitsY)) << (8 - m_sAS.nBitsY);
        outpix[1] = (unsigned char) (inpix[1] >> (8 - m_sAS.nBitsY)) << (8 - m_sAS.nBitsY);
        outpix[2] = (unsigned char) (inpix[2] >> (8 - m_sAS.nBitsY)) << (8 - m_sAS.nBitsY);
    }
}

//---------------------------------------------------------------------------------

void PWPTracker::_ClassifyImage(cv::Mat& cImage)
{
    if (!m_mClassifiedImage) {
        /*m_mClassifiedImage = new cv::Mat(cImage.rows,cImage.cols,CV_8SC1);*/
		m_mClassifiedImage = new cv::Mat(cImage.rows, cImage.cols, CV_8UC3);
    }
    m_mClassifiedImage->setTo(cv::Scalar(0));

	if(!m_bHasTarget)
		return;

	// Image info
	const int imagewidth = m_nImageWidth;
	const int imageheight = m_nImageHeight;
	ImageInfo* const pII = m_pasImageInfo;

	for (int r = 0; r < imageheight; r++)
		for (int c = 0; c<imagewidth; c++)
        {
            if (pII[r*imagewidth + c].LHF > pII[r*imagewidth + c].LHB)
				m_mClassifiedImage->at<cv::Vec3b>(r, c) = cv::Vec3b(255,255,255);
            
        }
}

//---------------------------------------------------------------------------------

Eigen::Matrix3f PWPTracker::_ComputeDriftCorrection()
{
	Eigen::Matrix3f mW;
	
	double fMinX = 999999;
	double fMaxX = -999999;
	double fMinY = 999999;
	double fMaxY = -999999;

	// Level set
	const int rowlength = m_sTarget.nBGWidth;
	LevelSetInfo* pLS = m_sTarget.pasLevelSet;

	for(int nr=0,cr=-m_sTarget.nBGHalfHeight; cr<=m_sTarget.nBGHalfHeight; nr++, cr++)
	{
		for(int nc=0,cc=-m_sTarget.nBGHalfWidth; cc<=m_sTarget.nBGHalfWidth; nc++, cc++)
		{
			if(pLS[nr*rowlength+nc].U<=0.0f) 
				continue;
		
			fMinX = min(double(cc),fMinX);
			fMaxX = max(double(cc),fMaxX);
			fMinY = min(double(cr),fMinY);
			fMaxY = max(double(cr),fMaxY);
		}
	}

	double fRB = m_sTarget.nFGHalfWidth-fMaxX;
	double fLB = fMinX+m_sTarget.nFGHalfWidth;
	double fBB = m_sTarget.nFGHalfHeight-fMaxY;
	double fTB = fMinY+m_sTarget.nFGHalfHeight;

	// Drift correction
	double fDTX = max(min((fLB-fRB),0.4),-0.4);
	double fDTY = max(min((fTB-fBB),0.4),-0.4);
	double fDSX = min(fRB,fLB);
	double fDSY = min(fTB,fBB);
	double fDScaleControl = max(min(0.005f*(BORDERDRIFT-min(fDSX,fDSY)),0.1f),-0.1f);
	double fDScaleControlX = max(min(0.02f*(BORDERDRIFT-fDSX),0.2f),-0.2f);
	double fDScaleControlY = max(min(0.02f*(BORDERDRIFT-fDSY),0.2f),-0.2f);
	
	if(m_sAS.eModelType == MODEL_T)
	{
		mW(0,0) = 1.0;							mW(0,1) = 0.0;							mW(0,2) = fDTX;
		mW(1,0) = 0.0;							mW(1,1) = 1.0;							mW(1,2) = fDTY;
		mW(2,0) = 0.0;							mW(2,1) = 0.0;							mW(2,2) = 1.0;
	}
	else if(m_sAS.eModelType <= MODEL_TSR)
	{
		double fDThetaControl = max(min(-m_sTarget.vPose[3],0.001f),-0.001f);
		mW(0,0) = (1.0+fDScaleControl)*cos(fDThetaControl);		mW(0,1) = -sin(fDThetaControl);							mW(0,2) = fDTX;
		mW(1,0) = sin(fDThetaControl);							mW(1,1) = (1.0+fDScaleControl)*cos(fDThetaControl);		mW(1,2) = fDTY;
		mW(2,0) = 0.0;											mW(2,1) = 0.0;											mW(2,2) = 1.0;
	}
	else
	{
		mW(0,0) = 1.0+fDScaleControlX;				mW(0,1) = 0.0;								mW(0,2) = fDTX;
		mW(1,0) = 0.0;								mW(1,1) = 1.0+fDScaleControlY;				mW(1,2) = fDTY;
		mW(2,0) = 0.0;								mW(2,1) = 0.0;								mW(2,2) = 1.0;
	}

	const double WDC_00 = mW(0,0);
	const double WDC_01 = mW(0,1);
	const double WDC_02 = mW(0,2);
	const double WDC_10 = mW(1,0);
	const double WDC_11 = mW(1,1);
	const double WDC_12 = mW(1,2);

	const int rowlength0 = m_sTarget.nBGWidth;
	LevelSetInfo* pLS0 = m_sTarget.pasLevelSet;

	const int ybegin0 = -m_sTarget.nBGHalfHeight;
	const int yend0 = m_sTarget.nBGHalfHeight;
	const int xbegin0 = -m_sTarget.nBGHalfWidth;
	const int xend0 =m_sTarget.nBGHalfWidth;

	// Copy stuff so we can use it for our bilinear interpolation
	int lssize = m_sTarget.nBGWidth*m_sTarget.nBGHeight;
	LevelSetInfo* pCLS = pLS0;
	while(lssize--)
	{
		pCLS->CopyU = pCLS->U;
		pCLS->CopyPF = pCLS->PF;
		pCLS->CopyPB = pCLS->PB;
		pCLS++;
	}

	pCLS = pLS0;
	for(int y=ybegin0; y<=yend0; y++)
	{
		for(int x=xbegin0; x<=xend0; x++)
		{
			const double xdc = WDC_00*x+WDC_01*y+WDC_02;
			const double ydc = WDC_10*x+WDC_11*y+WDC_12;
			const int LX = floor(xdc);
			const int LY = floor(ydc);

			if(LX>=xbegin0 && LX+1<=xend0 && LY>=ybegin0 && LY+1<=yend0)
			{
				const double RX = xdc-LX;
				const double RY = ydc-LY;
				const double OneMinusRX = 1.0-RX;
				const double OneMinusRY = 1.0-RY;

				LevelSetInfo* pLXLY = pLS0+(LY+yend0)*rowlength0+(LX+xend0);
				LevelSetInfo* pUXLY = pLS0+(LY+yend0)*rowlength0+(LX+xend0+1);
				LevelSetInfo* pLXUY = pLS0+(LY+yend0+1)*rowlength0+(LX+xend0);
				LevelSetInfo* pUXUY = pLS0+(LY+yend0+1)*rowlength0+(LX+xend0+1);

				pCLS->PF = OneMinusRY*(OneMinusRX*pLXLY->CopyPF+RX*pUXLY->CopyPF)+RY*(OneMinusRX*pLXUY->CopyPF+RX*pUXUY->CopyPF);
				pCLS->PB = OneMinusRY*(OneMinusRX*pLXLY->CopyPB+RX*pUXLY->CopyPB)+RY*(OneMinusRX*pLXUY->CopyPB+RX*pUXUY->CopyPB);
				pCLS->U =  OneMinusRY*(OneMinusRX*pLXLY->CopyU +RX*pUXLY->CopyU) +RY*(OneMinusRX*pLXUY->CopyU +RX*pUXUY->CopyU);
			}
			pCLS++;
		}
	}

	return mW;
}

//---------------------------------------------------------------------------------

void PWPTracker::_IterateShapeAlignment(const cv::Mat& cImage)
{
	// Pose
	const Pose vplast = m_sTarget.vPose;
	Pose& vp = m_sTarget.vPose;
	const Pose vppred = (vplast+m_sTarget.vVelocity);

	// Do a velocity prediction
	// vp = vppred;

	// Compute intial warp
	Eigen::Matrix3f mW = _BuildWarpMatrix(vp);

	// Image info
	const int imagewidth = m_nImageWidth;
	const int imageheight = m_nImageHeight;
	ImageInfo* const pII = m_pasImageInfo;
		
	// Level set
	const int rowlength0 = m_sTarget.nBGWidth;
	LevelSetInfo* const pLS0 = m_sTarget.pasLevelSet;

	bool bRankDeficiency;
	int nSize;

	if(m_sAS.eModelType == MODEL_T)
		nSize = 2;
	else if(m_sAS.eModelType == MODEL_TS)
		nSize = 3;
	else if(m_sAS.eModelType == MODEL_TSR)
		nSize = 4;
	else if(m_sAS.eModelType == MODEL_AFFINE)
		nSize = 6;
	else if(m_sAS.eModelType == MODEL_HOMOGRAPHY)
		nSize = 8;
	else
	{
		cerr << "Invalid model type" << endl;
		return;
	}

    Eigen::MatrixXf mJTJ(MAXPS,MAXPS); mJTJ.setZero();
    Eigen::MatrixXf mJTJPrior(MAXPS,MAXPS); mJTJPrior.setZero();
    
	Eigen::VectorXf vJTB(MAXPS); vJTB.setZero();
	Eigen::VectorXf vJTBPrior(MAXPS); vJTBPrior.setZero();
    
	//double avJTB[MAXPS];
	
	MODELTYPE eModelType = MODEL_T;
	const double fFiveTimesEpsilon = m_sAS.fEpsilon;

	// Pre compute stuff
    const int nWI = cImage.cols;
    const int nHI = cImage.rows;
	
	const int rbegin0 = 0+1;
	const int ybegin0 = -m_sTarget.nBGHalfHeight+1;
	const int yend0 = m_sTarget.nBGHalfHeight-1;
	const int cbegin0 = 0+1;
	const int xbegin0 = -m_sTarget.nBGHalfWidth+1;
	const int xend0 = m_sTarget.nBGHalfWidth-1;

	double fNb = 0.0;
	double fNf = 0.0;

	//ofstream ofs("e:/pwp/levelset_test.txt");
	//for (int r = rbegin0, _y = ybegin0; _y <= yend0; r++, _y++)
	//{
	//	for (int c = cbegin0, _x = xbegin0; _x <= xend0; c++, _x++)
	//	{
	//		LevelSetInfo* pCLS = pLS0 + r*rowlength0 + c;
	//		ofs << pCLS->U << "\t";
	//	}
	//	ofs << "\n";
	//}
	//ofs.close();

	int nPixels = 0;
	{
		// Compute warps
//		const double W_00 = mW(0, 0);
//		const double W_01 = mW(0, 1);
//		const double W_02 = mW(0, 2);
//		const double W_10 = mW(1, 0);
//		const double W_11 = mW(1, 1);
//		const double W_12 = mW(1, 2);

		for(int r=rbegin0,_y=ybegin0; _y<=yend0; r+=1, _y+=1)
		{
			for(int c=cbegin0,_x=xbegin0; _x<=xend0; c+=1, _x+=1)
			{
				const double fU = pLS0[r*rowlength0+c].U;

				if(abs(fU)>m_sAS.fNarrowBand)
					continue;

				PrecompStuff &cPC = m_asPreComp[nPixels++];
				
				cPC.nR = r;
				cPC.nC = c;
				cPC.nX = _x;
				cPC.nY = _y;
				cPC.H = _Heaviside(fU,fFiveTimesEpsilon);

				const double fH = cPC.H;

				fNb  += (1.0-fH);
				fNf  += fH;

				const double dx = pLS0[r*rowlength0+c].DX;
				const double dy = pLS0[r*rowlength0+c].DY;
				const double fDirac = _Dirac(fU,fFiveTimesEpsilon);

				// Compute jacobian
				double J0=0,J1=0,J2=0,J3=0,J4=0,J5=0,J6=0,J7=0;
				J0 = dx*fDirac;
				J1 = dy*fDirac;

				double fX = _x;
				double fY = _y;

				if(m_sAS.eModelType == MODEL_TS)
				{
					J2 = (dx*fX+dy*fY)*fDirac;
				}
				else if(m_sAS.eModelType == MODEL_TSR) // Evaluated at zero
				{
					J2 = (dx*fX+dy*fY)*fDirac;
					J3 = (-dx*fY+dy*fX)*fDirac;
				}
				else if(m_sAS.eModelType == MODEL_AFFINE)
				{
					J2 = dx*fX*fDirac;
					J3 = dy*fX*fDirac;	
					J4 = dx*fY*fDirac;	
					J5 = dy*fY*fDirac;
				}
				else if(m_sAS.eModelType == MODEL_HOMOGRAPHY)
				{
					J2 = dx*fX*fDirac;
					J3 = dy*fX*fDirac;	
					J4 = dx*fY*fDirac;	
					J5 = dy*fY*fDirac;
					J6 = fDirac*(-dx*fX*fX-dy*fX*fY);
					J7 = fDirac*(-dx*fX*fY-dy*fY*fY);
				}

				// Copy for use in main loop
				cPC.J0 = J0;
				cPC.J1 = J1;
				cPC.J2 = J2;
				cPC.J3 = J3;
				/*cPC.J4 = J4;
				cPC.J5 = J5;
				cPC.J6 = J6;
				cPC.J7 = J7;*/

				// JTJ
				mJTJ(0,0) += J0*J0;
				mJTJ(0,1) += J0*J1;
				mJTJ(1,1) += J1*J1;

				switch(m_sAS.eModelType)
				{
					case(MODEL_HOMOGRAPHY):
					{
						// JTJ
						mJTJ(1,6) += J0*J6;
						mJTJ(1,7) += J0*J7;
						mJTJ(1,6) += J1*J6;
						mJTJ(1,7) += J1*J7;
						mJTJ(2,6) += J2*J6;
						mJTJ(2,7) += J2*J7;
						mJTJ(3,6) += J3*J6;
						mJTJ(3,7) += J3*J7;
						mJTJ(4,6) += J4*J6;
						mJTJ(4,7) += J4*J7;
						mJTJ(5,6) += J5*J6;
						mJTJ(5,7) += J5*J7;
						mJTJ(6,6) += J6*J6;
						mJTJ(6,7) += J6*J7;
						mJTJ(7,7) += J7*J7;
					}
					case(MODEL_AFFINE):
					{
						// JTJ
						mJTJ(0,4) += J0*J4;
						mJTJ(0,5) += J0*J5;
						mJTJ(1,4) += J1*J4;
						mJTJ(1,5) += J1*J5;
						mJTJ(2,4) += J2*J4;
						mJTJ(2,5) += J2*J5;
						mJTJ(3,4) += J3*J4;
						mJTJ(3,5) += J3*J5;
						mJTJ(4,4) += J4*J4;
						mJTJ(4,5) += J4*J5;
						mJTJ(5,5) += J5*J5;
					}
					case(MODEL_TSR):
					{
						// JTJ
						mJTJ(0,3) += J0*J3;
						mJTJ(1,3) += J1*J3;
						mJTJ(2,3) += J2*J3;
						mJTJ(3,3) += J3*J3;
					}
					case(MODEL_TS):
					{
						// JTJ
						mJTJ(0,2) += J0*J2;
						mJTJ(1,2) += J1*J2;
						mJTJ(2,2) += J2*J2;
					}
                    case (MODEL_T):
                    {}
				}
			}
		}
	}

	// Copy upper triangle into lower triangle
	for(int r=0;r<nSize;r++)
		for(int c=r+1;c<nSize;c++)
			mJTJ(c,r) = mJTJ(r,c);

	//cout << mJTJ;

	// Optimisation 
	//double fScaleProb = m_sTarget.pcForeHist->Bins();
	int nIt;

	ImageInfo IIOutside;
	IIOutside.LHB = 0.5;
	IIOutside.LHF = 0.5;

	double fN = fNb+fNf;
	double fStepSize = m_sAS.fStepSize/**fN*/;
	//bool bUsingLargeStepSize = true;

	//cout << mW;

	for(nIt=0;nIt<m_sAS.nTrackIterations;nIt++)
	{
		double fForegroundProb = 0.0;
		double fBackgroundProb = 0.0;

		// Compute warps
		const double W_00 = mW(0, 0);
		const double W_01 = mW(0, 1);
		const double W_02 = mW(0, 2);
		const double W_10 = mW(1, 0);
		const double W_11 = mW(1, 1);
		const double W_12 = mW(1, 2);

		int nCurrPix = nPixels;
		PrecompStuff* pPC = m_asPreComp;

		double vJTB0 = 0.0;
		double vJTB1 = 0.0;
		double vJTB2 = 0.0;
		double vJTB3 = 0.0;

		double fCostLinPWP = 0.0;
		double fCostLogPWP = 0.0;
		double fCostLogLike = 0.0;
		double fCostLogLikeCremers = 0.0;

		while(nCurrPix--)
		{
			const int &x = pPC->nX;
			const int &y = pPC->nY;
			const int &r = pPC->nR;
			const int &c = pPC->nC;

			const double cf = W_00*x+W_01*y+W_02;
			const double rf = W_10*x+W_11*y+W_12;

			const int ci = floor(cf);
			const int ri = floor(rf);

			// Copy likelihoods from image data or outside pixel (uniform)
			ImageInfo* pIILCLR;
			ImageInfo* pIIUCLR;
			ImageInfo* pIILCUR;
			ImageInfo* pIIUCUR;
			double LHF,LHB;
			if(ci<0 || ci>=nWI-1 || ri<0 || ri>=nHI-1)
			{
				LHF = 0.5;
				LHB = 0.5;
			}
			else
			{
				const double RC = cf-ci;
				const double RR = rf-ri;
				const double OneMinusRC = 1.0-RC;
				const double OneMinusRR = 1.0-RR;

				pIILCLR = &pII[ri*imagewidth+ci];
				pIIUCLR = &pII[ri*imagewidth+ci+1];
				pIILCUR = &pII[(ri+1)*imagewidth+ci];
				pIIUCUR = &pII[(ri+1)*imagewidth+(ci+1)];

				LHF = OneMinusRR*(OneMinusRC*pIILCLR->LHF+RC*pIIUCLR->LHF)+RR*(OneMinusRC*pIILCUR->LHF+RC*pIIUCUR->LHF);
				LHB = OneMinusRR*(OneMinusRC*pIILCLR->LHB+RC*pIIUCLR->LHB)+RR*(OneMinusRC*pIILCUR->LHB+RC*pIIUCUR->LHB);
			}

			// Compute heaviside step functions
			const double fH = pPC->H;
			
			double fB;
			if(m_sAS.eOpinionPoolType==OP_LOGARITHMIC)
				fB = (LHF-LHB)/(LHF*fH+LHB*(1.0-fH));
			else
				fB = (LHF-LHB)/((fNf*LHF+fNb*LHB)/fN);

			//fForegroundProb += (LHF*fH+LHB*(1.0-fH))/(fNf*LHF+fNb*LHB);
			//fBackgroundProb += LHB*(1.0-m1*fH[1])*(1.0-m2*fH[2]))/(LHF+;

			//JTB
			/*avJTB[0]*/vJTB0 += fB*pPC->J0;
			/*avJTB[1]*/vJTB1 += fB*pPC->J1;
			/*avJTB[2]*/vJTB2 += fB*pPC->J2;
			/*avJTB[3]*/vJTB3 += fB*pPC->J3;
			/*avJTB[4] += fB*J4;
			avJTB[5] += fB*J5;
			avJTB[6] += fB*J6;
			avJTB[7] += fB*J7;*/

			pPC++;
		}

		m_sTarget.fForegroundProb = fCostLinPWP;
		m_sTarget.fBackgroundProb = fBackgroundProb;
//			m_sTarget.afCost[nIt] = m_sTarget.fForegroundProb+m_sTarget.fBackgroundProb;

        mJTJPrior.setZero();
        vJTBPrior.setZero();
		
		// Prior
		if(eModelType==MODEL_TS) 
		{
			mJTJPrior(2,2) += m_sAS.fDamping;
			vJTBPrior[2] += -m_sAS.fDamping*(vppred[2]-vp[2]);
		}
		if(eModelType==MODEL_TSR)
		{
			mJTJPrior(0,0) = m_sAS.fDamping;
			mJTJPrior(1,1) = m_sAS.fDamping;
			mJTJPrior(2,2) = 100.0*m_sAS.fDamping;
			mJTJPrior(3,3) = 100.0*m_sAS.fDamping;
			vJTBPrior[0] = -m_sAS.fDamping*(vppred[0]-vp[0]);
			vJTBPrior[1] = -m_sAS.fDamping*(vppred[1]-vp[1]);
			vJTBPrior[2] = -100.0*m_sAS.fDamping*(vppred[2]-vp[2]);
			vJTBPrior[3] = -100.0*m_sAS.fDamping*(vppred[3]-vp[3]);
		}
		/*if(eModelType==MODEL_AFFINE)
		{
			mJTJPrior[2][2] += m_sAS.fDamping*fN;
			mJTJPrior[3][3] += m_sAS.fDamping*fN;
			vJTBPrior[2] += -m_sAS.fDamping*fN*(vplast[2]-vp[2]);
			vJTBPrior[3] += -m_sAS.fDamping*fN*(vplast[3]-vp[3]);
		}
		if(eModelType==MODEL_HOMOGRAPHY)
		{
			amJTJ[2][2] += 100.0;
			amJTJ[3][3] += 100.0;
			amJTJ[4][4] += 100.0;
			amJTJ[5][5] += 100.0; 
			amJTJ[6][6] += m_sAS.fDamping*fN;
			amJTJ[7][7] += m_sAS.fDamping*fN; 
			avJTB[6] += m_sAS.fDamping*fN*vp[6];
			avJTB[7] += m_sAS.fDamping*fN*vp[7];
		}*/

		//mJTJPrior.Fill(0.0);
		//vJTBPrior.Fill(0.0);

		Pose vDP(MAXPS); vDP.setZero();

		/*for(int r=0;r<nSize;r++)
			vJTB[r] = avJTB[r];*/
		vJTB[0] = vJTB0;
		vJTB[1] = vJTB1;
		vJTB[2] = vJTB2;
		vJTB[3] = vJTB3;

        Eigen::MatrixXf combineJtJ = mJTJPrior + mJTJ;
        Eigen::Matrix4f finalJtJ;
    
		for (int r = 0; r < 4; r++) for (int c = 0; c < 4;c++)
			finalJtJ(r, c) = combineJtJ(r, c);

		Eigen::VectorXf finalJtB(4);
		finalJtB[0] = vJTBPrior[0] + vJTB[0];
		finalJtB[1] = vJTBPrior[1] + vJTB[1];
		finalJtB[2] = vJTBPrior[2] + vJTB[2];
		finalJtB[3] = vJTBPrior[3] + vJTB[3];

		//ofstream ofs("e:/pwp/out.txt");
		//ofs << finalJtJ << endl << endl;
		//ofs << finalJtB << endl << endl;

		Eigen::LLT<Eigen::MatrixXf> lltofJtJ(finalJtJ);
        Eigen::VectorXf ans = lltofJtJ.solve(finalJtB);

		//ofs << ans << endl << endl;
		//ofs.close();

		vDP[0] = ans[0]; vDP[1] = ans[1]; vDP[2] = ans[2]; vDP[3] = ans[3];

		//cout << vDP;

		bRankDeficiency = false;

		if(!bRankDeficiency)
		{
			if(eModelType!=MODEL_HOMOGRAPHY)
			{
				MODELTYPE eRealModelType = m_sAS.eModelType;
				m_sAS.eModelType = eModelType;

				//cout << "before\n" << mW << endl;
				//cout << "times\n" << _BuildInverseWarpMatrix(vDP) << endl;

				//mW = _BuildInverseWarpMatrix(vDP)*mW; // this one need to check whether it's this way
				mW = mW*_BuildInverseWarpMatrix(vDP); // when using Eigen, it's a different direction from ORUtil

				//cout << "after\n" << mW << endl;

				m_sAS.eModelType = eRealModelType;

				// Update pose parameters
				vp[0] = mW(0,2);
				vp[1] = mW(1,2);

				if(eModelType == MODEL_TS)
				{
					vp[2] = mW(0,0)-1;
					vp[2] = max(vp[2],-0.9f);
				}
				else if(eModelType == MODEL_TSR)
				{
					vp[3] = atan2(mW(1,0),mW(0,0));
					vp[2] =	sqrt((mW(0,0)*mW(1,1))-(mW(0,1)*mW(1,0)))-1;
					vp[2] = max(vp[2],-0.9f);
				}
				else if(eModelType == MODEL_AFFINE)
				{
					vp[2] = mW(0,0)-1;
					vp[3] = mW(1,0);
					vp[4] = mW(0,1);
					vp[5] = mW(1,1)-1;
					vp[2] = max(vp[2],-0.9f);
				}
			}
			else
			{
				// Inverse
				Pose vpi;
				double fDet = 1.0+vDP[5]-vDP[1]*vDP[7]+vDP[2]+vDP[2]*vDP[5]-vDP[2]*vDP[1]*vDP[7]-vDP[3]*vDP[4]+vDP[3]*vDP[0]*vDP[7]+vDP[6]*vDP[4]*vDP[1]-vDP[6]*vDP[0]-vDP[6]*vDP[0]*vDP[5];
				double fDetAlpha = fDet*((1.0+vDP[2])*(1.0+vDP[5])-vDP[3]*vDP[4]);
				vpi[0] = -vDP[0]-vDP[5]*vDP[0]+vDP[4]*vDP[1];
				vpi[1] = -vDP[1]-vDP[2]*vDP[1]+vDP[3]*vDP[0];
				vpi[2] = 1.0+vDP[5]-vDP[1]*vDP[7]-fDetAlpha;
				vpi[3] = -vDP[3]+vDP[1]*vDP[6];
				vpi[4] = -vDP[4]+vDP[0]*vDP[7];
				vpi[5] = 1.0+vDP[2]-vDP[0]*vDP[6]-fDetAlpha;
				vpi[6] = -vDP[6]-vDP[5]*vDP[6]+vDP[3]*vDP[7];
				vpi[7] = -vDP[7]-vDP[2]*vDP[7]+vDP[4]*vDP[6];
				vpi = vpi/fDetAlpha;

				// Composition
				Pose vpc = vp;
				vp[0] = vpc[0]+vpi[0]+vpc[2]*vpi[0]+vpc[4]*vpi[1];
				vp[1] = vpc[1]+vpi[1]+vpc[3]*vpi[0]+vpc[5]*vpi[1];
				vp[2] = vpc[2]+vpi[2]+vpc[2]*vpi[2]+vpc[4]*vpi[3]+vpc[0]*vpi[6]-vpc[6]*vpi[0]-vpc[7]*vpi[1];
				vp[3] = vpc[3]+vpi[3]+vpc[3]*vpi[2]+vpc[5]*vpi[3]+vpc[1]*vpi[6];
				vp[4] = vpc[4]+vpi[4]+vpc[2]*vpi[4]+vpc[4]*vpi[5]+vpc[0]*vpi[7];
				vp[5] = vpc[5]+vpi[5]+vpc[3]*vpi[4]+vpc[5]*vpi[5]+vpc[1]*vpi[7]-vpc[6]*vpi[0]-vpc[7]*vpi[1];
				vp[6] = vpc[6]+vpi[6]+vpc[6]*vpi[2]+vpc[7]*vpi[3];
				vp[7] = vpc[7]+vpi[7]+vpc[6]*vpi[4]+vpc[7]*vpi[5];
				vp = vp/(1.0+vpc[6]*vpi[0]+vpc[7]*vpi[1]);

				mW = _BuildWarpMatrix(vp);
			}
		}

		if(eModelType==MODEL_T && m_sAS.eModelType!=MODEL_T)
		{
			if (vDP[0] * vDP[0] + vDP[1] + vDP[1]<1.0)
			{
				eModelType = m_sAS.eModelType;
//					nEpsilon = m_nEpsilon;
				m_sTarget.nIterationsTOnly = nIt+1;
			}
		}
		else
		{
			float posediff=0; for (int i = 0; i < MAXPS; i++) posediff += vDP[i] * vDP[i];
			if(posediff<0.1) break;
		}
	}

	m_sTarget.nIterations = nIt+1;

	//m_sTarget.fTargetScore = m_sTarget.afCost[m_nMaxIterations-1];
    bRankDeficiency = false;
	if(bRankDeficiency /*|| !_InsideImage(m_nImageWidth,m_nImageHeight,vp[0],vp[1])*/)
	{
		Clear();
	}

	// Rather dodgy motion model alpha/beta filter (ish)
	m_sTarget.vVelocity = 0.5*m_sTarget.vVelocity+0.5*(vp-vplast);
	m_sTarget.vVelocity[2] = 0.0;
	m_sTarget.vVelocity[3] = 0.0;

}

//---------------------------------------------------------------------------------

void PWPTracker::_ComputePFPBFromImage(const cv::Mat& cImage)
{
	if(!m_bHasTarget)
		return;

	// Image info
	ImageInfo* pII = m_pasImageInfo;

	Histogram* const pcBH = m_pcBackHist;
	Histogram* const pcFH = m_sTarget.pcForeHist;
    
	const double fUniform = 1.0/double(pcBH->Bins());

	const double fPriorBackground = m_sAS.fPriorBackground;
	const double fPriorForeground = (1.0-m_sAS.fPriorBackground);


    for (int r=0; r<cImage.rows; r++)for (int c=0; c<cImage.cols; c++)
	{
        const cv::Vec3b &pix = cImage.at<cv::Vec3b>(r,c);
		pII->LHB = fPriorBackground*(pcBH->GetBinVal(pix[0], pix[1], pix[2]) + fUniform);
		pII->LHF = fPriorForeground*(pcFH->GetBinVal(pix[0], pix[1], pix[2]) + fUniform);
		pII++;
	}



}

//---------------------------------------------------------------------------------

void PWPTracker::_ComputeCostImageFromHistogramsAndEdgeMap(const cv::Mat& cImage, Eigen::Matrix3f* pmWDC /*= NULL*/)
{
	Histogram* const pcFH = m_sTarget.pcForeHist;
	Histogram* const pcBH = m_pcBackHist;
	const double fUniform = 1.0/m_pcBackHist->Bins();

	// Image info
	const int imagewidth = m_nImageWidth;
	const int imageheight = m_nImageHeight;
	ImageInfo* const pII = m_pasImageInfo;
	
	// Level set
	const int rowlength0 = m_sTarget.nBGWidth;
	LevelSetInfo* pLS0 = m_sTarget.pasLevelSet;

	Pose& vp = m_sTarget.vPose;

	// Build warp matrix
	Eigen::Matrix3f mW = _BuildWarpMatrix(vp);
	
	if(pmWDC)
		mW = mW*(*pmWDC);

	// Compute warps
	const double W_00 = mW(0,0);
	const double W_01 = mW(0,1);
	const double W_02 = mW(0,2);
	const double W_10 = mW(1,0);
	const double W_11 = mW(1,1);
	const double W_12 = mW(1,2);

	const double WDC_00 = pmWDC?(*pmWDC)(0,0):1.0;
	const double WDC_01 = pmWDC?(*pmWDC)(0,1):0.0;
	const double WDC_02 = pmWDC?(*pmWDC)(0,2):0.0;
	const double WDC_10 = pmWDC?(*pmWDC)(1,0):0.0;
	const double WDC_11 = pmWDC?(*pmWDC)(1,1):1.0;
	const double WDC_12 = pmWDC?(*pmWDC)(1,2):0.0;

    const int nWI = cImage.cols;
    const int nHI = cImage.rows;

	const int ybegin0 = -m_sTarget.nBGHalfHeight;
	const int yend0 = m_sTarget.nBGHalfHeight;
	const int xbegin0 = -m_sTarget.nBGHalfWidth;
	const int xend0 =m_sTarget.nBGHalfWidth;

	const int MFGHalfHeight = -m_sTarget.nFGHalfHeight;
	const int FGHalfHeight = m_sTarget.nFGHalfHeight;
	const int MFGHalfWidth = -m_sTarget.nFGHalfWidth;
	const int FGHalfWidth = m_sTarget.nFGHalfWidth;

	for(int r=0,y=ybegin0; y<=yend0; r++, y++)
	{
		for(int c=0,x=xbegin0; x<=xend0; c++, x++)
		{
			LevelSetInfo* pCLS = pLS0+r*rowlength0+c;
					
			if(y<ybegin0+BGBORDERSIZE || y>yend0-BGBORDERSIZE || x<xbegin0+BGBORDERSIZE || x>xend0-BGBORDERSIZE)
			{

				pCLS->PF = 0.0;
				pCLS->PB = 1.0;
				continue;
			}

			const int ci = int(W_00*x+W_01*y+W_02+0.5);
			const int ri = int(W_10*x+W_11*y+W_12+0.5);

			if(_InsideImage(nWI,nHI,ci,ri))
			{		
				pCLS->PF = pII[ri*imagewidth+ci].LHF;
				pCLS->PB = pII[ri*imagewidth+ci].LHB;
			}
			else
			{
				pCLS->PF = 0.0;
				pCLS->PB = fUniform;
			}
		}
	}
}

//---------------------------------------------------------------------------------

void PWPTracker::_InitialiseLevelSet()
{
	// Level set
	const int rowlength = m_sTarget.nBGWidth;
	LevelSetInfo* pLS = m_sTarget.pasLevelSet;

	// Initialise level set
	double fMaxU = -1e30;
	double fMinU = 1e30;

	double aa = double(m_sTarget.nFGHalfWidth)*double(m_sTarget.nFGHalfWidth);
	double bb = double(m_sTarget.nFGHalfHeight)*double(m_sTarget.nFGHalfHeight);
	double fScale = 0.5*min(double(m_sTarget.nFGHalfWidth),double(m_sTarget.nFGHalfHeight));

	for(int r=0,cr=-m_sTarget.nBGHalfHeight; cr<=m_sTarget.nBGHalfHeight; r++, cr++)
	{
		for(int c=0,cc=-m_sTarget.nBGHalfWidth; cc<=m_sTarget.nBGHalfWidth; c++, cc++)
		{
			LevelSetInfo* pCLS = pLS+r*rowlength+c;

			double fRad = sqrt(double(cr*cr)/bb+double(cc*cc)/aa)-1;
			const double fU = -fRad*fScale;

			//const double fU = (pCLS->PF-pCLS->PB);
			pCLS->U = fU;

			if(fU>fMaxU)
				fMaxU = fU;
			if(fU<fMinU)
				fMinU = fU;
		}
	}

	m_sTarget.fMaxU = fMaxU;
	m_sTarget.fMinU = fMinU;

	LevelSetInfo* pLS0 = m_sTarget.pasLevelSet;
	const int rowlength0 = m_sTarget.nBGWidth;
	const int rbegin0 = 1;
	const int ybegin0 = -m_sTarget.nBGHalfHeight+1;
	const int yend0 = m_sTarget.nBGHalfHeight-1;
	const int cbegin0 = 1;
	const int xbegin0 = -m_sTarget.nBGHalfWidth+1;
	const int xend0 = m_sTarget.nBGHalfWidth-1;

	// Compute Derivatives of Level Set			
	for(int r=rbegin0,_y=ybegin0; _y<=yend0; r++, _y++)
	{
		for(int c=cbegin0,_x=xbegin0; _x<=xend0; c++, _x++)
		{
			LevelSetInfo* pCLS = pLS0 + r*rowlength0 + c;

			// Derivative
			double fUDX = (pLS0[r*rowlength0+(c+1)].U-pLS0[r*rowlength0+(c-1)].U)*0.5;
			double fUDY = (pLS0[(r+1)*rowlength0+c].U-pLS0[(r-1)*rowlength0+c].U)*0.5;

			// Magnitude of derivative
			double fNormDU = 1.0/(sqrt(fUDX*fUDX+fUDY*fUDY)+1e-30);
			
			// Normalised derivative
			double fNUDX = fUDX*fNormDU;
			double fNUDY = fUDY*fNormDU;

			// Copy into structure
			pCLS->DX = fUDX;
			pCLS->DY = fUDY;
			pCLS->NDX = fNUDX;
			pCLS->NDY = fNUDY;
		}
	}
}

//---------------------------------------------------------------------------------

void PWPTracker::_EnforceNeumannBoundaryConditions()
{
	const int nW = m_sTarget.nBGWidth;
	const int nH = m_sTarget.nBGHeight;
	// Level set
	const int rowlength = m_sTarget.nBGWidth;
	LevelSetInfo* pLS = m_sTarget.pasLevelSet;
	
	for(int r=0; r<nH; r++)
	{
		pLS[r*rowlength].U = pLS[r*rowlength+1].U;
		pLS[r*rowlength+nW-1].U = pLS[r*rowlength+nW-2].U;
	}

	for(int c=0; c<nW; c++)
	{
		pLS[c].U = pLS[rowlength+c].U;
		pLS[(nH-1)*rowlength+c].U = pLS[(nH-2)*rowlength+c].U;
	}
}

//---------------------------------------------------------------------------------

void PWPTracker::_IterateLevelSet(int nIterations, double fTimestep)
{
	double fMu;

	if (m_sAS.fMu*fTimestep>0.24)
		fMu = 0.24 / fTimestep;
	else
		fMu = m_sAS.fMu;

	Eigen::Matrix3f mW = _BuildWarpMatrix(m_sTarget.vPose);

	LevelSetInfo* pLS0 = m_sTarget.pasLevelSet;
	const int rowlength0 = m_sTarget.nBGWidth;

	const int rbegin0 = 1;
	const int ybegin0 = -m_sTarget.nBGHalfHeight + 1;
	const int yend0 = m_sTarget.nBGHalfHeight - 1;
	const int cbegin0 = 1;
	const int xbegin0 = -m_sTarget.nBGHalfWidth + 1;
	const int xend0 = m_sTarget.nBGHalfWidth - 1;


	// Iterate PDEs
	for (int nIt = 0; nIt < nIterations; nIt++)
	{
		_EnforceNeumannBoundaryConditions();

		// Prior for different model types
		double fNf = 0.0;
		double fNb = 0.0;

		// Compute Derivatives of Level Set			
		for (int r = rbegin0, _y = ybegin0; _y <= yend0; r++, _y++)
		{
			for (int c = cbegin0, _x = xbegin0; _x <= xend0; c++, _x++)
			{
				LevelSetInfo* pCLS = pLS0 + r*rowlength0 + c;

				// Heaviside step function (blurred)
				pCLS->H = _Heaviside(pCLS->U, m_sAS.fEpsilon);

				// Nf and Nb used to give prior prob. of foreground or background
				fNf += pCLS->H;
				fNb += 1.0 - pCLS->H;

				// Derivative
				double fUDX = (pLS0[r*rowlength0 + (c + 1)].U - pLS0[r*rowlength0 + (c - 1)].U)*0.5;
				double fUDY = (pLS0[(r + 1)*rowlength0 + c].U - pLS0[(r - 1)*rowlength0 + c].U)*0.5;

				// Magnitude of derivative
				double fNormDU = 1.0 / (sqrt(fUDX*fUDX + fUDY*fUDY) + 1e-30);

				// Normalised derivative
				double fNUDX = fUDX*fNormDU;
				double fNUDY = fUDY*fNormDU;

				// Copy into structure
				pCLS->DX = fUDX;
				pCLS->DY = fUDY;
				pCLS->NDX = fNUDX;
				pCLS->NDY = fNUDY;
			}
		}

		const double fN = fNf + fNb;

		// Evolve Level Set
		for (int r = rbegin0+1, _y = ybegin0+1; _y <= yend0-1; r++, _y++)
		{
			for (int c = cbegin0+1, _x = xbegin0+1; _x <= xend0-1; c++, _x++)
			{
				LevelSetInfo* pCLS = pLS0 + r*rowlength0 + c;

				// Dirac
				double fDiracU = _Dirac(pCLS->U, m_sAS.fEpsilon);

				// Compute curvature
				const double fDNX = (pLS0[r*rowlength0 + (c + 1)].NDX - pLS0[r*rowlength0 + (c - 1)].NDX) / 2.0f;
				const double fDNY = (pLS0[(r + 1)*rowlength0 + c].NDY - pLS0[(r - 1)*rowlength0 + c].NDY) / 2.0f;
				const double fK = fDNX + fDNY;

				const double fDel2U = -(8.0f*pLS0[r*rowlength0 + c].U - (pLS0[(r - 1)*rowlength0 + c].U +
					pLS0[(r + 1)*rowlength0 + c].U +
					pLS0[r*rowlength0 + (c - 1)].U +
					pLS0[r*rowlength0 + (c + 1)].U +
					pLS0[(r - 1)*rowlength0 + (c + 1)].U +
					pLS0[(r - 1)*rowlength0 + (c - 1)].U +
					pLS0[(r + 1)*rowlength0 + (c + 1)].U +
					pLS0[(r + 1)*rowlength0 + (c - 1)].U));

				const double fJacNumer = (pCLS->PF - pCLS->PB);
				const double fJacDenom = (pCLS->PF*pCLS->H + pCLS->PB*(1.0f - pCLS->H));

				double fWeightedDataTerm;

				if (m_sAS.eOpinionPoolType == OP_LOGARITHMIC)
					fWeightedDataTerm = m_sAS.fPsi*fDiracU*fJacNumer / fJacDenom;
				else
					fWeightedDataTerm = m_sAS.fPsi*fDiracU*fJacNumer / ((pCLS->PF*fNf + pCLS->PB*fNb) / fN);

				const double fPenalizingTerm = fMu*(1.0*fDel2U - fK);

				pCLS->U += fTimestep*(fWeightedDataTerm + fPenalizingTerm);
			}
		}
	}
	// Compute useful things for drawing level set
	double fMaxU = -1e30;
	double fMinU = 1e30;
	double fNormU = 0.0f;
	const int lssize = m_sTarget.nBGWidth*m_sTarget.nBGHeight;
	for(int p=0;p<lssize; p++)
	{
		double fU = (pLS0+p)->U;
		if(fU>fMaxU)
			fMaxU = fU;
		if(fU<fMinU)
			fMinU = fU;
		if(fU>0.0f)
			fNormU += fU;
	}	
	m_sTarget.fMaxU = fMaxU;
	m_sTarget.fMinU = fMinU;
	m_sTarget.fNormU = fNormU;
}

//---------------------------------------------------------------------------------

void PWPTracker::_ComputeHistogramsUsingSegmentation(const cv::Mat& cImage)
{
	Histogram* const pcBH = m_pcBackHist;
	bool bBlendBackground = m_sSS.fAppearanceLearningRate<1.0;

	if(bBlendBackground)
		pcBH->BeginAddingNewSamples();
	else
		pcBH->Clear();

	Histogram* const pcFH = m_sTarget.pcForeHist;
	bool bBlendForeground = m_sSS.fAppearanceLearningRate<1.0;
	
	if(bBlendForeground)
		pcFH->BeginAddingNewSamples();
	else
		pcFH->Clear();

	// Level set
	const int rowlength0 = m_sTarget.nBGWidth;
	LevelSetInfo* pLS0 = m_sTarget.pasLevelSet;

	// Build warp matrix
	const Eigen::Matrix3f mW = _BuildWarpMatrix(m_sTarget.vPose);

	// Compute warps
	const double W_00 = mW(0, 0);
	const double W_01 = mW(0, 1);
	const double W_02 = mW(0, 2);
	const double W_10 = mW(1, 0);
	const double W_11 = mW(1, 1);
	const double W_12 = mW(1, 2);

    const int nWI = cImage.cols;
    const int nHI = cImage.rows;

	const int ybegin0 = -m_sTarget.nBGHalfHeight;
	const int yend0 = m_sTarget.nBGHalfHeight;
	const int xbegin0 = -m_sTarget.nBGHalfWidth;
	const int xend0 =m_sTarget.nBGHalfWidth;

	for(int r=0, y=ybegin0; y<=yend0; r++,y++)
	{
		for(int c=0,x=xbegin0; x<=xend0; c++,x++)
		{
			const int ci = int(W_00*x+W_01*y+W_02+0.5);
			const int ri = int(W_10*x+W_11*y+W_12+0.5);

			const bool bInsideImage = _InsideImage(nWI,nHI,ci,ri);
			
			if(!bInsideImage)
				continue;

            const cv::Vec3b& pix = cImage.at<cv::Vec3b>(ri,ci);
            const unsigned char rr = pix[0], gg = pix[1], bb = pix[2];

			double fH = _Heaviside(pLS0[r*rowlength0+c].U,m_sAS.fEpsilon);

			if(bBlendForeground)
				pcFH->AddNewSample(rr,gg,bb,fH);
			else
				pcFH->AddSample(rr,gg,bb,fH);

			if(bBlendBackground)
				pcBH->AddNewSample(rr,gg,bb,1.0-fH);
			else
				pcBH->AddSample(rr,gg,bb,1.0-fH);
		}
	}
	if(bBlendForeground)
		pcFH->CommitNewSamples(m_sSS.fAppearanceLearningRate);
	else
		pcFH->Normalise();

	if(bBlendBackground)
		pcBH->CommitNewSamples(m_sSS.fAppearanceLearningRate);
	else
		pcBH->Normalise();
}

//---------------------------------------------------------------------------------

void PWPTracker::AddTarget(cv::Mat& cImage, PWPBoundingBox cBB)
{
    int nW = cImage.cols;
    int nH = cImage.rows;
    
	// Check we are inside the image
	bool bInsideImage = _InsideImage(nW, nH, cBB.TLX-BGBORDERSIZE, cBB.BRY+BGBORDERSIZE);
	bInsideImage &= _InsideImage(nW, nH, cBB.TLX-BGBORDERSIZE, cBB.TLY-BGBORDERSIZE);
	bInsideImage &= _InsideImage(nW, nH, cBB.BRX+BGBORDERSIZE, cBB.TLY-BGBORDERSIZE);
	bInsideImage &= _InsideImage(nW, nH, cBB.BRX+BGBORDERSIZE, cBB.BRY+BGBORDERSIZE);

	if(!bInsideImage)
		return;

	// Initialise sizes that get used in loops
	double fWidth = cBB.BRX-cBB.TLX+1;
	double fHeight = cBB.BRY-cBB.TLY+1;
	
	double fScale = 1.0f;
	if(fWidth<fHeight)
		fScale = fWidth/double(PATCHSIZE);
	else
		fScale = fHeight/double(PATCHSIZE);

	m_sTarget.fOrigScale = fScale;

	fWidth /= fScale;
	fHeight /= fScale;

	m_sTarget.nFGHalfWidth = ceil(fWidth/2.0f);
	m_sTarget.nFGHalfHeight = ceil(fHeight/2.0f);
	m_sTarget.nBGHalfWidth = m_sTarget.nFGHalfWidth+BGBORDERSIZE;
	m_sTarget.nBGHalfHeight = m_sTarget.nFGHalfHeight+BGBORDERSIZE;
	m_sTarget.nBGWidth = 2*m_sTarget.nBGHalfWidth+1;
	m_sTarget.nBGHeight = 2*m_sTarget.nBGHalfHeight+1;

	// Add corners
	Point cPoint; cPoint[2] = 1.0;
	cPoint[0] = -m_sTarget.nFGHalfWidth-10;		cPoint[1] = -m_sTarget.nFGHalfHeight-10;	m_sTarget.vFGCorners.push_back(cPoint);
	cPoint[0] = m_sTarget.nFGHalfWidth+10;		cPoint[1] = -m_sTarget.nFGHalfHeight-10;	m_sTarget.vFGCorners.push_back(cPoint);
	cPoint[0] = m_sTarget.nFGHalfWidth+10;		cPoint[1] = m_sTarget.nFGHalfHeight+10;		m_sTarget.vFGCorners.push_back(cPoint);
	cPoint[0] = -m_sTarget.nFGHalfWidth-10;		cPoint[1] = m_sTarget.nFGHalfHeight+10;		m_sTarget.vFGCorners.push_back(cPoint);
	
	cPoint[0] = -m_sTarget.nBGHalfWidth;		cPoint[1] = -m_sTarget.nBGHalfHeight;		m_sTarget.vBGCorners.push_back(cPoint);
	cPoint[0] = m_sTarget.nBGHalfWidth;			cPoint[1] = -m_sTarget.nBGHalfHeight;		m_sTarget.vBGCorners.push_back(cPoint);
	cPoint[0] = m_sTarget.nBGHalfWidth;			cPoint[1] = m_sTarget.nBGHalfHeight;		m_sTarget.vBGCorners.push_back(cPoint);
	cPoint[0] = -m_sTarget.nBGHalfWidth;		cPoint[1] = m_sTarget.nBGHalfHeight;		m_sTarget.vBGCorners.push_back(cPoint);

	// Used for 3D view
	m_sDS.fRange = 1.5*max(m_sTarget.nBGWidth,m_sTarget.nBGHeight);

	// Create histogram distributions and stuff shared between targets
	if(!m_bHasTarget)
	{
		m_pcBackHist = new Histogram(m_sAS.nBitsY,m_sAS.nBitsU,m_sAS.nBitsV,m_sAS.fSigma);
		m_pasImageInfo = new ImageInfo[nH*nW];
		for(int p=0;p<nH*nW;p++)
			m_pasImageInfo[p].LHF=0.0;
		m_nImageWidth = nW;
		m_nImageHeight = nH;
	}

	m_sTarget.pcForeHist = new Histogram(m_sAS.nBitsY,m_sAS.nBitsU,m_sAS.nBitsV,m_sAS.fSigma);

	m_sTarget.pasLevelSet = new LevelSetInfo[m_sTarget.nBGHeight*m_sTarget.nBGWidth];
	m_sTarget.pmTemplate = new cv::Mat(m_sTarget.nBGHalfHeight, m_sTarget.nBGHalfWidth, cv::DataType<float>::type);
	m_sTarget.fTargetScore = m_sTarget.fLastTargetScore = 0.0f;
	m_sTarget.nIterations = 0;
	m_sTarget.bLost = false;
	m_sTarget.fForegroundProb = 1.0f;
	m_sTarget.fBackgroundProb = 1.0f;
	m_sTarget.ID = 0;

	m_sTarget.vPose.resize(MAXPS);
	m_sTarget.vVelocity.resize(MAXPS);
	m_sTarget.vPose.setZero();
	m_sTarget.vVelocity.setZero();
	
		
	// Initialise pose vector
	Pose& vp = m_sTarget.vPose;

	double fTX = (cBB.BRX+cBB.TLX)/2.0f;
	double fTY = (cBB.TLY+cBB.BRY)/2.0f;
	double fTheta = 0.0f;
	m_sTarget.fAspectRatio = 1.0f;

	// Initialise pose parameters
    vp.setZero();

	vp[0] = fTX;
	vp[1] = fTY;

	if(m_sAS.eModelType == MODEL_TS)
	{
		vp[2] = fScale-1.0f;
	}
	else if(m_sAS.eModelType == MODEL_TSR)
	{
		vp[2] =	fScale-1.0f;
		vp[3] = fTheta;
	}
	else if(m_sAS.eModelType == MODEL_AFFINE)
	{
		// Using Baker Matthews group parameterisation for affine model
		vp[2] = fScale*cos(fTheta)-1.0;
		vp[3] = fScale*sin(fTheta);
		vp[4] = -fScale*sin(fTheta);
		vp[5] = fScale*cos(fTheta)-1.0;
	}
	else if(m_sAS.eModelType == MODEL_HOMOGRAPHY)
	{
		// Using Baker Matthews group parameterisation for homography model
		vp[2] = fScale*cos(fTheta)-1.0;
		vp[3] = fScale*sin(fTheta);
		vp[4] = -fScale*sin(fTheta);
		vp[5] = fScale*cos(fTheta)-1.0;
	}

	// Build warp matrix
	Eigen::Matrix3f mW = _BuildWarpMatrix(vp);
	
	//cout << vp;

	Histogram* const pcFH = m_sTarget.pcForeHist;
	Histogram* const pcBH = m_pcBackHist;
	pcFH->Clear();
	pcBH->Clear();


	for(int cr=-m_sTarget.nBGHalfHeight; cr<=m_sTarget.nBGHalfHeight; cr++)
	{
		for(int cc=-m_sTarget.nBGHalfWidth; cc<=m_sTarget.nBGHalfWidth; cc++)
		{
			const double fC = mW(0,0)*cc+mW(0,1)*cr+mW(0,2);
			const double fR = mW(1,0)*cc+mW(1,1)*cr+mW(1,2);
			const int nC = int(fC+0.5);
			const int nR = int(fR+0.5);

			if(!_InsideImage(nW,nH,nC,nR))
				continue;
            
            cv::Vec3b &cPix = cImage.at<cv::Vec3b>(nR,nC);

			if(cr>-m_sTarget.nFGHalfHeight && cr<m_sTarget.nFGHalfHeight && cc>-m_sTarget.nFGHalfWidth && cc<m_sTarget.nFGHalfWidth) // Inside box
			{
				pcFH->AddSample(cPix[0],cPix[1],cPix[2],1.0);
			}
			else
			{
				pcBH->AddSample(cPix[0],cPix[1],cPix[2],1.0);
			}
		}
	}

	pcFH->Normalise();
	pcBH->Normalise();

	// Ignore the prior for a new target
	float fOldAppearanceLRate = m_sSS.fAppearanceLearningRate;
	m_sSS.fAppearanceLearningRate = 0.4;

	// Increment the number of targets
	m_bHasTarget = true;
	_InitialiseLevelSet();

	if(!m_sAS.bUseSegmentation)
	{
		_ComputeHistogramsUsingSegmentation(cImage);
		_ComputePFPBFromImage(cImage);
		_ComputeCostImageFromHistogramsAndEdgeMap(cImage);
		m_sSS.fAppearanceLearningRate = fOldAppearanceLRate;
		return;
	}
	for(int i=0;i<2;i++)
	{
		_ComputeHistogramsUsingSegmentation(cImage);
		_ComputePFPBFromImage(cImage);
		_ComputeCostImageFromHistogramsAndEdgeMap(cImage);
		_IterateLevelSet(25,2.0);
	}

	_ComputeHistogramsUsingSegmentation(cImage);
	_ComputePFPBFromImage(cImage);
	_ComputeCostImageFromHistogramsAndEdgeMap(cImage);

	m_sSS.fAppearanceLearningRate = fOldAppearanceLRate;
}

//---------------------------------------------------------------------------------

void PWPTracker::Process(cv::Mat& cImage)
{
	if(m_sDS.bAnimateLevelSet)
	{
		m_sDS.fTheta = DegToRad(-90.0+50.0*sin(m_fLSAnimationTime));
		m_sDS.fPhi = DegToRad(90.0*cos(1.0*m_fLSAnimationTime));
		m_fLSAnimationTime += 6.28/15.0/30.0;
	}
	
	if(m_sDS.bDiscretizeImage)
	{
		_DiscretizeImage(cImage);
	}
	else
	{
		_DELETE(m_pcDiscretizedImage);
	}

	if(!m_bHasTarget)
		return;

	_ComputePFPBFromImage(cImage);
	_IterateShapeAlignment(cImage);
	
	if(!m_bHasTarget)
		return;

	if(m_sAS.bUseSegmentation)
	{
		if(m_sAS.bUseDriftCorrection)
		{
			Eigen::Matrix3f mWDC = _ComputeDriftCorrection();
			_ComputeCostImageFromHistogramsAndEdgeMap(cImage, &mWDC);
		}
		else
		{
			_ComputeCostImageFromHistogramsAndEdgeMap(cImage);
		}


		_IterateLevelSet(m_sAS.nShapeIterations,m_sSS.fShapeLearningRate);
		_ComputeHistogramsUsingSegmentation(cImage);
	}

	if(m_sDS.bClassifyWholeImage)
	{
		_ClassifyImage(cImage);
	}
}

//---------------------------------------------------------------------------------

bool PWPTracker::GetSnapshotOfObject(cv::Mat& cImage, cv::Mat& cSnapshot)
{
	if(!m_bHasTarget)
		return false;

    const int nWI = cImage.cols;
    const int nHI = cImage.rows;

	// Build warp matrix
	const Eigen::Matrix3f mW = _BuildWarpMatrix(m_sTarget.vPose);

    if (cSnapshot.cols!=m_sTarget.nBGWidth || cSnapshot.rows!=m_sTarget.nBGHeight) {
        cSnapshot.create(m_sTarget.nBGHeight, m_sTarget.nBGWidth , CV_8UC3);
    }

	const double W_00 = mW(0, 0);
	const double W_01 = mW(0, 1);
	const double W_02 = mW(0, 2);
	const double W_10 = mW(1, 0);
	const double W_11 = mW(1, 1);
	const double W_12 = mW(1, 2);
	
	const int ybegin0 = -m_sTarget.nBGHalfHeight;
	const int yend0 = m_sTarget.nBGHalfHeight;
	const int xbegin0 = -m_sTarget.nBGHalfWidth;
	const int xend0 = m_sTarget.nBGHalfWidth;


	for(int r=0, y=ybegin0; y<=yend0; r++,y++)
	{
		for(int c=0,x=xbegin0; x<=xend0; c++,x++)
		{
			const double ci = W_00*x+W_01*y+W_02;
			const double ri = W_10*x+W_11*y+W_12;

			// Find colour by bi-linear interpolation
			const int x0=(int)ci;
			const int y0=(int)ri;

			if(x0 >=  nWI - 1 || x0 <= 0 || y0 >=  nHI - 1 || y0 <= 0)
			{
                cSnapshot.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,0);
				continue;
			}

			const double xfrac=ci-x0;
			const double yfrac=ri-y0;

			double mm_frac = (1-xfrac) * (1-yfrac);
			double mp_frac = (1-xfrac) * (yfrac);
			double pm_frac = (xfrac)   * (1-yfrac);
			double pp_frac = (xfrac)   * (yfrac);
  
            const cv::Vec3b& mm = cImage.at<cv::Vec3b>(y0,x0);
            const cv::Vec3b& pm = cImage.at<cv::Vec3b>(y0,x0+1);
            const cv::Vec3b& mp = cImage.at<cv::Vec3b>(y0+1,x0);
            const cv::Vec3b& pp = cImage.at<cv::Vec3b>(y0+1,x0+1);
            
            cv::Vec3b& ssptr = cSnapshot.at<cv::Vec3b>(r,c);
            ssptr[0] =(mm_frac*mm[0] + mp_frac*mp[0] + pm_frac*pm[0] + pp_frac*pp[0]);
            ssptr[1] =(mm_frac*mm[1] + mp_frac*mp[1] + pm_frac*pm[1] + pp_frac*pp[1]);
            ssptr[2] =(mm_frac*mm[2] + mp_frac*mp[2] + pm_frac*pm[2] + pp_frac*pp[2]);
        }
	}

	return true;
}

//---------------------------------------------------------------------------------

bool PWPTracker::GetPose(Pose &vPose)
{
	if(!m_bHasTarget)
		return false;

	vPose =  m_sTarget.vPose;

	vPose[0] -= m_nImageWidth/2;
	vPose[1] -= m_nImageHeight/2;
	vPose[2] = (1.0+vPose[2]);
	vPose[0] *= -1.0;

	return true;
}

//---------------------------------------------------------------------------------


bool PWPTracker::DrawTargeLSOverlay(cv::Mat& cImage)
{
	if (!m_bHasTarget)
		return false;

    const int nWI = cImage.cols;
    const int nHI = cImage.rows;
    
	// Build warp matrix
	const Eigen::Matrix3f mW = _BuildWarpMatrix(m_sTarget.vPose);

	const double W_00 = mW(0, 0);
	const double W_01 = mW(0, 1);
	const double W_02 = mW(0, 2);
	const double W_10 = mW(1, 0);
	const double W_11 = mW(1, 1);
	const double W_12 = mW(1, 2);

	const int ybegin0 = -m_sTarget.nBGHalfHeight;
	const int yend0 = m_sTarget.nBGHalfHeight;
	const int xbegin0 = -m_sTarget.nBGHalfWidth;
	const int xend0 = m_sTarget.nBGHalfWidth;

	LevelSetInfo* pLS0 = m_sTarget.pasLevelSet;
	const int rowlength0 = m_sTarget.nBGWidth;

	for (int r = 0, y = ybegin0; y <= yend0; r++, y++)
	{
		for (int c = 0, x = xbegin0; x <= xend0; c++, x++)
		{
			const double fU = pLS0[r*rowlength0 + c].U;
			if (abs(fU) > 2)
				continue;

			const double ci = W_00*x + W_01*y + W_02;
			const double ri = W_10*x + W_11*y + W_12;

			// Find colour by bi-linear interpolation
			const int x0 = (int)ci;
			const int y0 = (int)ri;

			if (x0 >= nWI - 2 || x0 <= 1 || y0 >= nHI - 2 || y0 <= 1)continue;
            cImage.at<cv::Vec3b>(y0,x0) = cv::Vec3b(0,0,255);
            cImage.at<cv::Vec3b>(y0+1,x0) = cv::Vec3b(0,0,255);
            cImage.at<cv::Vec3b>(y0,x0+1) = cv::Vec3b(0,0,255);
            cImage.at<cv::Vec3b>(y0-1,x0) = cv::Vec3b(0,0,255);
            cImage.at<cv::Vec3b>(y0,x0-1) = cv::Vec3b(0,0,255);
		}
	}


	return true;
}

void PWPTracker::GetBoundingBox(vector<Eigen::Vector2f>& ptlist)
{
	const int ybegin0 = -m_sTarget.nBGHalfHeight;
	const int yend0 = m_sTarget.nBGHalfHeight;
	const int xbegin0 = -m_sTarget.nBGHalfWidth;
	const int xend0 = m_sTarget.nBGHalfWidth;

	const Eigen::Matrix3f mW = _BuildWarpMatrix(m_sTarget.vPose);

	const double W_00 = mW(0, 0);
	const double W_01 = mW(0, 1);
	const double W_02 = mW(0, 2);
	const double W_10 = mW(1, 0);
	const double W_11 = mW(1, 1);
	const double W_12 = mW(1, 2);

	float tlc = W_00*xbegin0 + W_01*ybegin0 + W_02;
	float tlr = W_10*xbegin0 + W_11*ybegin0 + W_12;
	float brc = W_00*xend0 + W_01*yend0 + W_02;
	float brr = W_10*xend0 + W_11*yend0 + W_12;

	float trc = W_00*xend0 + W_01*ybegin0 + W_02;
	float trr = W_10*xend0 + W_11*ybegin0 + W_12;
	float blc = W_00*xbegin0 + W_01*yend0 + W_02;
	float blr = W_10*xbegin0 + W_11*yend0 + W_12;

    ptlist[0] = Eigen::Vector2f(tlc, tlr);
    ptlist[1] = Eigen::Vector2f(trc, trr);
    ptlist[2] = Eigen::Vector2f(brc, brr);
    ptlist[3] = Eigen::Vector2f(blc, blr);
}

//---------------------------------------------------------------------------------