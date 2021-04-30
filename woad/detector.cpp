#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

const int imageWidth = 640;
const int imageHeight = 480;

Size imageSize = Size(imageWidth, imageHeight);

int i = 0;

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL, validROIR;

Mat mapLx, mapLy, mapRx, mapRy;
Mat Rl, Rr, Pl, Pr, Q;
Mat xyz;

// check
int blockSize = 7, uniquenessRatio = 20, numDisparities = 6;
Ptr<StereoBM> bm = StereoBM::create(16, 9);
Mat cameraMatrixL = (Mat_<double>(3, 3) << 5.727240639079388e+02, 0, 3.348927572177254e+02,\
                                            0, 7.642579107425479e+02, 2.524851683818865e+02,\
                                            0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << 5.739904007489756e+02, 0, 3.480807759167425e+02,\
                                            0, 7.661585268839706e+02, 2.594853798430605e+02,\
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.173879002658564, 0.499995463028069,\
                         -0.002931519453850, 0.007181465286153, \
                         0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.179788172339507, 0.646411891137831,\
                         -3.077025769343296e-04, 0.005189588649867,\
                         0.00000);
Mat T = (Mat_<double>(3, 1) << 60.489055441567714, -0.376046366004474, -1.123445669345643);
Mat R = (Mat_<double>(3, 3) << 1, -1.355832825473907e-04, 0.007163468273113,\
                              1.148828189388144e-04, 1, 0.002890058895127,\
                              -0.007163830153230, -0.002889161756095, 1);


void stereo_match(int, void*);

int main() 
{    
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    if(!cam0.isOpened())
    {
       printf("cam0 is not opened.\n");
       return -1;
    }
    if(!cam1.isOpened())
    {
       printf("cam1 is not opened.\n");
       return -1;
    }

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize,\
                     R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, 0, imageSize, \
                     &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1,\
                             mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1,\
                            mapRx, mapRy);
    while(1)
    {
        cam0 >> rgbImageL;
        cam1 >> rgbImageR;
        //-- 1. Read the images
        cvtColor(rgbImageL, grayImageL, COLOR_RGB2GRAY);
        cvtColor(rgbImageR, grayImageR, COLOR_RGB2GRAY);

        imshow("ImageL Before Rectify", grayImageL);
        //imshow("ImageR Before Rectify", grayImageR);

        // remap
        remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
        remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

        // show rectified result
        Mat rgbRectifyImageL, rgbRectifyImageR;
        cvtColor(rectifyImageL, rgbRectifyImageL, COLOR_GRAY2BGR);
        cvtColor(rectifyImageR, rgbRectifyImageR, COLOR_GRAY2BGR);
        imshow("ImageL After Rectify", rgbRectifyImageL);
        //imshow("ImageR After Rectify", rgbRectifyImageR);

        // stereo match
        namedWindow("disparity", cv::WINDOW_AUTOSIZE);
        createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
        createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
        createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
        stereo_match(0, 0);
        // //-- And create the image in which we will save our disparities
        // Mat imgDisparity16S = Mat( imgLeft.rows, imgLeft.cols, CV_16S );
        // Mat imgDisparity8U = Mat( imgLeft.rows, imgLeft.cols, CV_8UC1 );
        // if( !imgLeft.data || !imgRight.data )
        // {
        //     std::cout<< " --(!) Error reading images " << std::endl; 
        //     return -1; 
        // }
        // //-- 2. Call the constructor for StereoBM
        // int ndisparities = 16*5;   /**< Range of disparity */
        // int SADWindowSize = 21; /**< Size of the block window. Must be odd */
        // cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(ndisparities, SADWindowSize);
        // //-- 3. Calculate the disparity image
        // bm->compute(imgLeft, imgRight, imgDisparity16S);
        // //-- Check its extreme values
        // double minVal; double maxVal;
        // minMaxLoc( imgDisparity16S, &minVal, &maxVal );
        // printf("Min disp: %f Max value: %f \n", minVal, maxVal);

        // //-- 4. Display it as a CV_8UC1 image
        // namedWindow("imgLeft", WINDOW_AUTOSIZE);
        // namedWindow("imgRight", WINDOW_AUTOSIZE);
        // imshow("imgLeft", imgLeft);
        // imshow("imgRight", imgRight);
        // imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));
        // namedWindow("windowDisparity", WINDOW_NORMAL );
        // imshow("windowDisparity", imgDisparity8U );

	    if((char)waitKey(30) == 27)
		    break;

	    if((char)waitKey(30) == 32)
	    {
	      char lpic_Name[128] = {};
	      char rpic_Name[128] = {};
	        sprintf(lpic_Name, "RL%d.jpg", i);
                sprintf(rpic_Name, "RR%d.jpg", i);
                imwrite(lpic_Name, rectifyImageL);
                imwrite(rpic_Name, rectifyImageL);
		std::cout<<"--- take a picture ---" <<std::endl;
		i++;
		  }

    }

    return 0;
}


void stereo_match(int, void*)
{
    bm->setBlockSize(21); //2 * blockSize + 5 SAD window?
    bm->setROI1(validROIL);
    bm->setROI2(validROIR);
    bm->setPreFilterCap(31);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numDisparities * 16 + 16);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(uniquenessRatio);

    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);
    Mat disp, disp8, copyImage;
    bm->compute(rectifyImageL, rectifyImageR, disp);
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16) * 16.));
    reprojectImageTo3D(disp, xyz, Q, true);

    xyz = xyz * 16;
    imshow("disparity", disp8);
    copyImage = disp8.clone();
    imshow("conter", copyImage);

    Mat threshold_output;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    RNG rng(12345);
    threshold(copyImage, threshold_output, 20, 255, cv::THRESH_BINARY);
    findContours(threshold_output, contours, hierarchy, cv::RETR_TREE,\
             cv::CHAIN_APPROX_SIMPLE, Point(0, 0));
    // calculate hull
    vector<vector<Point> >hull(contours.size());
    vector<vector<Point> > result;

    for (int i = 0; i < contours.size(); ++i)
		convexHull(Mat(contours[i]), hull[i], false);

    Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

    for (int i = 0; i< contours.size(); ++i)
	{
		if (contourArea(contours[i]) < 500)//面积小于area的凸包，可忽略
			continue;
		result.push_back(hull[i]);
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}
	imshow("contours", drawing);
}
