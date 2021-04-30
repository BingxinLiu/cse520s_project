#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<string>
#include<iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/time.h>

using namespace cv;
using namespace std;

const int imageWidth = 640;
const int imageHeight = 480;

Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;
//Mat disp;

Rect validROIL, validROIR;

Mat mapLx, mapLy, mapRx, mapRy;
Mat Rl, Rr, Pl, Pr, Q;
Mat xyz;

// check
/*
int blockSize = 7, uniquenessRatio = 20, numDisparities = 6;
Ptr<StereoBM> bm = StereoBM::create(16, 9);
Mat cameraMatrixL = (Mat_<double>(3, 3) << 5.736947040260504e+02, 0, 3.516888262024591e+02,\
                                            0, 7.658060600986764e+02, 2.620069179519548e+02,\
                                            0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << 5.705016025218727e+02, 0, 3.334997798979939e+02,\
                                            0, 7.614210625682780e+02, 2.512762814190333e+02,\
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.184935292440814, 0.532348112473990,\
                         0.001266171295213, 0.007286364241261, \
                         0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.170287406105907, 0.343107627629609,\
                         -9.516072314441116e-04, 0.005781565605912,\
                         0.00000);
Mat T = (Mat_<double>(3, 1) << -5.999816214128122, 0.051075447909790, -0.159748306013202);
Mat R = (Mat_<double>(3, 3) << 1, 2.299383474454858e-04, 8.702142851212463e-04,\
                              -2.314604897792139e-04, 1, 0.001749461756376,\
                              -8.698106617685305e-04, -0.001749662467942, 1);
*/
int blockSize = 7, uniquenessRatio = 20, numDisparities = 6;
Ptr<StereoBM> bm = StereoBM::create(16, 9);
Mat cameraMatrixL = (Mat_<double>(3, 3) << 572.74899, 0, 341.11564,\
                                            0, 763.28184, 267.75341,\
                                            0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << 565.75134, 0, 324.57437,\
                                            0, 754.92681, 256.71093,\
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.16967, 0.45644,\
                         0.00300, 0.00046, \
                         0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.17735, 0.41591,\
                         -0.00156, -0.00038,\
                         0.00000);
Mat T = (Mat_<double>(3, 1) << -599.79459, 2.60377, -19.39452);
Mat rec = (Mat_<double>(3, 1) << -0.00119, -0.00003, -0.00036);
Mat R;

void stereoSGBM(Mat imageL, Mat imageR);

int main()
{
    struct timeval tx;
    struct timeval t1;
    struct timeval t2;

    // initialize camera
    VideoCapture cam0("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)10/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    VideoCapture cam1("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)10/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
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

    // rectify camera
    Rodrigues(rec, R);
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize,\
                     R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, 0, imageSize, \
                     &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1,\
                             mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1,\
                            mapRx, mapRy);

    namedWindow("disparity", cv::WINDOW_AUTOSIZE);
    //createTrackbar("BlockSize:\n", "disparity", &blockSize, 5);
    //createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 5);
    //createTrackbar("NumDisparities:\n", "disparity", &numDisparities, ((imageSize.width / 8) + 15) & -16);

    while(1)
    {
        gettimeofday(&tx,NULL);
        printf("获取的秒时间 = %ld  获取的微秒时间 =%ld\n",tx.tv_sec,tx.tv_usec);
        cam0 >> rgbImageL;
        cam1 >> rgbImageR;

        imshow("Image Left", rgbImageL);
        imshow("Image Right", rgbImageR);

        cvtColor(rgbImageL, grayImageL, COLOR_RGB2GRAY);
        cvtColor(rgbImageR, grayImageR, COLOR_RGB2GRAY);

        //remap
        gettimeofday(&t1,NULL);
        remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
        remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
        gettimeofday(&t2,NULL);
        printf("remap:: = %ld.%ld\n",t2.tv_sec - t1.tv_sec,t2.tv_usec - t1.tv_usec);
        
        // disp
        gettimeofday(&t1,NULL);
        stereoSGBM(rectifyImageL, rectifyImageR);
        gettimeofday(&t2,NULL);
        printf("disp:: = %ld.%ld\n",t2.tv_sec - t1.tv_sec,t2.tv_usec - t1.tv_usec);
        if((char)waitKey(30) == 27)
            break;


    }

    cam0.release();
    cam1.release();

    destroyAllWindows();

    return 0;
}

void stereoSGBM(Mat imageL, Mat imageR)
{
    Mat disp;
    Mat disp1 = Mat(imageL.rows, imageL.cols, CV_8UC1);
    Size imgSize = imageL.size();
    Ptr<StereoSGBM> sgbm = StereoSGBM::create();

    numDisparities = ((imgSize.width / 8) + 15) & -16;
    int numChannels = imageL.channels();
    uniquenessRatio = 5;
    blockSize = 5;

    sgbm->setPreFilterCap(31);
    sgbm->setBlockSize(blockSize);
    sgbm->setP1(8 * numChannels * blockSize * blockSize);
    sgbm->setP2(32 * numChannels * blockSize * blockSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numDisparities);
    sgbm->setUniquenessRatio(uniquenessRatio);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);
    sgbm->compute(imageL,imageR, disp);

    disp.convertTo(disp1, CV_8U, 255 / (numDisparities * 16.));
    imshow("Left Image", imageL);
    imshow("Right Image", imageR);
    imshow("disparity", disp1);

}