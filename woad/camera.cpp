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

/*
//with 3 K
int blockSize = 7, uniquenessRatio = 20, numDisparities = 6;
Ptr<StereoBM> bm = StereoBM::create(16, 9);
Mat cameraMatrixL = (Mat_<double>(3, 3) << 5.738197136020820e+02, 0, 3.515077569234052e+02,\
                                            0, 7.659429842802297e+02, 2.617643552455839e+02,\
                                            0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << 5.707369528753381e+02, 0, 3.340813602012821e+02,\
                                            0, 7.616964734842732e+02, 2.509332720128013e+02,\
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.239755370472363, 1.495438136117500,\
                         0.001161022479945, 0.007037029004273, \
                         -4.513042926203764);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.224190155804730, 1.139177746370474,\
                         -9.516072314441116e-04, 0.005781565605912,\
                         -2.804814177028355);
Mat T = (Mat_<double>(3, 1) << -6.000331884043730, 0.051088812661147, -0.160837484233406);
Mat R = (Mat_<double>(3, 3) << 1, 2.769942888600303e-04, -4.762543607774524e-04,\
                              -2.760941581223142e-04, 1, 0.001889049065815,\
                              4.767767486689596e-04, -0.001888917288063, 1);
*/

/*
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
*/

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
    //Rodrigues(rec, R);
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