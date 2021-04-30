#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<string>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d.hpp>
#include<opencv2/imgcodecs.hpp>
#include<sys/time.h>

using namespace cv;
using namespace std;

int SWS = 5;
int PFS = 5;
int preFiltCap = 29;
int minDisp = -25;
int numOfDisp = 128;
int TxtrThrshld = 100;
int unicRatio = 10;
int SpcklRng = 15;
int SpklWinSze = 100;

string folder_name = "/home/lbx/cse520s/zcyd";
string calibration_data_folder = "";

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL, validROIR;

Mat imgLeft, imgRight;
Mat mapLx, mapLy, mapRx, mapRy;
Mat Rl, Rr, Pl, Pr, Q;
Mat xyz;

// check
Mat cameraMatrixL = (Mat_<double>(3, 3) << 5.717662471089529e+02, 0, 3.599894403338598e+02,\
                                            0, 7.636306316305219e+02, 2.947943548027684e+02,\
                                            0, 0, 1);
Mat cameraMatrixR = (Mat_<double>(3, 3) << 5.679819209188780e+02, 0, 3.452859895943097e+02,\
                                            0, 7.609519703287510e+02, 2.895787685416111e+02,\
                                            0, 0, 1);
Mat distCoeffL = (Mat_<double>(5, 1) << -0.284128304263526, 0.603976907145943,\
                         0.002650751621832, 6.311318302912327e-04, \
                         0.00000);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.236473331555064, 0.319212543138987,\
                         0.002170239381114, -0.002597546865672,\
                         0.00000);
Mat T = (Mat_<double>(3, 1) << -60.513327359166034, -0.037255253186699, -2.189518833850237);
Mat R = (Mat_<double>(3, 3) << 1, -0.001208891549226, -0.002628955911107,\
                              0.001199053991384, 1, -0.003740364031551,\
                              0.002633457325662, 0.003737196112728, 1);

long long getTimestamp() {
    const std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    const std::chrono::microseconds epoch = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return  epoch.count();
}

void loadParams()
{
    fprintf(stderr, "Loading params...\n");
    string filename = folder_name + "/para/3dmap_set.yml";
    FileStorage fs;
    if (fs.open(filename, FileStorage::READ))
    {
        fs["SWS"] >> SWS;
        fs["PFS"] >> PFS;
        fs["preFiltCap"] >> preFiltCap;
        fs["minDisp"] >> minDisp;
        fs["numOfDisp"] >> numOfDisp;
        fs["TxtrThrshld"] >> TxtrThrshld;
        fs["unicRatio"] >> unicRatio;
        fs["SpcklRng"] >> SpcklRng;
        fs["SpklWinSze"] >> SpklWinSze;
    }

    fprintf(stderr, "pfs = %d\n", PFS);

}

void saveParams()
{
    fprintf(stderr, "Saving params...\n");
    string filename = folder_name + "/para/3dmap_set.yml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "SWS" << SWS << "PFS" << PFS << "preFiltCap" << preFiltCap << "minDisp" << minDisp << "numOfDisp" << numOfDisp
       << "TxtrThrshld" << TxtrThrshld << "unicRatio" << unicRatio  << "SpcklRng" << SpcklRng << "SpklWinSze" << SpklWinSze;
}

void stereo_depth_map(Mat left, Mat right)
{
    Ptr<StereoBM> bm = StereoBM::create(16,9);

    if (SWS < 5) SWS = 5;
    if (SWS %2 == 0) SWS += 1;
    if (SWS > left.rows) SWS = left.rows - 1;
    if (numOfDisp < 16) numOfDisp = 16;
    if (numOfDisp % 16 != 0) numOfDisp -= (numOfDisp %16);
    if (preFiltCap < 1) preFiltCap = 1;

    bm->setPreFilterCap(preFiltCap);
    bm->setBlockSize(SWS);
    bm->setMinDisparity(minDisp);
    bm->setNumDisparities(numOfDisp);
    bm->setTextureThreshold(TxtrThrshld);
    bm->setUniquenessRatio(unicRatio);
    bm->setSpeckleWindowSize(SpklWinSze);
    bm->setSpeckleRange(SpcklRng);
    bm->setDisp12MaxDiff(1);

    Mat disp, disp8, colored;
    bm->compute(left, right, disp);
    disp.convertTo(disp8, CV_8U);
    applyColorMap(disp8, colored, COLORMAP_JET);
    imshow("Image", colored);
}

void onTrackbar(int, void *)
{
    stereo_depth_map(rectifyImageL, rectifyImageR);
}

void onMinDisp(int, void *)
{
    minDisp -= 40;
    stereo_depth_map(rectifyImageL, rectifyImageR);
}

int main()
{
    string imageToDisp = "";
    int photo_width = 640;
    int photo_height = 480;
    int image_width = 320;
    int image_height = 240;

    Size imageSize(photo_width, photo_height);

    imgLeft = imread( "/home/lbx/cse520s/zcyd/photots/0L.jpg", IMREAD_GRAYSCALE );
    imgRight = imread( "/home/lbx/cse520s/zcyd/photots/0R.jpg", IMREAD_GRAYSCALE );
    imshow("Left Image", imgLeft);
    imshow("Right Image", imgRight);

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize,\
                     R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, 0, imageSize, \
                     &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1,\
                             mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1,\
                            mapRx, mapRy);

    loadParams();

    cout << "Load Parameters done.\n";

    namedWindow("Image");
    moveWindow("Image", 50, 100);
    namedWindow("Left");
    moveWindow("Left", 450, 100);
    namedWindow("Right");
    moveWindow("Right", 850, 100);

    createTrackbar("SWS", "Image", &SWS, 255, onTrackbar);
    createTrackbar("PFS", "Image", &PFS, 255, onTrackbar);
    createTrackbar("PreFiltCap", "Image", &preFiltCap, 63, onTrackbar);
    createTrackbar("MinDISP", "Image", &minDisp, 100, onMinDisp);
    createTrackbar("NumOfDisp", "Image", &numOfDisp, 256, onTrackbar);
    createTrackbar("TxtrThrshld", "Image", &TxtrThrshld, 100, onTrackbar);
    createTrackbar("UnicRatio", "Image", &unicRatio, 100, onTrackbar);
    createTrackbar("SpcklRng", "Image", &SpcklRng, 40, onTrackbar);
    createTrackbar("SpklWinSze", "Image", &SpklWinSze, 300, onTrackbar);

    cout << "Creat Trackbar done.\n";

    long long prevTime = getTimestamp();
    float avgFps = 0.0;
    int frameNumber = 0;
    int prevFrameNumber = 0;

    //cvtColor(imgLeft, grayImageL, COLOR_RGB2GRAY);
    //cvtColor(imgRight, grayImageR, COLOR_RGB2GRAY);
    cout << "cvtColor done.\n";

    //remap(imgLeft, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    //remap(imgRight, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
    rectifyImageL = imgLeft;
    rectifyImageR = imgRight;

    cout << "Remap done.\n";

    while (1)
    {
        stereo_depth_map(rectifyImageL, rectifyImageR);

        cv::imshow("Left", rectifyImageL);
        cv::imshow("Right", rectifyImageR);

        char k = waitKey(1);
        if( k == 's' || k == 'S')
        {
            saveParams();
            break;
        } else if (k == 'q' || k == 'Q') {
            break;
        }
    }

    return 0;

}


