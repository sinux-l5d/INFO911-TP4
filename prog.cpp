#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    Mat img_input, img_seg, img_d_bgr, img_d_hsv, img_d_lab;
    VideoCapture *pCap = nullptr;
    const int width = 640;
    const int height = 480;
    const int size = 50;
    // Ouvre la camera
    pCap = new VideoCapture(0);
    if (!pCap->isOpened())
    {
        cout << "Couldn't open image / camera ";
        return 1;
    }
    // Force une camera 640x480 (pas trop grande).
    pCap->set(CAP_PROP_FRAME_WIDTH, 640);
    pCap->set(CAP_PROP_FRAME_HEIGHT, 480);
    (*pCap) >> img_input;
    if (img_input.empty())
        return 1; // probleme avec la camera
    Point pt1(width / 2 - size / 2, height / 2 - size / 2);
    Point pt2(width / 2 + size / 2, height / 2 + size / 2);
    namedWindow("input", 1);
    imshow("input", img_input);
    bool freeze = false;
    while (true)
    {
        char c = (char)waitKey(50); // attend 50ms -> 20 images/s
        if (pCap != nullptr && !freeze)
            (*pCap) >> img_input; // récupère l'image de la caméra
        if (c == 27 || c == 'q')  // permet de quitter l'application
            break;
        if (c == 'f') // permet de geler l'image
            freeze = !freeze;
        cv::rectangle(img_input, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
        imshow("input", img_input); // affiche le flux video
    }
    return 0;
}
