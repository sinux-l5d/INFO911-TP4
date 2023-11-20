#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

const int HIST_SIZE = 8;
struct ColorDistribution
{
    float data[HIST_SIZE][HIST_SIZE][HIST_SIZE]; // l'histogramme
    int nb;                                      // le nombre d'échantillons
    bool fin;

    ColorDistribution() { reset(); }
    ColorDistribution &operator=(const ColorDistribution &other) = default;
    // Met à zéro l'histogramme
    void reset()
    {
        nb = 0;
        memset(data, 0, sizeof(data));
        fin = false;
    }
    // Ajoute l'échantillon color à l'histogramme:
    // met +1 dans la bonne case de l'histogramme et augmente le nb d'échantillons
    void add(Vec3b color)
    {
        if (fin) // si on a fini de mettre les échantillons, on ne peut plus en ajouter
            return;
        data[color[0] * HIST_SIZE / 256][color[1] * HIST_SIZE / 256][color[2] * HIST_SIZE / 256]++;
        nb++;
    }
    // Indique qu'on a fini de mettre les échantillons:
    // divise chaque valeur du tableau par le nombre d'échantillons
    // pour que case représente la proportion des picels qui ont cette couleur.
    void finished()
    {
        fin = true;
        for (int i = 0; i < HIST_SIZE; i++)
            for (int j = 0; j < HIST_SIZE; j++)
                for (int k = 0; k < HIST_SIZE; k++)
                    data[i][j][k] /= static_cast<float>(nb);
    }
    // Retourne la distance chi-2 entre cet histogramme et l'histogramme other
    float distance(const ColorDistribution &other) const
    {
        if (!fin || !other.fin)
            return -1; // on ne peut pas calculer la distance si on n'a pas fini de mettre les échantillons
        float dist = 0;
        for (int i = 0; i < HIST_SIZE; i++)
            for (int j = 0; j < HIST_SIZE; j++)
                for (int k = 0; k < HIST_SIZE; k++)
                {
                    if (data[i][j][k] == 0 && other.data[i][j][k] == 0)
                        continue;
                    float denom = data[i][j][k] - other.data[i][j][k];
                    dist += (denom * denom) / (data[i][j][k] + other.data[i][j][k]);
                }
        return dist;
    }
};

ColorDistribution getColorDistribution(Mat input, Point pt1, Point pt2)
{
    ColorDistribution cd;
    for (int y = pt1.y; y < pt2.y; y++)
        for (int x = pt1.x; x < pt2.x; x++)
            cd.add(input.at<Vec3b>(y, x));
    cd.finished();
    return cd;
}

// retourne la plus petite distance entre h et les histogrammes de couleurs de hists.
float minDistance(const ColorDistribution &h,
                  const std::vector<ColorDistribution> &hists)
{
    float min_dist = 1000000;
    for (const ColorDistribution &h2 : hists)
    {
        float dist = h.distance(h2);
        if (dist < min_dist)
            min_dist = dist;
    }
    return min_dist;
}

//  fabrique une nouvelle image, où chaque bloc est coloré selon qu’il est “fond” ou “objet”.
Mat recoObject(Mat input,
               const std::vector<ColorDistribution> &col_hists,
               const std::vector<ColorDistribution> &col_hists_object,
               const std::vector<Vec3b> &colors,
               const int bloc)
{
    Mat img_seg = Mat::zeros(input.size(), CV_8UC3);
    for (int y = 0; y <= input.rows - bloc; y += bloc)
        for (int x = 0; x <= input.cols - bloc; x += bloc)
        {
            Point pt1(x, y);
            Point pt2(x + bloc, y + bloc);
            ColorDistribution cd = getColorDistribution(input, pt1, pt2);
            float dist_background = minDistance(cd, col_hists);
            float dist_object = minDistance(cd, col_hists_object);
            if (dist_background < dist_object)
                rectangle(img_seg, pt1, pt2, colors[0], FILLED);
            else
                rectangle(img_seg, pt1, pt2, colors[1], FILLED);
        }
    return img_seg;
}

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
    std::vector<ColorDistribution> col_hists;        // histogrammes du fond
    std::vector<ColorDistribution> col_hists_object; // histogrammes de l'objet
    const std::vector<Vec3b> colors = {Vec3b(0, 0, 0), Vec3b(0, 0, 255)};

    namedWindow("input", 1);
    imshow("input", img_input);

    bool freeze = false;
    bool reco = false;

    while (true)
    {
        char c = (char)waitKey(20);
        if (pCap != nullptr && !freeze)
            (*pCap) >> img_input; // récupère l'image de la caméra
        if (c == 'q')             // permet de quitter l'application
            break;
        else if (c == 'f') // permet de geler l'image
            freeze = !freeze;
        else if (c == 'v')
        {
            Point gh(0, 0);
            Point gb(width / 2, height);
            Point dh(width / 2, 0);
            Point db(width, height);
            ColorDistribution cd_gh = getColorDistribution(img_input, gh, gb);
            ColorDistribution cd_dh = getColorDistribution(img_input, dh, db);
            float dist = cd_gh.distance(cd_dh);
            cout << "Distance : " << dist << endl;
        }
        else if (c == 'b')
        {
            const int bbloc = 128;
            for (int y = 0; y <= height - bbloc; y += bbloc)
                for (int x = 0; x <= width - bbloc; x += bbloc)
                {
                    Point pt1(x, y);
                    Point pt2(x + bbloc, y + bbloc);
                    col_hists.push_back(getColorDistribution(img_input, pt1, pt2));
                }
        }
        else if (c == 'a')
        {
            col_hists_object.push_back(getColorDistribution(img_input, pt1, pt2));
        }
        else if (c == 'r')
        {
            reco = !reco;
        }

        Mat output = img_input;
        if (reco)
        { // mode reconnaissance
            Mat gray;
            cvtColor(img_input, gray, COLOR_BGR2GRAY);
            Mat reco = recoObject(img_input, col_hists, col_hists_object, colors, 8);
            cvtColor(gray, img_input, COLOR_GRAY2BGR);
            output = 0.5 * reco + 0.5 * img_input; // mélange reco + caméra
        }
        else
            cv::rectangle(output, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
        cv::rectangle(output, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
        imshow("input", output); // affiche le flux video
    }
    return 0;
}
