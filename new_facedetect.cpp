#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// försök få bort dessa
bool showHist = true;
Mat image;

//original (slighty trimmed) detectAndDraw function
void detectAndDraw(Mat &img, CascadeClassifier &cascade, vector<Rect> &faces);

//the modified camshift function
int myCamShift(Rect &face, Mat &frame);

int main(int argc, const char **argv)
{
    VideoCapture capture;
    Mat frame;
    CascadeClassifier cascade;
    string cascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    //add this for speed
    vector<Rect> faces;

    if (!cascade.load(cascadeName))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;

        return -1;
    }


    if (!capture.open(0))
        cout << "Capture from camera didn't work" << endl;

    // if camera successfully opened
    if (capture.isOpened())
    {
        cout << "Video capturing has been started ..." << endl;

        /**** THIS IS THE MAIN FOREVER LOOP ***/
        for (;;)
        {
            //object capture gets frame from camera and stores in frame
            capture >> frame;
            if (frame.empty())
                break;
            //copy frame
            Mat frame1 = frame.clone();

            if (faces.empty())
            {
                //DETECTION FUNCTION
                detectAndDraw(frame1, cascade,faces);
            }
            else
            {
                myCamShift(faces[0], frame1);
            }

            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }


return 0;
}

//this will become myCamshift (rect selection(faces[0]), Mat& image(frame1)
//remove everything that has to do with the capture
int myCamShift(Rect &face, Mat &frame)
{
    int trackObject = -1;

    int hsize = 16;
    float hranges[] = {0, 180};
    const float *phranges = hranges;
    Rect selection = face;
    Rect trackWindow;

    int vmin = 10, vmax = 256, smin = 30;

    namedWindow("Histogram", 0);
    namedWindow("CamShift Demo", 0);
    createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
    createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
    createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);

    Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;

    vmin = 10, vmax = 256, smin = 30;

    frame.copyTo(image);
    cvtColor(image, hsv, COLOR_BGR2HSV);

    int _vmin = vmin, _vmax = vmax;

    inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
            Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    if (trackObject < 0)
    {

        // Object has been selected by user, set up CAMShift search properties once
        // roi is our field of interest
        Mat roi(hue, selection), maskroi(mask, selection);
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);

        trackWindow = selection;
        trackObject = 1; // Don't set up again, unless user selects new ROI

        // drawing the histogram: start
        histimg = Scalar::all(0);
        int binW = histimg.cols / hsize;
        Mat buf(1, hsize, CV_8UC3);
        for (int i = 0; i < hsize; i++)
            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);
        cvtColor(buf, buf, COLOR_HSV2BGR);

        for (int i = 0; i < hsize; i++)
        {
            int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
            rectangle(histimg, Point(i * binW, histimg.rows),
                      Point((i + 1) * binW, histimg.rows - val),
                      Scalar(buf.at<Vec3b>(i)), -1, 8);
        }
        // drawing the histogram: end
    }

    // Perform CAMShift
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    // FACE TRACKING
    RotatedRect rFaceBox = CamShift(backproj, trackWindow,
                                    TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));


    // trackbox is not (x,y). check out the function to change RotatedRect -> Rect
    Rect tempBox = rFaceBox.boundingRect();
    Rect faceBox = Rect(tempBox.x - 30, tempBox.y -30, tempBox.width + 30, tempBox.height + 30);


    //change trackwindow = REct(0,0,omage, cols, row)
    Rect newTrackWindow = Rect(0, 0, image.cols, image.rows);

    //get the intersection to avoid out of bounds errors
    faceBox = faceBox & newTrackWindow;

    // backproj: make pixels inside trackbox area zero
    backproj(faceBox).setTo(0);



    //do the above fucniton again RotatetedRect handBox = Camshift(Backproj[changed], newTrackwindow, )
    RotatedRect rHandBox = CamShift(backproj, newTrackWindow,
                                    TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    Rect handBox = rHandBox.boundingRect();
    //then rotate the new rotatedrect again to -> Rect
    if (trackWindow.area() <= 1)
    {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                           trackWindow.x + r, trackWindow.y + r) &
                      Rect(0, 0, cols, rows);
    }

    ellipse(image, rHandBox, Scalar(0, 0, 255), 3, LINE_AA);

    //rectangle(image, handBox, Scalar(255,0,0)); //for testing

    Rect bigHandBox = Rect(handBox.x + 20, handBox.y +20, handBox.width, handBox.height);
    bigHandBox = bigHandBox & newTrackWindow;
    handBox = handBox & newTrackWindow;

    imshow("Backproj", backproj);
    imshow("CamShift", image);
    imshow("Histogram", histimg);

    char c = (char)waitKey(10);

    switch (c)
    {
    case 'c':
        trackObject = 0;
        histimg = Scalar::all(0);
        break;
    case 'h':
        showHist = !showHist;
        if (!showHist)
            destroyWindow("Histogram");
        else
            namedWindow("Histogram", 1);
        break;
    case 'p':
        paused = !paused;
        break;
    case 's':
        imshow("Handsign capture", image(bigHandBox));
        break;
    case 'x':{
        // 32 corresponds to space bar
        Mat hand = backproj(handBox);
        resize(hand, hand, Size(16,16), 0, 0, INTER_LINEAR);
        imshow("new hand", hand);
        Mat img2 = hand.reshape(0,1);
        std::ofstream os ("letter.txt", ios::out | ios::app);

        //this is where the MLP model and prediction
        os << "C,";
        os << format(img2, Formatter::FMT_CSV ) << endl;
        os.close();
        imwrite("Handsign capture", image(handBox));
        break;}
    default:
        ;
    }
}

void detectAndDraw(Mat &img, CascadeClassifier &cascade, vector<Rect> &faces) //add faces for better speed
{
    // in xml we have a list of patterns, thresholds and their confidence values
    double t = 0;
    vector<Rect> faces2;
    const static Scalar colors[] =
        {//BGR
         Scalar(255, 0, 0),
         Scalar(255, 128, 0),
         Scalar(255, 255, 0),
         Scalar(0, 255, 0),
         Scalar(0, 128, 255),
         Scalar(0, 255, 255),
         Scalar(0, 0, 255),
         Scalar(255, 0, 255)};
    Mat gray, smallImg;
    //convert color
    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1;
    // make image smaller with a linear interpolation
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    //
    equalizeHist(smallImg, smallImg);

    //time it takes to face detect
    t = (double)getTickCount();
    // ada-boost
    cascade.detectMultiScale(smallImg, faces,
                             1.1, 2, 0
                                         //|CASCADE_FIND_BIGGEST_OBJECT
                                         //|CASCADE_DO_ROUGH_SEARCH
                                         | CASCADE_SCALE_IMAGE,
                             Size(30, 30));

    t = (double)getTickCount() - t;
    printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
    // go through all face
    for (size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
        Mat smallImgROI;
        Point center;
        Scalar color = colors[i % 8];
        int radius;

        double aspect_ratio = (double)r.width / r.height;
        if (0.75 < aspect_ratio && aspect_ratio < 1.3)
        {
            center.x = cvRound((r.x + r.width * 0.5));
            center.y = cvRound((r.y + r.height * 0.5));
            radius = cvRound((r.width + r.height) * 0.25);
            circle(img, center, radius, color, 3, 8, 0);
        }
        else
            rectangle(img, cvPoint(cvRound(r.x), cvRound(r.y)),
                      cvPoint(cvRound(r.x + r.width - 1), cvRound(r.y + r.height - 1)),
                      color, 3, 8, 0);
    }
    imwrite("result.jpg", img);
}
