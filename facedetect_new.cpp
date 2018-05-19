#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

//global variables is bad practice..
//but I think it's even worse practise to declare them every loop
int trackObject = -1;
bool backprojMode = false;
int hsize = 16;
float hranges[] = {0,180};
const float* phranges = hranges;
Mat frame, hsv, hue, mask, hist = Mat::zeros(200, 320, CV_8UC3);
int _vmin = 10, _vmax = 256, smin = 30;
int letterCount = 30;

//original (slighty trimmed) detectAndDraw function
void detectAndDraw(Mat &img, CascadeClassifier &cascade, Rect &face);

// to exctract part of the camshift code
void camshiftPrep();

// main camshift function
void myCamshift(Rect &face, Mat &frame, Rect &trackWindow);

//code to load classifier
template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";
    return model;
}

//the main function
int main(int argc, const char **argv)
{
    VideoCapture capture;
    Mat frame;
    CascadeClassifier cascade;
    string cascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    Rect face; //add this for speed
    Rect trackWindow; //face

    if (!cascade.load(cascadeName))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if (!capture.open(0))
        cout << "Capture from camera didn't work" << endl;

    if (capture.isOpened()) // if camera successfully opened
    {

        namedWindow( "CamShift Demo", 0 );
        cout << "Video capturing has been started ..." << endl;
        for (;;) //first "forever" loop
        {
            capture >> frame; //object capture gets frame from camera and stores in frame
            if (frame.empty())
                break;

            if (face.empty()) //if a face hasn't been detected
            {   //DETECTION FUNCTION. 
                detectAndDraw(frame, cascade, face);
            }
            else
            {   //Camshift and hand detection
                myCamshift(face, frame, trackWindow);
            }

            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }
    return 0;
}


void myCamshift(Rect &selection, Mat &image, Rect &trackWindow)
{
    Mat backproj;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    camshiftPrep(); //to portion out some of the code

    if( trackObject < 0 ) //this happens ONCE
    { // Object has been selected by detectAndDraw, set up CAMShift search properties once
        Mat roi(hue, selection), maskroi(mask, selection);
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);

        trackWindow = selection;
        trackObject = 1; // Don't set up again
        cout << "Creating Camshift search profile" << endl;
    }
    // Perform CAMShift
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;

    // FACE TRACKING
    RotatedRect rFaceBox = CamShift(backproj, trackWindow,
                                    TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

    // backproj: make pixels inside facebox area zero
    backproj(trackWindow).setTo(0);

    int cols = backproj.cols, rows = backproj.rows;
    //new trackwindow that is the whole frame
    Rect newTrackWindow = Rect(0, 0, cols, rows);
    //make the trackwindow a little bit bigger, for the next iteration
    if( trackWindow.area() <= 1 )
    {
        int r = (MIN(cols, rows) + 5)/6;
        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                           trackWindow.x + r, trackWindow.y + r) &
                      newTrackWindow;
    }  

    //Do CAMSHIFT again, to detect hand when the face is blocked out
    RotatedRect rHandBox = CamShift(backproj, newTrackWindow,
                                    TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    Rect handBox = rHandBox.boundingRect(); //get it in the right format
    handBox = handBox & newTrackWindow; //to avoid out of bounds errors

    rectangle(image, handBox, Scalar(0, 255, 0), 3, LINE_AA); //draw rectangle

    imshow( "detection", image );
    cvtColor( backproj, image, COLOR_GRAY2BGR );
    imshow( "backproj", image );

    char c = (char)waitKey(10);
    switch(c)
    {
      case 'x':{ //capture hands for training

        Mat hand = backproj(handBox);
        resize(hand, hand, Size(16,16), 0, 0, INTER_LINEAR);
        Mat img2 = hand.reshape(0,1);
        std::ofstream os ("letter_new.txt", ios::out | ios::app);
        os << "Y,";
        os << format(img2, Formatter::FMT_CSV ) << endl;
        os.close();
        //imwrite("Letter_Y_" + std::to_string(letterCount) + ".jpg", image(handBox));
        letterCount--;
        break;}
      case 'p':{ //predict hand sign using trained model
          Mat hand = backproj(handBox);
          resize(hand, hand, Size(16,16), 0, 0, INTER_LINEAR);
          Mat img2 = hand.reshape(0,1);
          Ptr<ANN_MLP> model = load_classifier<ANN_MLP>("backup_network");
          img2.convertTo(img2, CV_32F); //added this line to make it work
          float r = model->predict(img2);
          r = r + (int)('A');
          cout << (char)r << endl;
          break;}
      case 'r': //redo face detect
          selection = {};
          trackObject = -1;
          break;
      default:
          ;
    }
}

void camshiftPrep()
{
    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
            Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);
}

void detectAndDraw(Mat &img, CascadeClassifier &cascade, Rect &face) //add faces for better speed
{
    // in xml we have a list of patterns, thresholds and their confidence values
    double t = 0;
    vector<Rect> faces;
    Mat gray, smallImg;
    cvtColor(img, gray, COLOR_BGR2GRAY); //convert color
    double fx = 1;
    // make image smaller with a linear interpolation
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(smallImg, smallImg);
    //time it takes to face detect
    t = (double)getTickCount();
    // ada-boost
    cascade.detectMultiScale(smallImg, faces,
                             1.1, 2, 0 | CASCADE_SCALE_IMAGE,
                             Size(30, 30));
    t = (double)getTickCount() - t;
    printf("detection time = %g ms\n", t * 1000 / getTickFrequency());
    face = faces[0];
    // add dimensions to make the selection area bigger
    //face = Rect(face.x - 20, face.y -20, face.width +40, face.height +40);
    //draw a rectangle instead of circle
    rectangle(img, face, Scalar(255,0,0));
    imwrite("result_from_detectAndDraw.jpg", img);

}
