#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace std;
using namespace cv;


bool showHist = true;
Mat image;


int myCamShift( Rect face, Mat& frame );

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, bool tryflip, vector<Rect>& faces );

static void help()
{
    cout << "\nThis program demonstrates the cascade recognizer. Now you can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}


string cascadeName = "/home/user/src/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml";


int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade;
    double scale;
    

    
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    // cascadeName = parser.get<string>("cascade");
    // nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    /*
 if ( !nestedCascade.load( nestedCascadeName ) )
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    */
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }
    // if input is empty then it will launch the camera
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        if(!capture.open(camera))
            cout << "Capture from camera #" <<  camera << " didn't work" << endl;
    }
    // if no camera, use image/imput
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
        {
            if(!capture.open( inputName ))
                cout << "Could not read " << inputName << endl;
        }
    }
    //else load a default image
    else
    {
        image = imread( "../data/lena.jpg", 1 );
        if(image.empty()) cout << "Couldn't read ../data/lena.jpg" << endl;
    }
    

    //add this for speed
	vector<Rect> faces;
	
    // if camera successfully opened
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;

	    
	
	    //forever
        for(;;)
        {
	        //object capture gets frame from camera and stores in frame
            capture >> frame;
            if( frame.empty() )
                break;
	        //copy frame
            Mat frame1 = frame.clone();
	       
           if(faces.empty()){
                //DETECTION FUNCTION
                detectAndDraw( frame1, cascade,  scale, tryflip, faces );
           }
           else{
               myCamShift(faces[0], frame1);
           }
           

            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else
    {
        cout << "Detecting face(s) in " << inputName << endl;
        if( !image.empty() )
        {
	  detectAndDraw( image, cascade, scale, tryflip, faces );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;
                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
		      detectAndDraw( image, cascade, scale, tryflip, faces );
                        char c = (char)waitKey(0);
                        if( c == 27 || c == 'q' || c == 'Q' )
                            break;
                    }
                    else
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }
    
    return 0;
}

//this will become myCamshift (rect selection(faces[0]), Mat& image(frame1)
//remove everything that has to do with the capture
int myCamShift( Rect face, Mat& frame )
{
    int trackObject = -1;
    
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    Rect selection = face;
    Rect trackWindow;

    int vmin = 10, vmax = 256, smin = 30;
    
    namedWindow( "Histogram", 0 );
    namedWindow( "CamShift Demo", 0 );    
    createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
    createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
    createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );
    

    Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;
    
    vmin = 10, vmax = 256, smin = 30;
    
    frame.copyTo(image);
    cvtColor(image, hsv, COLOR_BGR2HSV);

    int _vmin = vmin, _vmax = vmax;

    inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
            Scalar(180, 256, MAX(_vmin, _vmax)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    if( trackObject < 0 )
    {   

        // Object has been selected by user, set up CAMShift search properties once
        Mat roi(hue, selection), maskroi(mask, selection);
        calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);

        trackWindow = selection;
        trackObject = 1; // Don't set up again, unless user selects new ROI

        histimg = Scalar::all(0);
        int binW = histimg.cols / hsize;
        Mat buf(1, hsize, CV_8UC3);
        for( int i = 0; i < hsize; i++ )
            buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
        cvtColor(buf, buf, COLOR_HSV2BGR);

        for( int i = 0; i < hsize; i++ )
        {
            int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
            rectangle( histimg, Point(i*binW,histimg.rows),
                        Point((i+1)*binW,histimg.rows - val),
                        Scalar(buf.at<Vec3b>(i)), -1, 8 );
            }
        
    }

    // Perform CAMShift
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    RotatedRect trackBox = CamShift(backproj, trackWindow,
                        TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    if( trackWindow.area() <= 1 )
    {
        int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
        trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                            trackWindow.x + r, trackWindow.y + r) &
                        Rect(0, 0, cols, rows);
    }

    
    ellipse( image, trackBox, Scalar(0,0,255), 3, LINE_AA );

    imshow( "CamShift", image );
    imshow( "Histogram", histimg );
    
    
    



    char c = (char)waitKey(10);

    switch(c)
    {
    case 'c':
        trackObject = 0;
        histimg = Scalar::all(0);
        break;
    case 'h':
        showHist = !showHist;
        if( !showHist )
            destroyWindow( "Histogram" );
        else
            namedWindow( "Histogram", 1 );
        break;
    case 'p':
        paused = !paused;
        break;
    default:
        ;
    }
    
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade, double scale, bool tryflip, vector<Rect>& faces )//add faces for better speed
{
  // in xml we have a list of patterns, thresholds and their confidence values
    double t = 0;
    vector<Rect> faces2;
    const static Scalar colors[] =
      {//BGR
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;
    //convert color
    cvtColor( img, gray, COLOR_BGR2GRAY );
    double fx = 1 / scale;
    // make image smaller with a linear interpolation
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    // 
    equalizeHist( smallImg, smallImg );

    //time it takes to face detect
    t = (double)getTickCount();
    // ada-boost
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    //if there is only light on one side, try fliping image to detect more faces
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    // go through all face
    for ( size_t i = 0; i < faces.size(); i++ )
      {
        Rect r = faces[i];
        Mat smallImgROI;
        Point center;
        Scalar color = colors[i%8];
        int radius;

	
        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);

    }
    imwrite( "result.jpg", img );
}

