#include <string>
#include <thread>
#include <chrono>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace cv;
using namespace std;

const int key_escape = 1048603;
const int key_enter = 1048586;
const int key_space = 1048608;

const int mW = 640;
const int mH = 480;
const int mF = 30;

Size IMG_SIZE = Size(mW, mH);

boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (
  	new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  //viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}

void getPointCloud(
	 Mat& img_rgb,
	 Mat& img_disparity,
	const Mat& Q,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr)
{
    //Get the interesting parameters from Q
	double Q03, Q13, Q23, Q32, Q33;
	Q03 = Q.at<double>(0,3);
	Q13 = Q.at<double>(1,3);
	Q23 = Q.at<double>(2,3);
	Q32 = Q.at<double>(3,2);
	Q33 = Q.at<double>(3,3);

  
  point_cloud_ptr -> clear();
  
  double px, py, pz;
  uchar pr, pg, pb;
  
  for (int i = 0; i < img_rgb.rows; i++)
  {
    uchar* rgb_ptr = img_rgb.ptr<uchar>(i);
    uchar* disp_ptr = img_disparity.ptr<uchar>(i);

    for (int j = 0; j < img_rgb.cols; j++)
    {
      //Get 3D coordinates
      uchar d = disp_ptr[j];
      if ( d == 0 ) continue; //Discard bad pixels
      double pw = -1.0 * static_cast<double>(d) * Q32 + Q33; 
      px = static_cast<double>(j) + Q03;
      py = static_cast<double>(i) + Q13;
      pz = Q23;
      
      px = px/pw;
      py = py/pw;
      pz = pz/pw;
      
      //Get RGB info
      pb = rgb_ptr[3*j];
      pg = rgb_ptr[3*j+1];
      pr = rgb_ptr[3*j+2];
      
      //Insert info into point cloud structure
      pcl::PointXYZRGB point;
      point.x = px;
      point.y = py;
      point.z = pz;
      
      //std::cout << "XYZ: " << px << " " << py << " " << pz << std::endl;
      
      uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
              static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
      point.rgb = *reinterpret_cast<float*>(&rgb);
      point_cloud_ptr->points.push_back (point);
    }
  }
  point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
  
  std::cout << "Cloude size: " << point_cloud_ptr->width << std::endl;
  point_cloud_ptr->height = 1;
}

int main(int argc, char** argv)
{
	Mat img1, img2;
	Mat gray1(IMG_SIZE, CV_8UC1);
    Mat gray2(IMG_SIZE, CV_8UC1);
    
    VideoCapture cap1;
    VideoCapture cap2;
    
	cap1.set(CV_CAP_PROP_FRAME_WIDTH, mW);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, mH);
	cap1.set(CV_CAP_PROP_FPS, mF);

	cap2.set(CV_CAP_PROP_FPS, mF);
	cap2.set(CV_CAP_PROP_FRAME_WIDTH, mW);
	cap2.set(CV_CAP_PROP_FRAME_HEIGHT, mH);
    cap1.open(0);
	cap2.open(1);
    

	namedWindow("image1", 1);
	namedWindow("image2", 1);
	namedWindow("depth", 1);
	moveWindow("image1", 0, 0);
	moveWindow("image2", 800, 0);
	moveWindow("depth", 0, 300);

	Mat M1, D1, M2, D2, R, T, R1, R2, P1, P2, Q;
	Rect roi1, roi2;

	FileStorage fs(argv[1], FileStorage::READ);
	fs["M1"] >> M1;
	fs["M2"] >> M2;
	fs["D1"] >> D1;
	fs["D2"] >> D2;
	fs["R"] >> R;
	fs["T"] >> T;
	fs["R1"] >> R1;
	fs["R2"] >> R2;
	fs["P1"] >> P1;
	fs["P2"] >> P2;
	fs["Q"] >> Q;

	
	stereoRectify(M1, D1, M2, D2, IMG_SIZE, R, T, R1, R2, P1, P2, Q, 
				  0, -1, IMG_SIZE, &roi1, &roi2 );

	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, IMG_SIZE, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, IMG_SIZE, CV_16SC2, map21, map22);

	int PreFilterCap = 31;
	//int PreFilterSize = 5;
	int SADWindowSize = 9;
	int MinDisparity = 100;
	int NumberOfDisparities = 128;
	int TextureThreshold = 15;
	int UniqnessRatio = 15;
	int SpeckleWindowSize = 100;
	int SpeckleRange = 32;
	int MaxDiff = 1;

	StereoBM bm;
	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = PreFilterCap;
    bm.state->SADWindowSize =  SADWindowSize;
    bm.state->minDisparity = MinDisparity-100;
    bm.state->numberOfDisparities = NumberOfDisparities;
    bm.state->textureThreshold = TextureThreshold;
    bm.state->uniquenessRatio = UniqnessRatio;
    bm.state->speckleWindowSize = SpeckleWindowSize;
    bm.state->speckleRange = SpeckleRange;
    bm.state->disp12MaxDiff = MaxDiff;
    
    namedWindow("Track Bar Window", CV_WINDOW_NORMAL);
    
    cvCreateTrackbar("Pre Filter Cap", "Track Bar Window", &PreFilterCap, 61);
    cvCreateTrackbar("Number of Disparities", "Track Bar Window", &NumberOfDisparities, mW);
    cvCreateTrackbar("SAD", "Track Bar Window", &SADWindowSize, 100);
    cvCreateTrackbar("Minimum Disparity", "Track Bar Window", &MinDisparity, 200);
    cvCreateTrackbar("Texture Threshold", "Track Bar Window", &TextureThreshold, 100);
    cvCreateTrackbar("Uniqueness Ratio", "Track Bar Window", &UniqnessRatio, 100);
    cvCreateTrackbar("Speckle Window Size", "Track Bar Window", &SpeckleWindowSize, 200);
    cvCreateTrackbar("Speckle Range", "Track Bar Window", &SpeckleRange, 500);
    cvCreateTrackbar("MaxDiff", "Track Bar Window", &MaxDiff, mW);


	int k = 0;
	Mat img1r, img2r;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	//Create visualizer
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  viewer = createVisualizer( point_cloud_ptr );
	while(!viewer->wasStopped())
	{
	
	// Settings   
        if (PreFilterCap % 2 == 0)
           PreFilterCap = PreFilterCap + 1;

        if (SADWindowSize % 2 == 0)
            SADWindowSize = SADWindowSize + 1;

        if (SADWindowSize < 5)
            SADWindowSize = 5;

        if (NumberOfDisparities % 16 != 0)
            NumberOfDisparities = NumberOfDisparities + (16 - NumberOfDisparities % 16);
            
       	bm.state->preFilterCap = PreFilterCap;
		bm.state->SADWindowSize =  SADWindowSize;
		bm.state->minDisparity = MinDisparity-100;
		bm.state->numberOfDisparities = NumberOfDisparities;
		bm.state->textureThreshold = TextureThreshold*0.01;
		bm.state->uniquenessRatio = 0.01*UniqnessRatio;
		bm.state->speckleWindowSize = SpeckleWindowSize;
		bm.state->speckleRange = SpeckleRange;
		bm.state->disp12MaxDiff = MaxDiff;
                  
     // Capture images
		cap1 >> img1;
		cap2 >> img2;
		
		/*std::thread t1([&]{fastNlMeansDenoisingColored(img1, img1);});
		fastNlMeansDenoisingColored(img2, img2);
		t1.join();*/
	
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;

		cvtColor(img1, gray1, CV_BGR2GRAY);
		cvtColor(img2, gray2, CV_BGR2GRAY);

	 	//imshow("image1", gray1);
		//imshow("image2", gray2);

		Mat map;
		bm( gray1, gray2, map);
	
		// normalize	
		Mat disp8;
		map.convertTo(disp8, CV_8U, 255/(NumberOfDisparities*16.));

		//imshow("depth", disp8);

		k = waitKey(5);
		if (k == key_escape)
			break;
			
		
			
		// Visualise in PCL
		getPointCloud(img1, disp8, Q, point_cloud_ptr );
		viewer->updatePointCloud(point_cloud_ptr, "reconstruction");
		viewer->spinOnce(100);
    	std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
		
	destroyAllWindows();
	cap1.release();
	cap2.release();
}


