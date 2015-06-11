//Test program PCL
//Agustin Ortega 
//OCT 2014
#include <string>
#include <iostream>
#include <vector>

#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


#include <pcl/point_types.h>


//visualization 
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>

//segmentation
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>  

#include "pcl/io/io.h"
#include "pcl/io/pcd_io.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/passthrough.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/features/normal_3d.h"

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include "pcl/kdtree/io.h"
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//convexhull
#include "pcl/surface/convex_hull.h"
#include "pcl/segmentation/extract_polygonal_prism_data.h"
#include "pcl/segmentation/extract_clusters.h"

#include <pcl/pcl_base.h>



#include <pcl/surface/mls.h>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>

#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/range_image/range_image.h>


#include <Eigen/Eigen>
#include <Eigen/SVD> 

#include <boost/thread/thread.hpp>


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <math.h>
#include <string.h>



//#include "Lu.h"
#include "Utilities.h"
//#include "FindChessBoard3D.h"

//#include "FindChessBoardImage.h"


using namespace cv;

using namespace Eigen;

using namespace std;
using namespace pcl;

////////////////////////////////////////////////
// global variables 
/////////////////////////////////////////////////
pcl::PointCloud<PointXYZ>::Ptr cloud_p(new PointCloud<pcl::PointXYZ> ());
pcl::PointCloud<PointXYZ>::Ptr cloud_p2(new PointCloud<pcl::PointXYZ> ());
int idx=1;
Mat image_aux;

void mouseHandler(int event, int x, int y, int flags, void *param)
{
  
  /* left button down */
  if (event==CV_EVENT_LBUTTONDOWN){
    fprintf(stdout, "Left button down (%d, %d).\n", x, y);
    
    pcl::PointXYZ  PointAux;
    
    PointAux.x=x;
    PointAux.y=y;
    PointAux.z=1.0f;
    cloud_p2->points.push_back (PointAux);
    
    //           frame_left2 = cvCloneImage(frame_left);
    rectangle(image_aux,
		cvPoint(x - 5, y - 5),
		cvPoint(x + 5, y + 5),
		cvScalar(0, 0, 255, 0), 2, 8, 0);
		imshow("Image", image_aux);
		
  }else
    return;
}

/////////////////////////////////////////////////////////////////////////
void PickingEventOccurred(const visualization::PointPickingEvent& event, void* viewer_void){
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<
  visualization::PCLVisualizer> *>(viewer_void);
  
  
  
  idx = event.getPointIndex ();
  if (idx == -1)
    return;
  
  vector<int> indices (1);
  vector<float> distances (1);
  
  
  // Use mutices to make sure we get the right cloud
  //boost::mutex::scoped_lock lock1 (cloud_mutex_);
  
  pcl::PointXYZ   PointAux;
  
  event.getPoint ( PointAux.x,  PointAux.y,  PointAux.z);
  
  char str[512];
  sprintf(str, "sphere_%d",idx);
  
  std::cout << " (" << PointAux.x << ", " << PointAux.y <<", " << PointAux.z<< "......)"
  << str<<std::endl;
  
  cloud_p->points.push_back (PointAux);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_p, 255, 0, 0);
  //viewer_void->addSphere(PointAux, str);
  viewer->addPointCloud<pcl::PointXYZ>(cloud_p,single_color, str,0);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,10,str);
  
}



void help(){
     std::cout << "Wrong argument values\n";
    
    cout << std::endl;
    cout << "--pcd.                           Laser topic\n"; //this part is gonna be changed by the topics
    cout << "--img.                         image topic\n"; //this part is gonna be changed by the topics
    cout << "--method.                         [LU|SOLVEPNP] Method for estimating the pose\n"; //this part is gonna be changed by the topics
  ;

    std::cout << std::endl;
    
    cout<<"try ./calib_laser --pcd prenav.pcd --img IMG_3.ppm --method LU"<<endl;

  
}

int ReadArgumens( int argc, char** argv, 
		 const pcl::PointCloud<PointXYZ>::Ptr&  cloud_main,
		 Mat& image, 
		   string& method,
		  string& selection,
		   int& h,int& w) {
  
  
  pcl::PCDReader reader;  
  
  string pcd,image_arg,h_arg,w_arg;
  //read pcl
  if (pcl::console::parse_argument (argc, argv, "--pcd", pcd) != -1){
    
    if( reader.read<pcl::PointXYZ> (pcd, *cloud_main)==-1){
      PCL_ERROR("File was not found ");
      
      return (-1);
    }
    
      PCL_INFO("Reading PCD %s with %d points \n\n", pcd.c_str(), cloud_main->points.size());
    
    //return(1);
    
  }else
    return(-1);
  //read image
  if(pcl::console::parse_argument (argc, argv, "--img", image_arg) != -1){
    
    image =imread(image_arg.c_str(),1);
    
     PCL_INFO("Image (%d,%d)\n\n", image.size().width,image.size().height);
    
    if(image.empty()){
      PCL_ERROR("Image File was not found ");
      return(-1);
    }
    
    //return(1);
    
  }else 
    return(-1);
  

  selection="MANUAL";
  
  
  if(pcl::console::parse_argument (argc, argv, "--method", method) != -1){
    
    cout<<method<<endl;
    
    if (method.compare("LU")!=0 && method.compare("SOLVEPNP")!=0 && !method.compare("POSIT")!=0 )
    {
      
      PCL_INFO("This method  does not exist");
      return (-1);
    }
    
  //  return(1);
    
  }
  //else return (1); //all ok
    
   
    return (1);
     
}



int main (int argc, char** argv){
  
  PointCloud<PointXYZ>::Ptr  cloud(new pcl::PointCloud<pcl::PointXYZ> ());//principal cloud
  
  ///////////////////////////////////////////////////////////////////////////////7
  ///////////clouds for autmatic segmentation
  ///////////////////////////////////////////////////////////////////////////////7
   pcl::PointCloud<PointXYZ>::Ptr cloud_trans (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ> ());
  
  
  ///////////////////////////////////////////////////////
  // variables for segmenting data
  ///////////////////////////////////////////////////////
  
   pcl::PointCloud<PointXYZ>::Ptr cloud_nonPlanes (new PointCloud<PointXYZ> ());
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Planes(new PointCloud<PointXYZ> ());
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new PointCloud<pcl::PointXYZ>);
   
   
   ///////////////////////////////////////////////////////
  // Clustering data
  ///////////////////////////////////////////////////////
   
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB_cluster (new PointCloud<PointXYZRGB>);
   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_chessboard (new PointCloud<PointXYZ>);
   
   
   ///////////////////////////////////////////////////////////7
   // plane extraction
   /////////////////////////////////////////////////////////////////
     PointCloud<PointXYZ>::Ptr cloud_plane_cluster (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_nonplane_cluster (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_plane_projected (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_chessboard_square(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_hull (new PointCloud<PointXYZ>);
   
  
  /////////////////////////////////////////////////////////7
  // points for pose estimation
  /////////////////////////////////////////////////////////7  
  MatrixXf x3d;//3d points
  MatrixXf x2d;//2d points
  MatrixXf x2d_2;//2d points
  
  vector <Point3f> points3d;
  vector <Point2f> imagePoints;
  MatrixXf points_projected;
  MatrixXf R;// rotation
  Vector3f t;// translation
  MatrixXf xcam_; // points of the pose
  Vector3f xw;// projected data
  PointCloud<PointXYZRGB>::Ptr cloudRGB (new PointCloud<PointXYZRGB>());
 //cloud RGB image and laser
  PointCloud<PointXYZ>::Ptr cloud_laser_cord (new PointCloud<PointXYZ>()); //points selected in the cloud
  Mat image;
  
  string selection;
  string method;
  int h,w;
  Matrix3f K;// intrisic parameter still to see how to read of a topic
    Size boardSize;
  
  ///////////////////////////////////////7
  // object automatic find chessboard
  ////////////////////////////////////////////
  

  
  
     
  
  //////////////////////////////////////////////////////////////////////////////
  //read arguments this sections can be modified by topic read
  //////////////////////////////////////////////////////////////////////////////
  if (ReadArgumens( argc, argv, cloud,image,method,selection, h, w)!=-1){
     PCL_INFO("Argumenst readed\n");
     // only debug erase after
      PCL_INFO("camera point clouds %d \n",cloud->points.size());
      PCL_INFO("image size (%d, %d)\n",image.size().width,image.size().height);
      PCL_INFO("h %d \n",h);
      PCL_INFO("w %d \n",w);
      //PCL_INFO("Method %s \n",method);
      cout <<"Method " <<method <<endl;
      cout <<"selection " <<selection <<endl;
      //PCL_INFO("Selection %s \n",selection);
      image_aux=image.clone();
       boardSize.height=h;
       boardSize.width=w;
      // Noise filtering 
      StatisticalOutlierRemoval<PointXYZ> sor;
      sor.setInputCloud (cloud);
      sor.setMeanK (60);
      sor.setStddevMulThresh (1.0);
      sor.filter (*cloud);
    
    }else{
      help();
     return 0; 
    }
  
    
 /////////////////////////////////////////////////////////////////////////////   
 // here we have both options  Manual and automatic
 /////////////////////////////////////////////////////////////////////////////
 if (  selection.compare("MANUAL")==0){

   //////////////////////////////////////////////////
   // selection of 3d points
   //////////////////////////////////////////////////
   
   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
   
   viewer->setBackgroundColor (0, 0, 0);
   viewer->addCoordinateSystem (1.0);
   
   // viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);
  viewer->registerPointPickingCallback (PickingEventOccurred , (void*)&viewer);
   
   viewer->addPointCloud<pcl::PointXYZ>(cloud,"sample cloud",0);
   
   while (!viewer->wasStopped ()){
     viewer->spinOnce (100);
   }
   
   //////////////////////////////////////////////////
   // selection of image points
   //////////////////////////////////////////////////
   namedWindow( "Image", CV_WINDOW_NORMAL);// Create a window for display. //NORMAL small AUTOSIZE big
   setMouseCallback( "Image",  mouseHandler, 0);//mouse event
   imshow( "Image", image_aux);                   // Show our image inside it.
   
   waitKey(0);
   
   //////////////////////////////////////////////
   // save data in matrices
   //////////////////////////////////////////////
    if (Get3d_2dPoints( cloud_p, cloud_p2, x3d, x2d, points3d,imagePoints)==-1){//if you select bad number of points
      std::cerr<<"the number of points in the image have to be the same than in the laser"<<std::endl;
    }
   
 }
  
  /////////////////////////////////////////////////////////////////////////////
  //pose estimation we have to know internal camera parmeters
  /////////////////////////////////////////////////////////////////////////////
   //the estimation it depends of the method
    // K<<  1616.77 ,0 ,615.92,
      //   0 ,1613.90 ,428.87,
        //0 , 0  ,1.00;

    K<<   2797.41 ,0 ,1643.61,
        0 ,2799.00 ,1212.57,
        0 , 0  ,1.00;
   if (method.compare("LU")==0){ // apply lu method
    cout<<"x2d\n " <<x2d<<endl;
       cout<<"x3d\n"<< x3d<<endl;

  
     
	       
	 
	    Lu_method(x3d, x2d, R,t, xcam_, K,
	       cloudRGB,//projected data
	       cloud_laser_cord,//laser coordinates
	       points_projected, image, cloud ) ;
	       
	       
     std::cout << "rot.\n"<<R<<std::endl;
     
     std::cout << "t.\n"<<t<<std::endl;
	
   }
   else if (method.compare("SOLVEPNP")==0){
     
     cout<<" option selected Solvepnp"<<endl;
     
     SolvePNP_method(points3d,imagePoints, R,t, xcam_, K,
	       cloudRGB,//projected data
	       cloud_laser_cord,//laser coordinates
	       points_projected, image, cloud ) ;
     std::cout << "ROTATION.\n"<<R<<std::endl;
     
     std::cout << "Translation.\n"<<t<<std::endl;
     
   }
   
   
   ////////////////////////////////////////////////////////////////////////////
   //visualize data if you want
   ///////////////////////////////////////////////////////////////////////////
   pcl::visualization::PCLVisualizer viewer2("3D Viewer results");
   viewer2.setBackgroundColor (0, 0, 0);
   pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloudRGB); 
   
   viewer2.addPointCloud<pcl::PointXYZRGB>(cloudRGB,rgb, "sample cloud RGB",0);
   
   PointCloud<PointXYZ> PointAuxRGB2;
   PointAuxRGB2.points.resize(1);
   PointAuxRGB2.points[0].x=t(0);
   PointAuxRGB2.points[0].y=t(1);
   PointAuxRGB2.points[0].z=t(2);
   
   cloud_laser_cord->points.push_back(PointAuxRGB2.points[0]);
   /////////////////////////////////////////////////////
   ///visualize camera coordinates systems and laser
   //viewer2.addPointCloud<pcl::PointXYZ>(cloud_laser_cord, "sample cloud laser cordinates",0);
   //viewer2.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,30,"sample cloud laser cordinates");
   viewer2.addCoordinateSystem(1.0);//laser
   viewer2.addText3D ("laser", PointXYZ(0,0,0),0.1);

  Affine3f trans;
    
  Matrix3f R_aux=R.inverse();
Vector3f t_w=-R_aux*t;
 Vector3f euler = R_aux.eulerAngles(2, 1, 0);
/*yaw = euler(0,0);
pitch = euler(1,0);
roll = euler(2,0)*/
PCL_INFO("EURLE %f %f %f",euler[0],euler[1],euler[2]);
trans = pcl::getTransformation(t_w(0),t_w(1),t_w(2),euler[0],euler[1],euler[2]);
   viewer2.addCoordinateSystem(1.0,trans);//camera
   viewer2.addText3D ("Camera", PointXYZ(t_w(0),t_w(1),t_w(2)),0.1);

     
   //viewer2.addCoordinateSystem();
   
   while (!viewer2.wasStopped ()){
     viewer2.spinOnce (100);
   }
   
   int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
     double fontScale = 2;
     int thickness = 3; 
   for (int i=0;i<x2d.rows();i++){
     
     circle ( image, cvPoint(x2d(i,0) ,x2d(i,1)), 1, CV_RGB ( 255, 0,0 ),3 );	
     xw=Vector3f(x3d(i,0),
		x3d(i,1),
		x3d(i,2));
		
    xw=K*( R*xw+t);//transform
    xw= xw/xw(2);
    circle ( image, cvPoint( xw(0) , xw(1)), 0, CV_RGB ( 0, 255, 0 ),5 );
    stringstream ss2;//create a stringstream
       ss2 << i;//add number to the stream
       
       putText(image, ss2.str(), Point(x2d(i,0),x2d(i,1)), fontFace, 1,
	       Scalar(0, 255, 100), thickness, 8);
       
  
   }
   
  //image points
   for (int j=0;j<points_projected.cols();j++){
     
     circle ( image, cvPoint(points_projected(0,j) ,points_projected(1,j)), 1, CV_RGB ( 0, 0,255 ),1 );
   
     
   }
   
   namedWindow( "Image2", CV_WINDOW_NORMAL );// Create a window for display.
   
   imshow( "Image2", image); 
      
   waitKey(0);
   ////////////////////////////////////////////////////////////////////////////
   //Finally publish data or save data
   ///////////////////////////////////////////////////////////////////////////
    SaveCalibrationData(R, t);
       cloudRGB->width=1;
       cloudRGB->height=cloudRGB->size();
    
  io::savePCDFileASCII("result_RGBD.pcd",*cloudRGB);
    PCL_INFO("pcl file saved: result_RGBD.pcd ");
   return 0;
   
  
}
