#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <sys/types.h>
#include <dirent.h>

class Rectifier {

public:

    std::map<int, cv::Mat> intrinsics;
    std::map<int, cv::Mat> distortions;
    std::map<int, cv::Mat> rotations;
    std::map<int, cv::Mat> translations;

    
    Rectifier() {};
    
    void parseCalibrationFile(char *filename) {

        std::ifstream file(filename);
        // if(!file.is_open()) 
        //     throw 1; // TODO: do this properly!
        
        
        int camera_id;
        double parameter;

        while(!file.eof()) {
            
            file >> camera_id;

            /*
             * 3x3 intrisics
             */
            intrinsics[camera_id] = cv::Mat(3,3,CV_64F);
            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 3; j++)
                    file >> intrinsics[camera_id].at<double>(i, j);


            /*
             * 4x1 distortions
             */
            distortions[camera_id] = cv::Mat(4,1,CV_64F);
            file >> distortions[camera_id].at<double>(0,0);
            file >> distortions[camera_id].at<double>(0,1);
            distortions[camera_id].at<double>(0,2) = 0;
            distortions[camera_id].at<double>(0,3) = 0;


            /*
             * rotation and translation
             */
            rotations[camera_id] = cv::Mat(3,3,CV_64F);
            translations[camera_id] = cv::Mat(3,1,CV_64F);
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    file >> rotations[camera_id].at<double>(i, j);
                }
                file >> translations[camera_id].at<double>(i);
            }
        }
        file.close();
    };

    void printCameraMatrices(int camera_id) {

        std::cout << "Intrinsics:" << "\n";
        
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                std::cout << intrinsics[camera_id].at<double>(i,j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        
        std::cout << "Distortion:" << "\n";
        
        for(int i = 0; i < 4; i++)
            std::cout << distortions[camera_id].at<double>(i) << " ";
        std::cout << "\n";
        std::cout << "\n";

        std::cout << "Rotation:" << "\n";
        
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                std::cout << rotations[camera_id].at<double>(i,j) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        
        std::cout << "Translation:" << "\n";
        
        for(int i = 0; i < 3; i++) 
            std::cout << translations[camera_id].at<double>(i) << " ";
        std::cout << "\n";
        std::cout << "\n";
    }

    void printMat(cv::Mat mat) {

        std::cout << mat << "\n";
        
        // for(int i = 0; i < mat.rows; i++) {
        //     for(int j = 0; j < mat.cols; j++) {

        //         switch(mat.type()) {
        //         case CV_32F:
        //             std::cout << mat.at<float>(i,j) << " ";
        //             break;
        //         case CV_64F:
        //             std::cout << mat.at<double>(i,j) << " ";
        //             break;
        //         case CV_8U:
        //             std::cout << (int)mat.at<char>(i,j) << " ";
        //             break;
        //             case CV_16U
        //         }
                
        //     }
        //     std::cout << "\n";
        // }
        
    }
    
    void rectify(cv::Mat image_l, cv::Mat image_r, cv::Mat rectified_l, cv::Mat rectified_r, int camera_id_l, int camera_id_r) {

        cv::Mat R1;
        cv::Mat R2;
        cv::Mat P1;
        cv::Mat P2;
        cv::Mat Q;
        cv::Mat mapx1;
        cv::Mat mapx2;
        cv::Mat mapy1;
        cv::Mat mapy2;

        cv::stereoRectify(intrinsics[camera_id_l], distortions[camera_id_l],
                          intrinsics[camera_id_r], distortions[camera_id_r],
                          image_l.size(),
                          rotations[camera_id_r],
                          translations[camera_id_r],
                          R1, R2, P1, P2, cv::noArray());

        cv::initUndistortRectifyMap(intrinsics[camera_id_l], distortions[camera_id_l], R1, P1,
                                    image_l.size(), image_l.type(), mapx1, mapy1);
        
        cv::initUndistortRectifyMap(intrinsics[camera_id_r], distortions[camera_id_r], R2, P2,
                                    image_r.size(), image_r.type(), mapx2, mapy2);
        
        //cv::remap(image_l, rectified_l, mapx1, mapy1, cv::INTER_LINEAR);
        cv::remap(image_r, rectified_r, mapx2, mapy2, cv::INTER_LINEAR);

    };

};

int main(int argc, char *argv[]) {

    if(argc < 4) {
        std::cerr << "Too few arguments" << std::endl;
        return 1;
    }
    argc--; argv++;

    Rectifier r;
    r.parseCalibrationFile(argv[0]);

    
    //    cv::namedWindow("right");
    cv::namedWindow("left");

    std::stringstream output_name_left;
    std::stringstream output_name_right;
    
    argc--; argv++;
    char c = 0, cc = 0;
    for(int i = 0; i < argc/2; i++) {

        //std::cout << "Left: " << argv[i] << " Right: " << argv[i+argc/2] << "\n";
        cv::Mat image_l = cv::imread(argv[i], 0);
        cv::Mat image_r = cv::imread(argv[i+argc/2], 0);
        cv::Mat rectified_l = image_l.clone();//(image_l.size(), image_l.type());
        cv::Mat rectified_r = image_l.clone(); //(image_l.size(), image_l.type());
        
        
        r.rectify(image_l, image_r, rectified_l, rectified_r, 4, 3);

        output_name_left.str("");
        output_name_left << "dataset/" << "f" << std::setw(3) << std::setfill('0') << i << "_rectified_left.bmp";
        output_name_right.str("");
        output_name_right << "f" << std::setw(3) << std::setfill('0') << i << "_rectified_right.bmp";

        std::cout << output_name_left.str() << ", " << output_name_right.str() << "\n";

        // cv::imwrite(output_name_left.str(), rectified_l);
        // cv::imwrite(output_name_right.str(), rectified_r);
           

        cv::imshow("left", rectified_l);
        c = (char)cv::waitKey(40);
        cv::imshow("left", rectified_r);
        cc = (char)cv::waitKey(40);

        if(c == 'q' || cc == 'q')
            break;
    }

    std::cout << "Done" << "\n";
    

    return 0;
}
