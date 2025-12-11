/**
 *
 * @Name:
 * @Description: xxxxxxx
 * @Version: v1.0
 * @Date: 2019-03-18
 * @Copyright (c) 2019 BYD Co.,Ltd
 * @Author: luo peng <luopengchn@yeah.net>
 *
 */
#include "CalibIntrinsic.h"

using namespace cv;
using namespace std;


int main(int argc, char *argv[]) {

    CalibIntrinsic::Setting setting;
    Mat img1 = imread(setting.get_path_image());
    if (img1.empty()) {
        std::cout << "please check the image path" << std::endl;
        return -1;
    }
    Mat img = img1.clone();  // Use original image without mask
    /**
     * dectect corners
     */
    cbdetect::Corner corners;
    std::vector<cbdetect::Board> boards;
    CalibIntrinsic::detect(img, cbdetect::SaddlePoint, boards, corners);
    /**
     * calculate corner points960
     */
    vector<vector<Point2f>> cornerPointSeq = CalibIntrinsic::calcCornerImagePoints(boards, corners);
    vector<vector<Point3f>> objectPointSeq = CalibIntrinsic::calcCornerWorldPoints(boards, setting.get_size_square());
    /**
     *  wide-angle calibration
     */
    cv::Mat intrinsicMat;
    cv::Mat distCoeffs;
    vector<cv::Vec3d> rVec, tVec;
    assert(objectPointSeq.size() == cornerPointSeq.size());
    CalibIntrinsic::calibIntrinsic(objectPointSeq, cornerPointSeq, img.size(), intrinsicMat, distCoeffs, rVec, tVec);
    cout << "intrinsic:" << intrinsicMat << endl;
    /**
     * evaluate calibration
     */
    vector<vector<Point2f >> imageProjectPointSeq;
    imageProjectPointSeq.clear();
    CalibIntrinsic::calibEvaluate(objectPointSeq, cornerPointSeq, imageProjectPointSeq, boards, rVec,
                                  tVec, intrinsicMat, distCoeffs);
    SVM_ASSERT((int) imageProjectPointSeq.size() == (int) cornerPointSeq.size());
    Mat img_show = img.clone();
    int imgProjectNum = imageProjectPointSeq.size();
    for (int i = 0; i < imgProjectNum; ++i) {

        for (int j = 0; j < imageProjectPointSeq[i].size(); ++j) {
            circle(img_show, imageProjectPointSeq[i][j], 1, cv::Scalar(0, 0, 255), -1);
            circle(img_show, cornerPointSeq[i][j], 1, cv::Scalar(0, 255, 0), -1);
            //set threshold
            if (fabs(imageProjectPointSeq[i][j].x - cornerPointSeq[i][j].x) > 1.2) {
                cout << "----------------------------" << endl;
                cout << "calibrate error" << endl;
                cout << "board" << i << " id:" << (j) << "  cornerPoint:" << cornerPointSeq[i][j]
                     << "  worldPoint:" << objectPointSeq[i][j] << "  ProjectPoint:" << imageProjectPointSeq[i][j]
                     << endl;
//                break;
            }
        }

    }

    /**
     * undistor image
     */
    Mat undistort_img;
    CalibIntrinsic::undistort(img, undistort_img, intrinsicMat, distCoeffs);
    if (setting.get_save_debug_image()) {
        imwrite("UndistorImage.png", undistort_img);
        imwrite("project_point.jpg", img_show);
        cout << "Debug images saved" << endl;
    }
    CalibIntrinsic::saveIntrinsicFile(setting.get_path_intrinsic(), intrinsicMat, distCoeffs);
    cout << "Calibrate OK..." << endl;
    return 0;
}