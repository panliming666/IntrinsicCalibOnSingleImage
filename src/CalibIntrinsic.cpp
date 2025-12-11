
#include "CalibIntrinsic.h"
#include "configuration.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

namespace CalibIntrinsic {
/**
 * save intrinsic parameters to xml file
 * @param fileName
 * @param camMatrix
 * @param coeffs
 */
    void saveIntrinsicFile(string fileName, const Mat camMatrix, const cv::Mat coeffs) {
//    SVM_ASSERT(!fileName.empty());
//    SVM_ASSERT(!rotationMat.empty() && !translationMat.empty());
        FileStorage fs(fileName, FileStorage::WRITE);
        time_t rawtime;
        time(&rawtime);
        fs << NODE_NAME_DATE << asctime(localtime(&rawtime));
        fs << NODE_NAME_INTRINSIC << camMatrix;
        fs << NODE_NAME_COEFF << coeffs;
        fs.release();
        cout << "save successed：" << fileName << endl;
    }

    void detect(Mat img, cbdetect::CornerType corner_type, std::vector<cbdetect::Board> &boards,
                cbdetect::Corner &corners) {

        cbdetect::Params params;
        params.corner_type = corner_type;
        params.show_processing = false;
        params.show_debug_image = false;
        params.show_grow_processing = false;
        auto t1 = high_resolution_clock::now();
        cbdetect::find_corners(img, corners, params);
        auto t2 = high_resolution_clock::now();
        auto t3 = high_resolution_clock::now();
        cbdetect::boards_from_corners(img, corners, boards, params);
        auto t4 = high_resolution_clock::now();
        printf("Find corners took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0);
        printf("Find boards took: %.3f ms\n", duration_cast<microseconds>(t4 - t3).count() / 1000.0);
        printf("Total took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0
                                        + duration_cast<microseconds>(t4 - t3).count() / 1000.0);

    }

/**
 * Calculate corner 2-D point on image plane
 * @param boards
 * @param corners
 * @return
 */
    vector<vector<Point2f>> calcCornerImagePoints(vector<cbdetect::Board> boards, cbdetect::Corner corners) {
        vector<vector<Point2f>> cornerPointSeq;
        vector<Point2f> cornerPoints;
        for (int n = 0; n < boards.size(); ++n) {
            const auto &board = boards[n];
            cornerPoints.clear();
            for (int i = 2; i < board.idx.size() - 1; ++i) {
                for (int j = 2; j < board.idx[i].size() - 1; ++j) {
                    // if (board.idx[i][j] < 0) continue;
                    cornerPoints.push_back(corners.p[board.idx[i][j]]);
                }
            }
            cornerPointSeq.push_back(cornerPoints);
        }
        return cornerPointSeq;
    }


    cv::Mat addMask(cv::Mat imgSrc, cv::Rect rect) {
        cv::Mat imgDest;
        int img_width = imgSrc.cols, img_height = imgSrc.rows;
        cv::Mat mask_black = cv::Mat::zeros(img_height, img_width, CV_8U);
        mask_black(rect).setTo(255);
        imgSrc.copyTo(imgDest, mask_black);
        return imgDest;
    }


    cv::Mat addMask(cv::Mat imgSrc, cv::Size mask_size) {

        cv::Mat imgDest;
        int img_width = imgSrc.cols, img_height = imgSrc.rows;
        Mat mask_black = Mat::zeros(img_height, img_width, CV_8U);
        mask_black(cv::Rect(mask_size.width, mask_size.height, (img_width - mask_size.width * 2),
                            (img_height - mask_size.height * 2))).setTo(255);
        imgSrc.copyTo(imgDest, mask_black);
        return imgDest;
    }

/**
 * Calculate corner 3-D world points
 * @param boards
 * @param corners
 * @return
 */
    vector<vector<Point3f>> calcCornerWorldPoints(vector<cbdetect::Board> boards, cv::Size corner_size) {
        vector<vector<Point3f>> objectPointSeq;
        vector<Point3f> objectPoints;
        for (int n = 0; n < boards.size(); ++n) {
            const auto &board = boards[n];
            objectPoints.clear();
            for (int i = 2; i < board.idx.size() - 1; ++i) {
                if (i == 2) {
                    cout << "board" << n << "boardsize :" << board.idx.size() - 2
                         << "x" << board.idx[i].size() - 2 << endl;
                }
                for (int j = 2; j < board.idx[i].size() - 1; ++j) {

                    Point3f tempPoint;
                    tempPoint.x = (i - 2) * corner_size.height;
                    tempPoint.y = (j - 2) * corner_size.width;
                    tempPoint.z = 0;
                    objectPoints.push_back(tempPoint);
                }
            }
            objectPointSeq.push_back(objectPoints);
        }
        return objectPointSeq;
    }

/**
 * fisheye intrinsic calibrate
 * @param objectPointSeq
 * @param cornerPointSeq
 * @param imgSize
 * @param intrinsicMat
 * @param distCoeffs
 * @param rVecMat
 * @param tVecMat
 */
    void calibIntrinsic(const vector<vector<Point3f>> objectPointSeq,
                        const vector<vector<Point2f>> cornerPointSeq,
                        Size imgSize, Mat &intrinsicMat, cv::Mat &distCoeffs, vector<cv::Vec3d> &rVec,
                        vector<cv::Vec3d> &tVecMat) {
        int flags = 0;
        // flags |= cv::CALIB_FIX_K3;  // 可选：固定k3系数
        cv::calibrateCamera(objectPointSeq, cornerPointSeq, imgSize, intrinsicMat, distCoeffs, rVec,
                            tVecMat, flags, cv::TermCriteria(3, 20, 1e-6));
    }

//    void calibIntrinsic(const vector<vector<Point3f>> objectPointSeq,
//                        const vector<vector<Point2f>> cornerPointSeq,
//                        Size imgSize, Mat &intrinsicMat, cv::Vec4d &distCoeffs, vector<cv::Vec3d> &rVec,
//                        vector<cv::Vec3d> &tVecMat) {
//        int flags = 0;
//        flags |= cv::CALIB_FIX_K3;
////        cv::Mat intrinsic_matrix= cv::Mat::eye(3,3,CV_32FC1);
////        cv::Mat distortion_coeffs, rotation_vectors, translation_vectors;
//        cv::calibrateCamera(objectPointSeq, cornerPointSeq, imgSize,
//                            intrinsicMat, distCoeffs, rVec, tVecMat,flags);
//    }

/**
 * undistort image
 * @param distorImage
 * @param undistort_img
 * @param intrinsicMat
 * @param distCoeffs
 */
    void undistort(Mat distorImage, Mat &undistort_img, const Mat intrinsicMat, const cv::Mat distCoeffs)
    {
        cv::undistort(distorImage, undistort_img, intrinsicMat, distCoeffs);
    }

/**
 * TODO need
 */
/**
 * TODO need
 */
    void calibEvaluate(const vector<vector<Point3f>> &objectPointSeq,
                       const vector<vector<Point2f>> &cornerPointSeq,
                       vector<vector<Point2f >> &imageProjectPointSeq,
                       const vector<cbdetect::Board> &boards, const vector<cv::Vec3d> &rVec,
                       const vector<cv::Vec3d> &tVec, Mat intrinsicMat, cv::Mat distCoeffs) {

        double total_err = 0.0;                 //all image sum of average errors
        double err = 0.0;                       // single image average error
        vector<Point2f> imagePoints2;           // re-project image points
        int boardNum = boards.size();
        imageProjectPointSeq.clear();
        for (int n = 0; n < boardNum; ++n) {

            imagePoints2.clear();
            //contain new project image 2-D points
            cv::projectPoints(objectPointSeq[n], rVec[n], tVec[n], intrinsicMat, distCoeffs, imagePoints2);
            //cacluate error between original image points and re-project image points
            imageProjectPointSeq.push_back(imagePoints2);
            vector<Point2f> tempImagePoint = cornerPointSeq[n];
            Mat tempImagePointMat = Mat(1, (int) tempImagePoint.size(), CV_32FC2);
            Mat imagePoints2Mat = Mat(1, (int) imagePoints2.size(), CV_32FC2);
            for (int i = 0; i != tempImagePoint.size(); i++) {
                imagePoints2Mat.at<Vec2f>(0, i) = Vec2f(imagePoints2[i].x, imagePoints2[i].y);
                tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
            }
            err = norm(imagePoints2Mat, tempImagePointMat, NORM_L2);
            total_err += err /= objectPointSeq[n].size();
            cout << "board " << n << " average error：" << err << " (pixel)" << endl;
        }
        cout << "total average of images：" << total_err / boardNum << " (pixel)" << endl;
        cout << "evaluate calibration success！" << endl;
    }

}