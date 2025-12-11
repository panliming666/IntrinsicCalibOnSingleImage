/**
 *
 * @Name:
 * @Description: xxxxxxx
 * @Version: v1.0
 * @Date: 2019-03-12
 * @Copyright (c) 2019 BYD Co.,Ltd
 * @Author: luo peng <luopengchn@yeah.net>
 *
 */

#ifndef FISHEYECALIB_H_H
#define FISHEYECALIB_H_H

#include <vector>
#include <chrono>
#include "config.h"
#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/plot_boards.h"
#include "libcbdetect/plot_corners.h"
#include <opencv2/opencv.hpp>
#include "configuration.h"
#include "log_defs.h"

namespace CalibIntrinsic {

    cv::Mat addMask(cv::Mat imgSrc, cv::Size mask_size);

    cv::Mat addMask(cv::Mat imgSrc, cv::Rect rect);

    void saveIntrinsicFile(std::string fileName, cv::Mat camMatrix, cv::Mat coeffs);

    void detect(cv::Mat img, cbdetect::CornerType corner_type, std::vector<cbdetect::Board> &boards,
                cbdetect::Corner &corners);

    std::vector<std::vector<cv::Point2f>> calcCornerImagePoints(std::vector<cbdetect::Board> boards,
                                                                cbdetect::Corner corners);

    std::vector<std::vector<cv::Point3f>>
    calcCornerWorldPoints(std::vector<cbdetect::Board> boards, cv::Size corner_size);

    void calibIntrinsic(std::vector<std::vector<cv::Point3f>> objectPointSeq,
                        std::vector<std::vector<cv::Point2f>> cornerPointSeq,
                        cv::Size imgSize, cv::Mat &intrinsicMat, cv::Mat &distCoeffs,
                        std::vector<cv::Vec3d> &rVecMat, std::vector<cv::Vec3d> &tVecMat);

    void undistort(cv::Mat distorImage, cv::Mat &undistort_img, cv::Mat intrinsicMat, cv::Mat distCoeffs);

    void calibEvaluate(const std::vector<std::vector<cv::Point3f>> &objectPointSeq,
                       const std::vector<std::vector<cv::Point2f>> &cornerPointSeq,
                       std::vector<std::vector<cv::Point2f >> &imageProjectPointSeq,
                       const std::vector<cbdetect::Board> &boards, const std::vector<cv::Vec3d> &rVecMat,
                       const std::vector<cv::Vec3d> &tVecMat, cv::Mat intrinsicMat, cv::Mat distCoeffs);

};


#endif //FISHEYECALIB_H_H
