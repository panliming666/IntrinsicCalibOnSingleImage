/**
 *
 * @Name:
 * @Description: xxxxxxx
 * @Version: v1.0
 * @Date: 2019-03-23
 * @Copyright (c) 2019 BYD Co.,Ltd
 * @Author: luo peng <luopengchn@yeah.net>
 *
 */

#ifndef LIBCBDETECT_SETTING_H
#define LIBCBDETECT_SETTING_H

// xml  intrisic files node
#define NODE_NAME_COEFF      "coeffs"
#define NODE_NAME_INTRINSIC  "intrinsicMat"
#define NODE_NAME_DATE  "date"

#ifndef SVM_PI
#define SVM_PI 3.1415926f
#endif

#define PropertyBuilder(type, name)\
    inline void set_##name( type &v) {\
          name = v;\
    }\
    inline type get_##name() {\
        return name;\
    }\


namespace CalibIntrinsic {

    class Setting {

     public:

        Setting() :
                save_debug_image(true),
                 show_debug_info(true),
                size_square(cv::Size(20, 20)),//20mm x 20mm
//                path_image("../../example_data/0711/wuling0514/3/0.jpg"),
//                path_intrinsic("../../example_data/0711/wuling0514/3/intrinsic_coeffs_left0.xml"){}
                path_image("/mnt/nvme/byd/amr/camera_calibration/IntrinsicCalibOnSingleImage/example_data/test/test.png"),
                path_intrinsic("/mnt/nvme/byd/amr/camera_calibration/IntrinsicCalibOnSingleImage/example_data/test/test.xml"){}


        PropertyBuilder(bool, save_debug_image)

        PropertyBuilder(bool, show_debug_info)

        PropertyBuilder(cv::Size, size_square)

        PropertyBuilder(std::string, path_image)

        PropertyBuilder(std::string, path_intrinsic)

    private:
        bool save_debug_image;
        bool show_debug_info;
        cv::Size size_square;
        std::string path_intrinsic;
        std::string path_image;
    };

}

#endif //LIBCBDETECT_SETTING_H
