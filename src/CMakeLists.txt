include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(FisheyeCalibOnSingleImage CalibIntrinsic.cpp main.cpp)
target_link_libraries(FisheyeCalibOnSingleImage cbdetect ${OpenCV_LIBS})
