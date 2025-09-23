#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// 1. 相机参数：
Mat camera_matrix = (Mat_<double>(3, 3) << 1120, 0, 1120,
    0, 1120, 630,
    0, 0, 1);
// 畸变矩阵：
Mat dist_coeffs = (Mat_<double>(5, 1) << -0.1, 0.05, 0.0, 0.0, 0.0);
// 世界坐标：
Mat objectPoints = (Mat_<float>(4, 3) << 0, 0, 94,
    102, 0, 94,
    102, 0, 0,
    0, 0, 0);
// 输出旋转、平移向量
Mat rvec, tvec;

int main() {
    VideoCapture cap(0);
    Mat src;
    if (!cap.isOpened()) {
        cout << "相机打开失败！" << endl;
        return -1;
    }
    // 读取第一帧：增加失败处理，避免后续初始化错误
    if (!cap.read(src)) {
        cout << "无法读取第一帧！" << endl;
        return -1;
    }

    // 初始化视频写入器：增加打开失败判断（避免路径权限问题）
    bool isColor = (src.type() == CV_8UC3);
    VideoWriter writer;
    int codec = writer.fourcc('M', 'J', 'P', 'G');
    double fps = 25.0;
    string filename = "./live.avi";
    if (!writer.open(filename, codec, fps, src.size(), isColor)) {
        cout << "视频文件创建失败！检查路径是否有权限。" << endl;
        return -1;
    }

   

    // 主循环：逐帧处理
    for (;;) {
        // 读取当前帧：失败则退出循环（避免死循环打印ERROR）
        if (!cap.read(src)) {
            cout << "帧读取失败，退出程序！" << endl;
            break;
        }

       
        Mat bgr = src;
        Mat hsv, dst;
        cvtColor(src, hsv, COLOR_BGR2HSV);
        dst = Mat::zeros(bgr.size(), bgr.type());
     
        Mat mask;
        inRange(hsv, Scalar(136, 32, 67), Scalar(165, 175, 186), mask);
        
        bgr.copyTo(dst, mask);
        namedWindow("dst");
        
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
        morphologyEx(dst, dst, MORPH_CLOSE, element, Point(-1, -1), 2);
        Mat finall_dst, finall_dst_1;
        GaussianBlur(dst, finall_dst, Size(3, 3), 0, 0);
        Canny(finall_dst, finall_dst_1, 50, 200);

        // 查找轮廓+筛选最大轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hirearchy;
        vector<Point> approx;
        int max_contour_idx = -1;
        double max_contour_area = -1;
        findContours(finall_dst_1, contours, hirearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

      
        Mat imagePoints;
        for (int t = 0; t < contours.size(); t++) {
            double contour_len = arcLength(contours[t], true);
            double contour_area = contourArea(contours[t]);

            
            if (contour_area > 400 && contour_len > 0.03 * 80) {
                if (contour_area > max_contour_area) {
                    max_contour_area = contour_area;
                    max_contour_idx = t;
                }
            }
        }

        
        bool pnp_success = false;
        if (max_contour_idx != -1) {
            // 多边形逼近：获取轮廓的近似形状
            double approx_accuracy = 0.03 * arcLength(contours[max_contour_idx], true);
            approxPolyDP(contours[max_contour_idx], approx, approx_accuracy, true);

            // 计算最小外接矩形
            Rect bounding_rect = boundingRect(approx);
            rectangle(src, bounding_rect, Scalar(0, 255, 0), 2); // 绘制矩形，可视化目标

           
            imagePoints = (Mat_<float>(4, 2) << bounding_rect.x, bounding_rect.y,
                bounding_rect.x + bounding_rect.width, bounding_rect.y,
                bounding_rect.x + bounding_rect.width, bounding_rect.y + bounding_rect.height,
                bounding_rect.x, bounding_rect.y + bounding_rect.height);

            pnp_success = solvePnP(
                objectPoints,
                imagePoints,
                camera_matrix,
                dist_coeffs,
                rvec, tvec,
                false,
                SOLVEPNP_ITERATIVE
            );
        }

       
        if (pnp_success) {
            Mat axis_3d = (Mat_<float>(4, 3) <<
                51, 0, 47,    
                51 + 51, 0, 47, 
                51, 0 + 51, 47, 
                51, 0, 47 + 51  
                );

            Mat newImagePoint; // 投影后的2D浮点数坐标
            projectPoints(
                axis_3d,
                rvec, tvec,
                camera_matrix,
                dist_coeffs,
                newImagePoint
            );

            float x = newImagePoint.at<Vec2f>(0)[0];
            float y = newImagePoint.at<Vec2f>(0)[1];
            Point2i start_pt(cvRound(x), cvRound(y)); // 浮点数→整数坐标
            for (int i = 0; i < newImagePoint.rows; i++) {
                // 当前点：转换为整数坐标
                float xi = newImagePoint.at<Vec2f>(i)[0];  // 第i行的x坐标
                float yi = newImagePoint.at<Vec2f>(i)[1];  // 第i行的y坐标
                Point2i end_pt(cvRound(xi), cvRound(yi));   // 终点（整数化）

                // 画线段：起点→终点，蓝色（255,0,0），线宽2（与Python参数一致）
                cv::line(src, start_pt, end_pt, cv::Scalar(255, 0, 0), 2);
                
            }
        }
        else {
           
            cout << "当前帧未检测到有效目标，或solvePnP求解失败！" << endl;
        }

        // 写入视频+显示画面
        writer.write(src);
        imshow("Live", src);
        imshow("dst", dst); // 显示目标提取效果，便于调试

        waitKey(1);
        // 等待按键退出
        int c = waitKey(40);
        if (c >= 0) {
            break;
        }
    }

    // 释放资源：避免内存泄漏
    cap.release();
    writer.release();
    destroyAllWindows();

    return 0;
}