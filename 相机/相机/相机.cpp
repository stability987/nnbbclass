#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// 1. ���������
Mat camera_matrix = (Mat_<double>(3, 3) << 1120, 0, 1120,
    0, 1120, 630,
    0, 0, 1);
// �������
Mat dist_coeffs = (Mat_<double>(5, 1) << -0.1, 0.05, 0.0, 0.0, 0.0);
// �������꣺
Mat objectPoints = (Mat_<float>(4, 3) << 0, 0, 94,
    102, 0, 94,
    102, 0, 0,
    0, 0, 0);
// �����ת��ƽ������
Mat rvec, tvec;

int main() {
    VideoCapture cap(0);
    Mat src;
    if (!cap.isOpened()) {
        cout << "�����ʧ�ܣ�" << endl;
        return -1;
    }
    // ��ȡ��һ֡������ʧ�ܴ������������ʼ������
    if (!cap.read(src)) {
        cout << "�޷���ȡ��һ֡��" << endl;
        return -1;
    }

    // ��ʼ����Ƶд���������Ӵ�ʧ���жϣ�����·��Ȩ�����⣩
    bool isColor = (src.type() == CV_8UC3);
    VideoWriter writer;
    int codec = writer.fourcc('M', 'J', 'P', 'G');
    double fps = 25.0;
    string filename = "./live.avi";
    if (!writer.open(filename, codec, fps, src.size(), isColor)) {
        cout << "��Ƶ�ļ�����ʧ�ܣ����·���Ƿ���Ȩ�ޡ�" << endl;
        return -1;
    }

   

    // ��ѭ������֡����
    for (;;) {
        // ��ȡ��ǰ֡��ʧ�����˳�ѭ����������ѭ����ӡERROR��
        if (!cap.read(src)) {
            cout << "֡��ȡʧ�ܣ��˳�����" << endl;
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

        // ��������+ɸѡ�������
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
            // ����αƽ�����ȡ�����Ľ�����״
            double approx_accuracy = 0.03 * arcLength(contours[max_contour_idx], true);
            approxPolyDP(contours[max_contour_idx], approx, approx_accuracy, true);

            // ������С��Ӿ���
            Rect bounding_rect = boundingRect(approx);
            rectangle(src, bounding_rect, Scalar(0, 255, 0), 2); // ���ƾ��Σ����ӻ�Ŀ��

           
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

            Mat newImagePoint; // ͶӰ���2D����������
            projectPoints(
                axis_3d,
                rvec, tvec,
                camera_matrix,
                dist_coeffs,
                newImagePoint
            );

            float x = newImagePoint.at<Vec2f>(0)[0];
            float y = newImagePoint.at<Vec2f>(0)[1];
            Point2i start_pt(cvRound(x), cvRound(y)); // ����������������
            for (int i = 0; i < newImagePoint.rows; i++) {
                // ��ǰ�㣺ת��Ϊ��������
                float xi = newImagePoint.at<Vec2f>(i)[0];  // ��i�е�x����
                float yi = newImagePoint.at<Vec2f>(i)[1];  // ��i�е�y����
                Point2i end_pt(cvRound(xi), cvRound(yi));   // �յ㣨��������

                // ���߶Σ������յ㣬��ɫ��255,0,0�����߿�2����Python����һ�£�
                cv::line(src, start_pt, end_pt, cv::Scalar(255, 0, 0), 2);
                
            }
        }
        else {
           
            cout << "��ǰ֡δ��⵽��ЧĿ�꣬��solvePnP���ʧ�ܣ�" << endl;
        }

        // д����Ƶ+��ʾ����
        writer.write(src);
        imshow("Live", src);
        imshow("dst", dst); // ��ʾĿ����ȡЧ�������ڵ���

        waitKey(1);
        // �ȴ������˳�
        int c = waitKey(40);
        if (c >= 0) {
            break;
        }
    }

    // �ͷ���Դ�������ڴ�й©
    cap.release();
    writer.release();
    destroyAllWindows();

    return 0;
}