#include<iostream>
#include<opencv2/opencv.hpp>
#include"opencv2\imgproc\types_c.h"
using namespace std;
using namespace cv;
//回调函数
struct all_img {
	Mat dst;//输出图像
	Mat img;//数入图像
	Mat hsv;//hsv图像
	Mat bgr;//bgr图像
	int hmin, smin, vmin;
	int hmax, smax, vmax;
};
//void callback(int, void* userdata) {
//	all_img* allImg = static_cast<all_img*>(userdata);
//	allImg->dst = Mat::zeros(allImg->img.size(), allImg->img.type());
//	Mat mask;
//	inRange(allImg->hsv, Scalar(allImg->hmin, allImg->smin, allImg->vmin), Scalar(allImg->hmax, allImg->smax, allImg->vmax), mask);
//	//掩码到原图的转换
//	for (int r = 0; r < allImg->bgr.rows; r++) {
//		for (int c = 0; c < allImg->bgr.cols; c++) {
//			if (mask.at<uchar>(r, c) == 255)
//				allImg->dst.at<Vec3b>(r, c) = allImg->bgr.at<Vec3b>(r, c);
//
//		}
//	}
//	imshow("dst", allImg->dst);
//}

int main() {
	all_img allimg;
	allimg.img = imread("tupian.PNG");
	if (allimg.img.empty()) {
		cout << "图片加载错误" << endl;
	}
	/*int width = 600;
	int height = 400;
	resize(allimg.img, allimg.img, Size(width, height), INTER_LINEAR);*/
	allimg.bgr = allimg.img;
	cvtColor(allimg.img, allimg.hsv, COLOR_BGR2HSV);
	allimg.dst = Mat::zeros(allimg.bgr.size(), allimg.bgr.type());
	namedWindow("dst");
	/*allimg.hmin = 0;
	allimg.smin = 0;
	allimg.vmin = 0;
	allimg.hmax = 179;
	allimg.smax = 255;
	allimg.vmax = 255;
	
	createTrackbar("hmin", "dst", &allimg.hmin,  360, callback, &allimg);
	createTrackbar("hmax", "dst", &allimg.hmax, 179, callback, &allimg);
	createTrackbar("smin", "dst", &allimg.smin, 255, callback, &allimg);
	createTrackbar("smax", "dst", &allimg.smax, 255, callback, &allimg);
	createTrackbar("vmin", "dst", &allimg.vmin, 255, callback, &allimg);
	createTrackbar("vmax", "dst", &allimg.vmax, 255, callback, &allimg);
	callback(0, &allimg);*///(78,40,168)  (104,104,255)
	Mat mask;
	inRange(allimg.hsv, Scalar(78, 40, 168), Scalar(104, 104, 255), mask);
	for (int r = 0; r < allimg.bgr.rows; r++) {
		for (int c = 0; c < allimg.bgr.cols; c++) {
			if (mask.at<uchar>(r, c) == 255)
				allimg.dst.at<Vec3b>(r, c) = allimg.bgr.at<Vec3b>(r, c);
		}
	}

//降低噪音
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(-1, -1));
	morphologyEx(allimg.dst, allimg.dst, MORPH_CLOSE,element,Point(-1,-1),3);
	Mat finall_dst,finall_dst_1;
	GaussianBlur(allimg.dst, finall_dst, Size(3, 3), 0, 0);
	Canny(finall_dst, finall_dst_1, 50, 200);
//查找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hirearchy;
	vector<Point> approx;
	findContours(finall_dst_1, contours, hirearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	for (int t = 0; t < contours.size(); t++) {
		double len =0.03* arcLength(contours[t], true);
		double area = contourArea(contours[t]);
		int number;
		if (area > 320 && len > 0.03 * 80) {
			number = t;
			approxPolyDP(contours[number], approx, len, true);
			//计算多边形的最小外接矩形
			Rect bounding_rect = boundingRect(approx);
			rectangle(allimg.img, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
		}
	}
	
	imshow("dst", allimg.img);
	waitKey(0);
	destroyAllWindows();  
	return 0;
}