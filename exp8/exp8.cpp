// 霍夫变换
// 实现基于霍夫变换的图像圆检测
//（边缘检测可以用opencv的canny函数）
// 并尝试对其准确率和效率进行优化实现。

#include <opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;

int t1 = 200, t2 = 50;
Mat image, grayImage, edgeImage;
vector<Vec3f> circles;

void trackbar(int, void*) {
	Canny(grayImage, edgeImage, t1, t2);
	imshow("canny", edgeImage);
}

//自己实现基于霍夫变换的图像圆检测
void my_HoughCircles(const Mat& grayImage, vector<Vec3f>& circles, double param1, int minRadius, int maxRadius) {
	Canny(grayImage, edgeImage, param1, param1 / 2);
	int row = edgeImage.rows, col = edgeImage.cols, radius = maxRadius - minRadius;
	int *H = new int[row * col * radius];
	memset(H, 0, sizeof(int)*row*col*radius);
	Mat grad_x, grad_y;
	Mat abs_x, abs_y;
	Sobel(edgeImage, grad_x, CV_16S, 1, 0, 3, 1, 1);
	convertScaleAbs(grad_x, abs_x);
	Sobel(edgeImage, grad_y, CV_16S, 0, 1, 3, 1, 1);
	convertScaleAbs(grad_y, abs_y);
	for (int y = 0; y < edgeImage.rows; ++y) {
		for (int x = 0; x < edgeImage.cols; ++x) {
			if ((int)edgeImage.at<uchar>(y, x) == 0) {
				continue;
			}
			short dy = grad_y.at<short>(y, x);
			short dx = grad_x.at<short>(y, x);
			int theta = ((int)atan2(dy, dx) + 180) % 360;
			int minTheta, maxTheta;
			if (theta > 0 && theta < 90) {
				minTheta = 0, maxTheta = 90;
			}
			else if (theta < 180) {
				minTheta = 90, maxTheta = 180;
			}
			else if (theta < 270) {
				minTheta = 180, maxTheta = 270;
			}
			else {
				minTheta = 270, maxTheta = 360;
			}
			for (int r = minRadius; r < maxRadius; ++r) {
				for (int theta = minTheta; theta < maxTheta; theta += 5) {
					int v = y + int(r * sin(theta)), u = x - int(r * cos(theta));
					if (v >= 0 && v < row && u >= 0 && u < col) {
						H[v * col * radius + u * radius + r - minRadius] += 1;
					}
				}
			}
		}
	}
	int threshold = 11;
	for (int y = 0; y < row; ++y) {
		for (int x = 0; x < col; ++x) {
			for (int r = 0; r < radius; ++r) {
				if (H[y * col * radius + x * radius + r] > threshold) {
					circles.push_back(Vec3f((float)x, (float)y, (float)r + minRadius));
				}
			}
		}
	}
	delete[] H;
}

int main() {
	namedWindow("canny", WINDOW_AUTOSIZE);
	image = imread("exp8a.jpg", -1);
	imshow("原图", image);
	cvtColor(image, grayImage, COLOR_BGR2GRAY);
	GaussianBlur(grayImage, grayImage, Size(5, 5), 0, 0);

	clock_t tic = clock();
	//HoughCircles(grayImage, circles, HOUGH_GRADIENT, 1, grayImage.rows / 5, 150, 70, 0, 0);
	my_HoughCircles(grayImage, circles, t1, 50, 150);
	for (size_t i = 0; i < circles.size(); ++i) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(image, center, radius, Scalar(0, 255, 0));
	}
	imshow("结果", image);
	clock_t toc = clock();
	cout << "cost " << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;

	createTrackbar("threshold1", "canny", &t1, 200, trackbar);
	trackbar(t1, 0);
	createTrackbar("threshold2", "canny", &t2, 100, trackbar);
	trackbar(t2, 0);
	waitKey();
	return 0;
}
