//实现Harris角点检测算法，
//并与OpenCV的cornerHarris函数的结果进行比较。
#include <opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;

int thresh = 100;
Mat image, gray, dst;

Mat non_maximum_suppresion(const Mat& src, int wsize) {
	Mat RR(src);
	for (int y = wsize / 2; y < src.rows - wsize / 2; ++y) {
		for (int x = wsize / 2; x < src.cols - wsize / 2; ++x) {
			double now = src.at<double>(y, x);
			bool isLarge = true;
			for (int yy = y - wsize / 2; (yy <= y + wsize / 2) && isLarge; ++yy) {
				for (int xx = x - wsize / 2; xx <= x + wsize / 2; ++xx) {
					if (src.at<double>(yy, xx) > now) {
						isLarge = false;
						break;
					}
				}
			}
			if (isLarge) {
				RR.at<double>(y, x) = now;
			}
			else {
				RR.at<double>(y, x) = 0;
			}
		}
	}
	return RR;
}

void harris(Mat& src, Mat& dst, double k, bool isGauss) {
	Mat Ix = Mat::zeros(src.size(), CV_64FC1), Iy = Mat::zeros(src.size(), CV_64FC1);
	Mat Ixx = Mat::zeros(src.size(), CV_64FC1), Iyy = Mat::zeros(src.size(), CV_64FC1);
	Mat Ixy = Mat::zeros(src.size(), CV_64FC1), R = Mat::zeros(src.size(), CV_64FC1);

	Sobel(src, Ix, CV_64FC1, 0, 1);
	Sobel(src, Iy, CV_64FC1, 1, 0);

	int wsize = 3;  // 定义窗口大小
	for (int y = wsize / 2; y < src.rows - wsize / 2; ++y) {
		for (int x = wsize / 2; x < src.cols - wsize / 2; ++x) {
			double ixx = 0, iyy = 0, ixy = 0;
			for (int yy = y - wsize / 2; yy <= y + wsize / 2; ++yy) {
				for (int xx = x - wsize / 2; xx <= x + wsize / 2; ++xx) {
					double xv = Ix.at<double>(yy, xx), yv = Iy.at<double>(yy, xx);
					ixx += xv * xv; iyy += yv * yv; ixy += xv * yv;
				}
			}
			double r = ixx * iyy - ixy * ixy - k * (ixx + iyy) * (ixx + iyy);
			Ixx.at<double>(y, x) = ixx;
			Iyy.at<double>(y, x) = iyy;
			Ixy.at<double>(y, x) = ixy;
			R.at<double>(y, x) = r;
		}
	}
	if (isGauss) {
		GaussianBlur(Ixx, Ixx, Size(3, 3), 0, 0);
		GaussianBlur(Iyy, Iyy, Size(3, 3), 0, 0);
		GaussianBlur(Ixy, Ixy, Size(3, 3), 0, 0);
		for (int y = wsize / 2; y < src.rows - wsize / 2; ++y) {
			for (int x = wsize / 2; x < src.cols - wsize / 2; ++x) {
				double ixx = 0, iyy = 0, ixy = 0;
				for (int yy = y - wsize / 2; yy <= y + wsize / 2; ++yy) {
					for (int xx = x - wsize / 2; xx <= x + wsize / 2; ++xx) {
						double xv = Ixx.at<double>(yy, xx), yv = Iyy.at<double>(yy, xx), xyv = Ixy.at<double>(yy, xx);
						ixx += xv; iyy += yv; ixy += xyv;
					}
				}
				double r = ixx * iyy - ixy * ixy - k * (ixx + iyy) * (ixx + iyy);
				R.at<double>(y, x) = r;
			}
		}
	}
	// 非极大值抑制
	dst = non_maximum_suppresion(R, 5);
}

void my_harris(int, void*) {
	clock_t tic = clock();
	harris(gray, dst, 0.04, false);
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_64FC1);
	convertScaleAbs(dst, dst);
	Mat out = image.clone();
	for (int y = 0; y < out.rows; ++y) {
		uchar *row = dst.ptr<uchar>(y);
		for (int x = 0; x < out.cols; ++x) {
			if ((int)row[x] > thresh) {
				circle(out, Point(x, y), 2, Scalar(0, 255, 0));
			}
		}
	}
	clock_t toc = clock();
	cout << "cost " << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;
	imshow("my", out);
}

void cv_harris(int, void*) {
	clock_t tic = clock();
	dst = Mat(gray.size(), CV_64FC1);
	cornerHarris(gray, dst, 2, 3, 0.04, BORDER_DEFAULT);
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_64FC1);
	convertScaleAbs(dst, dst);
	Mat out = image.clone();
	for (int y = 0; y < out.rows; ++y) {
		uchar *row = dst.ptr<uchar>(y);
		for (int x = 0; x < out.cols; ++x) {
			if ((int)row[x] > thresh) {
				circle(out, Point(x, y), 2, Scalar(0, 255, 0));
			}
		}
	}
	clock_t toc = clock();
	cout << "cost " << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;
	imshow("cv", out);
}

int main() {
	string path = "ling.jpg";
	image = imread(path, -1);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("cv", WINDOW_AUTOSIZE);
	createTrackbar("threshold", "cv", &thresh, 255, cv_harris);
	cv_harris(0, 0);
	namedWindow("my", WINDOW_AUTOSIZE);
	createTrackbar("threshold", "my", &thresh, 255, my_harris);
	my_harris(0, 0);
	waitKey();
	return 0;
}
