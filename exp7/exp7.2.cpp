#include<opencv2\opencv.hpp>
#include<algorithm>
#include<iostream>
using namespace std;
using namespace cv;

void disTrans(const Mat& image, Mat dis_image) {
	int row = image.rows, col = image.cols;
	for (int y = 0; y < row; ++y) {
		const uchar *p = image.ptr<uchar>(y);
		int *dp = dis_image.ptr<int>(y);
		for (int x = 0; x < col; ++x) {
			if (p[x] == 0) dp[x] = 0;
			else if (p[x] == 255) dp[x] = UCHAR_MAX;
		}
	}
	for (int x = 1; x < col; ++x) {
		dis_image.at<int>(0, x) = min(dis_image.at<int>(0, x), dis_image.at<int>(0, x - 1) + 1);
	}
	for (int y = 1; y < row; ++y) {
		dis_image.at<int>(y, 0) = min(dis_image.at<int>(y, 0), dis_image.at<int>(y - 1, 0) + 1);
	}
	for (int y = 1; y < row; ++y) {
		for (int x = 1; x < col; ++x) {
			dis_image.at<int>(y, x) = min(dis_image.at<int>(y, x), dis_image.at<int>(y - 1, x) + 1);
			dis_image.at<int>(y, x) = min(dis_image.at<int>(y, x), dis_image.at<int>(y, x - 1) + 1);
		}
	}

	for (int x = col - 2; x >= 0; --x) {
		dis_image.at<int>(row - 1, x) = min(dis_image.at<int>(row - 1, x), dis_image.at<int>(row - 1, x + 1) + 1);
	}
	for (int y = row - 2; y >= 0; --y) {
		dis_image.at<int>(y, col - 1) = min(dis_image.at<int>(y, col - 1), dis_image.at<int>(y + 1, col - 1) + 1);
	}
	for (int y = row - 2; y >= 0; --y) {
		for (int x = col - 2; x >= 0; --x) {
			dis_image.at<int>(y, x) = min(dis_image.at<int>(y, x), dis_image.at<int>(y + 1, x) + 1);
			dis_image.at<int>(y, x) = min(dis_image.at<int>(y, x), dis_image.at<int>(y, x + 1) + 1);
		}
	}

	
}

Mat cv_distance_transform(const Mat& image) {
	Mat dis_image(image.size(), CV_32FC1);
	distanceTransform(image, dis_image, DIST_L2, 0);
	Mat show_image(image.size(), CV_8UC1);
	for (int y = 0; y < image.rows; ++y) {
		uchar *row1 = show_image.ptr<uchar>(y);
		float *row2 = dis_image.ptr<float>(y);
		for (int x = 0; x < image.cols; ++x) {
			row1[x] = (uchar)row2[x];
		}
	}
	return show_image;
}

Mat my_distance_transform(const Mat& image) {
	Mat dis_image(image.size(), CV_32SC1);
	disTrans(image, dis_image);
	Mat show_image(image.size(), CV_8UC1);
	for (int y = 0; y < image.rows; ++y) {
		uchar *row1 = show_image.ptr<uchar>(y);
		int *row2 = dis_image.ptr<int>(y);
		for (int x = 0; x < image.cols; ++x) {
			row1[x] = (uchar)row2[x];
		}
	}
	return show_image;
}

int main(){
	Mat image = imread("exp7b.jpg", 0);
	imshow("image", image);
	threshold(image, image, 205, 255, THRESH_BINARY);
	imshow("binary", image);
	Mat show_image = cv_distance_transform(image);
	//Mat show_image = my_distance_transform(image);
	imshow("distance", show_image);
	waitKey();
	return 0;
}
