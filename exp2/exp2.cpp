#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
using namespace std;
using namespace cv;

Mat img1, img2;
int contrast;

// 调整对比度
void change_contrast(int, void*)
{
	for (int y = 0; y < img1.rows; ++y)
	{
		for (int x = 0; x < img1.cols; ++x)
		{
			for (int c = 0; c < 3; ++c)
			{
				img2.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(255 / (1 + exp((127 - img1.at<Vec3b>(y, x)[c]) * (contrast + 20.0) / 4000)));
			}
		}
	}
	imshow("origin-img", img1);
	imshow("new-img", img2);
}

// 实验2.1
void exp1()
{
	string path = "C:/Users/20181014/Desktop/jpg.jpg";
	namedWindow("origin-img", WINDOW_AUTOSIZE);
	namedWindow("new-img", WINDOW_AUTOSIZE);
	img1 = imread(path, -1);
	imshow("origin-img", img1);
 	img2 = Mat(img1.size(), img1.type());
	img2 = 0;
	// 设定初始值
	contrast = 50;
	// 添加slider
	createTrackbar("对比度", "new-img", &contrast, 100, change_contrast);
	// 交互调整
	change_contrast(contrast, 0);
}

// 背景相减
void background_subtract(Mat imgI, Mat imgB, int threshold)
{
	Mat img = Mat(imgI.size(), imgI.type());
	img = 0;
	for (int y = 0; y < imgI.rows; ++y)
	{
		for (int x = 0; x < imgI.cols; ++x)
		{
			int diff = 0;
			for (int c = 0; c < 3; ++c)
			{
				auto tmp = imgI.at<Vec3b>(y, x)[c] - imgB.at<Vec3b>(y, x)[c];
				diff += tmp * tmp;
			}
			int color;
			// 大于阈值设为白否则为黑
			if (diff > threshold) color = 255;
			else color = 0;

			for (int c = 0; c < 3; ++c)
			{
				img.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(color);
			}
		}
	}
	imshow("img" + to_string(threshold), img);
}

// 实验2.2
void exp2()
{
	string path0 = "C:/Users/20181014/Desktop/CV/exp/exp2/bgs-data/";
	string pathI = "13.jpg", pathB = "13_bg.jpg";
	Mat imgI = imread(path0 + pathI, -1);
	Mat imgB = imread(path0 + pathB, -1);
	imshow("imgI", imgI);
	imshow("imgB", imgB);
	// 尝试多个阈值
	for (int i = 2; i <= 8; ++i)
	{
		background_subtract(imgI, imgB, 1000 * i);
	}
}

int main()
{
	exp1();
	exp2();

	waitKey();
	return 0;
}
