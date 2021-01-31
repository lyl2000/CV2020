#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <time.h>
#include "pch.h"
using namespace std;
using namespace cv;

int sigmaSpace, sigmaColor;
Mat img;

// 对称复制填充
void MirrorFill(const Mat &input, Mat &output, int length)
{
	output = Mat(input.rows + 2 * length, input.cols + 2 * length, input.type());
	for (int y = 0; y < input.rows; ++y)
	{
		const uchar *in = input.ptr<uchar>(y);
		uchar *out = output.ptr<uchar>(y + length);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x];
		}
	}
	for (int y = 0; y < length; ++y)  // 上边界复制
	{
		const uchar *in = output.ptr<uchar>(length + y);
		uchar *out = output.ptr<uchar>(length - 1 - y);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x + length];
		}
	}
	for (int y = 0; y < length; ++y)  // 下边界复制
	{
		const uchar *in = output.ptr<uchar>(input.rows + length - 1 - y);
		uchar *out = output.ptr<uchar>(input.rows + length + y);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x + length];
		}
	}
	for (int y = 0; y < input.rows + 2 * length; ++y)  // 左边界复制
	{
		const uchar *in = output.ptr<uchar>(y);
		uchar *out = output.ptr<uchar>(y);
		for (int x = 0; x < length; ++x)
		{
			out[length - 1 - x] = in[length + x];
		}
	}
	for (int y = 0; y < input.rows + 2 * length; ++y)  // 右边界复制
	{
		const uchar *in = output.ptr<uchar>(y);
		uchar *out = output.ptr<uchar>(y);
		for (int x = 0; x < length; ++x)
		{
			out[input.cols + length + x] = in[input.cols + length - 1 - x];
		}
	}
}

// 常量填充
void ConstantFill(const Mat &input, Mat &output, int length, int fill = 0)
{
	output = Mat(input.rows + 2 * length, input.cols + 2 * length, input.type());
	output = fill;
	for (int y = 0; y < input.rows; ++y)
	{
		const uchar *in = input.ptr<uchar>(y);
		uchar *out = output.ptr<uchar>(y + length);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x];
		}
	}
}

// 双边滤波
void MyFilter(const Mat &input, Mat &output, int sigmaColor, int sigmaSpace)
{
	int window_size = 5;
	Mat m1;
	MirrorFill(input, m1, window_size / 2);
	Mat m2(m1);	
	double *g = new double[window_size * window_size];
	for (int y = window_size / 2; y < input.rows + window_size / 2; ++y)
	{
		uchar *q = m1.ptr<uchar>(y);
		uchar *out = m2.ptr<uchar>(y);
		for (int x = window_size / 2; x < input.cols + window_size / 2; ++x)
		{
			double sum = 0.0;
			for (int j = 0; j < window_size; ++j)
			{
				int v = y - window_size / 2 + j;  // m1[v, u]
				uchar *p = m1.ptr<uchar>(v);
				for (int i = 0; i < window_size; ++i)
				{
					int u = x - window_size / 2 + i;
					g[j * window_size + i] = exp(-(pow(u - x, 2) + pow(v - y, 2)) / (2 * pow(sigmaSpace, 2))) * exp(-pow(p[u] - q[x], 2) / (2 * pow(sigmaColor, 2)));
					sum += g[j * window_size + i];
				}
			}
			double s = 0.0;
			for (int j = 0; j < window_size; ++j)
			{
				int v = y - window_size / 2 + j;  // m1[v, u]
				uchar *in = m1.ptr<uchar>(v);
				for (int i = 0; i < window_size; ++i)
				{
					int u = x - window_size / 2 + i;
					s += g[j * window_size + i] / sum * in[u];  // 归一化后卷积
				}
			}
			out[x] = (uchar)s;
		}
	}
	delete[] g;
	Rect roi(window_size / 2, window_size / 2, input.cols, input.rows);
	output = m2(roi);
}

void BilateralFilter(const Mat& input, Mat& output, int sigmaColor, int sigmaSpace)
{
	bilateralFilter(input, output, 5, sigmaColor, sigmaSpace);	
}

void trackbar(int, void*)
{
	Mat img2, img3;
	clock_t tic, toc;

	tic = clock();
	MyFilter(img, img2, sigmaColor, sigmaSpace);
	toc = clock();
	cout << "MyFilter total time: " << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;
	
	tic = clock();
	BilateralFilter(img, img3, sigmaColor, sigmaSpace);
	toc = clock();
	cout << "BilateralFilter total time: " << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;
	
	imshow("原图", img);
	imshow("My Filter", img2);
	imshow("Bilateral Filter", img3);
}

// 实验5 双边滤波
void exp()
{
	int max_sigmaSpace = 10, max_sigmaColor = 400;
	sigmaSpace = max_sigmaSpace / 2;
	sigmaColor = max_sigmaColor / 2;
	img = imread("C:/Users/20181014/Desktop/jpg.jpg", 0);
	img = img(Rect(0, img.rows / 4, img.cols / 2, img.rows / 2));  // 取一个小区域
	namedWindow("My Filter", WINDOW_AUTOSIZE);
	createTrackbar("Space", "My Filter", &sigmaSpace, max_sigmaSpace, trackbar);
	trackbar(sigmaSpace, 0);
	createTrackbar("Color", "My Filter", &sigmaColor, max_sigmaColor, trackbar);
	trackbar(sigmaColor, 0);
	waitKey();
}

int main()
{
	exp();
	return 0;
}
