#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <time.h>
using namespace std;
using namespace cv;

int sigma = 5;
int sz = 5;
Mat img;

// 对称复制填充
void MirrorFill(const Mat &input, Mat &output, int length)
{
	output = Mat(input.rows + 2 * length, input.cols + 2 * length, input.type());
	for (int y = 0; y < input.rows; ++y)
	{
		const Vec3b *in = input.ptr<Vec3b>(y);
		Vec3b *out = output.ptr<Vec3b>(y + length);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x];
		}
	}
	for (int y = 0; y < length; ++y)  // 上边界复制
	{
		const Vec3b *in = output.ptr<Vec3b>(length + y);
		Vec3b *out = output.ptr<Vec3b>(length - 1 - y);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x + length];
		}
	}
	for (int y = 0; y < length; ++y)  // 下边界复制
	{
		const Vec3b *in = output.ptr<Vec3b>(input.rows + length - 1 - y);
		Vec3b *out = output.ptr<Vec3b>(input.rows + length + y);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x + length];
		}
	}
	for (int y = 0; y < input.rows + 2 * length; ++y)  // 左边界复制
	{
		const Vec3b *in = output.ptr<Vec3b>(y);
		Vec3b *out = output.ptr<Vec3b>(y);
		for (int x = 0; x < length; ++x)
		{
			out[length - 1 - x] = in[length + x];
		}
	}
	for (int y = 0; y < input.rows + 2 * length; ++y)  // 右边界复制
	{
		const Vec3b *in = output.ptr<Vec3b>(y);
		Vec3b *out = output.ptr<Vec3b>(y);
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
		const Vec3b *in = input.ptr<Vec3b>(y);
		Vec3b *out = output.ptr<Vec3b>(y + length);
		for (int x = 0; x < input.cols; ++x)
		{
			out[x + length] = in[x];
		}
	}
}

// 高斯滤波
void Gaussian(const Mat &input, Mat &output, double sigma)
{
	int window_size = int(sigma * 6 - 1);  // 向下取整且是奇数
	if (window_size % 2 == 0) window_size--;
	Mat m1;
	// ConstantFill(input, m1, window_size / 2, 0);  // 边界用常数0填充
	MirrorFill(input, m1, window_size / 2);
	Mat m2(m1);

	double *g = new double[window_size];
	double sum = 0.0;
	for (int i = 0; i < window_size; ++i)
	{
		g[i] = exp(-pow(i - window_size / 2, 2) / (2 * pow(sigma, 2)));
		sum += g[i];
	}
	for (int i = 0; i < window_size; ++i)
	{
		g[i] /= sum;  // 归一化
	}
	for (int y = 0; y <= m2.rows - window_size / 2; ++y)  // 行
	{
		Vec3b *out = m2.ptr<Vec3b>(y), *in = m1.ptr<Vec3b>(y);
		for (int x = 0; x <= m2.cols - window_size; ++x)
		{
			for (int c = 0; c < 3; ++c)
			{
				double s = 0;
				for (int t = 0; t < window_size; ++t)
				{
					s += in[x + t][c] * g[t];
				}
				out[x + window_size / 2][c] = (int)s;
			}
		}
	}
	m1 = m2;
	for (int y = 0; y <= m2.rows - window_size; ++y)  // 列
	{
		Vec3b *out = m2.ptr<Vec3b>(y + window_size / 2);
		for (int x = 0; x <= m2.cols - window_size / 2; ++x)
		{
			for (int c = 0; c < 3; ++c)
			{
				double s = 0;
				for (int t = 0; t < window_size; ++t)
				{
					Vec3b *in = m1.ptr<Vec3b>(y + t);
					s += in[x][c] * g[t];
				}
				out[x][c] = (int)s;
			}
		}
	}
	delete[] g;
	Rect roi(window_size / 2, window_size / 2, input.rows, input.cols);
	output = m2(roi);
}

void trackbar1(int, void*)
{
	Mat img2;
	Gaussian(img, img2, sigma + 1);  // sigma最小值不能为0
	imshow("原图", img);
	imshow("Gaussian", img2);
}

// 实验4.1 高斯滤波
void exp1()
{
	img = imread("C:/Users/20181014/Desktop/jpg.jpg", -1);
	namedWindow("Gaussian", WINDOW_AUTOSIZE);
	createTrackbar("sigma", "Gaussian", &sigma, 10, trackbar1);
	trackbar1(sigma, 0);
	waitKey();
}

// 计算积分图
void integral_image(const uchar *src, int width, int height, int sstride, int *pint, int istride)
{
	int *prow = new int[width];
	memset(prow, 0, sizeof(int)*width);
	for (int yi = 0; yi < height; ++yi, src += sstride, pint += istride)
	{
		prow[0] += src[0];  pint[0] = prow[0];   //for the first pixel
		for (int xi = 1; xi < width; ++xi)
		{
			prow[xi] += src[xi];
			pint[xi] = pint[xi - 1] + prow[xi];
		}
	}
	delete[] prow;
}

// 均值滤波
void MeanFilter(const Mat &input, Mat &output, int window_size)
{
	clock_t tic, toc;
	if (window_size % 2 == 0) window_size--;
	Mat m1;
	// ConstantFill(input, m1, window_size / 2, 0);
	MirrorFill(input, m1, window_size / 2);
	// imshow("m1", m1);
	tic = clock();
	Mat m2(m1);
	// 计算m1的积分图
	int *pint = new int[m1.rows * m1.cols];
	Mat *m = new Mat[3];
	cv::split(m1, m);
	for (int c = 0; c < 3; ++c)
	{
		integral_image(m[c].data, m1.cols, m1.rows, m[c].step, pint, m1.cols);
		for (int y = 0; y < m1.rows - window_size; ++y)
		{
			Vec3b *out = m2.ptr<Vec3b>(y + window_size / 2);
			for (int x = 0; x < m1.cols - window_size; ++x)
			{
				// y, x, y+sz, x+sz
				out[x + window_size / 2][c] = (pint[(y + window_size)*m1.cols + (x + window_size)]
					- pint[y*m1.cols + (x + window_size)] - pint[(y + window_size)*m1.cols + x]
					+ pint[y*m1.cols + x]) * 1.0 / (window_size * window_size);
			}
		}
	}
	Rect roi(window_size / 2, window_size / 2, input.cols, input.rows);
	output = m2(roi);
	toc = clock();
	cout << "积分图加速 total time: " << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;
}

void CvFilter(const Mat &input, Mat &output, int window_size)
{
	clock_t tic, toc;
	if (window_size % 2 == 0) window_size--;
	Mat m1;
	// ConstantFill(input, m1, window_size / 2, 0);
	MirrorFill(input, m1, window_size / 2);

	// OpenCV boxFilter
	tic = clock();
	Mat m2(m1);
	boxFilter(m1, m2, -1, Size(window_size, window_size));
	Rect roi(window_size / 2, window_size / 2, input.rows, input.cols);
	output = m2(roi);
	toc = clock();
	cout << "boxFilter total time: " << (double)(toc - tic) / CLOCKS_PER_SEC << "s\n" << endl;
}

void trackbar2(int, void*)
{
	Mat img2, img3;
	MeanFilter(img, img2, sz + 3);  // 小于3的卷积核没意义
	CvFilter(img, img3, sz + 3);
	imshow("img", img);  // 原图
	imshow("MeanFilter", img2);  // MeanFilter得到的图像
	imshow("CvFilter", img3);  // OpenCV boxFilter得到的图像
}

// 实验4.2 快速均值滤波
void exp2()
{
	img = imread("C:/Users/20181014/Desktop/jpg.jpg", -1);
	namedWindow("MeanFilter", WINDOW_AUTOSIZE);
	createTrackbar("size", "MeanFilter", &sz, img.rows / 2, trackbar2);
	trackbar2(sigma, 0);
	waitKey();
}

int main()
{
	exp1();
	exp2();
	return 0;
}
