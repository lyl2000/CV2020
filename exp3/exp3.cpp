#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;

// 双线性插值
Vec3b bilinear(double y, double x, const Mat& img)
{	
	int width = img.cols, height = img.rows;
	int y1 = floor(y), y2 = min((int)ceil(y), height - 1), x1 = floor(x), x2 = min((int)ceil(x), width - 1);
	double dy = y - y1, dx = x - x1;
	Vec3b A = img.at<Vec3b>(y1, x1), B = img.at<Vec3b>(y1, x2), C = img.at<Vec3b>(y2, x1), D = img.at<Vec3b>(y2, x2);
	Vec3b h1 = A + dx * (B - A), h2 = C + dx * (D - C);
	return (h1 + dy * (h2 - h1));
}

// 图像变形
Mat image_distortion(const Mat& img1)
{
	Mat img2 = Mat(img1.size(), img1.type());
	int width = img2.cols, height = img2.rows;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			// 中心归一化坐标
			double v = (2.0 * y - height) / height, u = (2.0 * x - width) / width;
			double r2 = u * u + v * v;  // r^2
			double r = sqrt(r2);
			if (r >= 1)
			{
				// img2.at<Vec3b>(y, x) = bilinear((v + 1) * height / 2, (u + 1) * width / 2, img1);
				img2.at<Vec3b>(y, x) = img1.at<Vec3b>(y, x);
			}
			else
			{
				double theta = 1 - 2 * r + r2;
				double newY = sin(theta) * u + cos(theta) * v;
				double newX = cos(theta) * u - sin(theta) * v;
				// 还原归一化的坐标
				img2.at<Vec3b>(y, x) = bilinear((newY + 1) * height / 2, (newX + 1) * width / 2, img1);
			}
		}
	}
	return img2;
}

// 放大哈哈镜
Mat haha1(const Mat& img1)
{
	Mat img2(img1.size(), img1.type());
	int width = img1.cols, height = img1.rows;
	Point center(width / 2, height / 2);
	double R = sqrt(width * width + height * height) / 2;  // 半径为R的圆形内放大图像
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			double dis = norm(Point(x, y) - center);  // 获得当前点到中心点的距离
			if (dis < R)  // 设置变化区间
			{
				int newX = (x - center.x) * dis / R + center.x;
				int newY = (y - center.y) * dis / R + center.y;
				// 防止访问越界
				if (newX < 0) newX = 0;
				else if (newX >= width) newX = width - 1;
				if (newY < 0) newY = 0;
				else if (newY >= height) newY = height - 1;
				img2.at<Vec3b>(y, x) = img1.at<Vec3b>(newY, newX);
			}
			else
			{
				img2.at<Vec3b>(y, x) = img1.at<Vec3b>(y, x);
			}
		}
	}
	return img2;
}

// 缩小哈哈镜
Mat haha2(const Mat& img1) {
	Mat img2 = Mat(img1.size(), img1.type());
	int width = img2.cols, height = img2.rows;
	Point center(width / 2, height / 2);  // 根据离中心点的距离设置缩放
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			double theta = atan2((double)(y - center.y), (double)(x - center.x));
			double dis = sqrt(norm(Point(x, y) - center)) * 10;
			// 缩小
			int newX = center.x + (int)(dis * cos(theta));
			int newY = center.y + (int)(dis * sin(theta));
			// 防止访问越界
			if (newX < 0) newX = 0;
			else if (newX >= width) newX = width - 1;
			if (newY < 0) newY = 0;
			else if (newY >= height) newY = height - 1;
			img2.at<Vec3b>(y, x) = img1.at<Vec3b>(newY, newX);
		}
	}
	return img2;
}

// 实验3.1
void exp1()
{
	string path = "C:/Users/20181014/Desktop/img1.jpg";
	Mat img1 = imread(path, -1);
	imshow("原图像", img1);

	Mat img2 = image_distortion(img1);
	imshow("变形图像", img2);
	waitKey();
}

// 实验3.2
void exp2()
{
	// 打开摄像头
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "video open error" << endl;
		return;
	}
	Mat frame;
	
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	// 帧数大时视频倍速很快
	VideoWriter writer("video.avi", VideoWriter::fourcc('M', 'P', '4', '2'), 10, Size(width, height), true);
	if (!writer.isOpened())
	{
		cout << "writer open error" << endl;
		return;
	}

	bool stop = false;
	int ha = 0;
	while (!stop)
	{
		// 读取帧
		if (!capture.read(frame))
		{
			cout << "no video frame" << endl;
			break;
		}

		// do_something(frame)
		if (ha == 1)
		{
			frame = haha1(frame);
		}
		else if (ha == 2)
		{
			frame = haha2(frame);
		}
		else if (ha == 3)
		{
			frame = image_distortion(frame);
		}

		writer.write(frame);  // 录制视频
		imshow("video", frame);

		int key = waitKey(10);
		if (key == 27)  // 退出
		{
			stop = true;
			cout << "finish" << endl;
		}
		else if (key == 49)  // 摁 1 放大哈哈镜效果
		{
			ha = 1;
			cout << "haha1 effect" << endl;
		}
		else if (key == 50)  // 摁 2 缩小哈哈镜效果
		{
			ha = 2;
			cout << "haha2 effect" << endl;
		}
		else if(key == 51)  // 摁 3 实验3.1效果
		{
			ha = 3;
			cout << "no effect" << endl;
		}
		else if (key == 52)  // 摁 4 无效果
		{
			ha = 0;
			cout << "3.1 effect" << endl;
		}
	}
	capture.release();
	writer.release();
}

int main()
{
	exp1();
	exp2();
	return 0;
}
