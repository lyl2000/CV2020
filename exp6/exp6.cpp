#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

const string path = "demo.avi";
Mat image, copyImage, targetImage;
bool canDraw = false;
bool hasTarget = false;
Rect roi;  // 目标区域
int hist[256 * 3];  // roi区域的直方图
Point startPoint, endPoint;

// 鼠标事件
void onMouse(int event, int x, int y, int flags, void* ustc)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		startPoint = Point(x, y);
		endPoint = startPoint;  // 两点重合
		canDraw = true;
	}
	if (event == EVENT_MOUSEMOVE && canDraw)
	{
		endPoint = Point(x, y);
		roi = Rect(startPoint, endPoint);
		copyImage = image.clone();  // 在原图上画会留下痕迹，在临时的复制品上画框
		rectangle(copyImage, roi, Scalar(255, 255, 255));
		imshow("vedio", copyImage);
	}
	if (event == EVENT_LBUTTONUP)
	{
		rectangle(copyImage, roi, Scalar(0, 255, 0));
		imshow("vedio", copyImage);
		targetImage = image(roi);
		canDraw = false;
		hasTarget = true;
	}
}

// 获取目标区域
void get_target_image()
{
	VideoCapture capture(path);
	if (!capture.isOpened())
	{
		cout << "vedio open error" << endl;
		return;
	}
	double fps = capture.get(CAP_PROP_FPS);
	int wait_time = 1000.0 / fps;
	while (true)
	{
		// 左键按下时暂停  canDraw == true
		if (!canDraw)
		{
			capture >> image;  // 先读入一帧
		}
		// Esc或播完退出
		if (!image.data || waitKey(wait_time) == 27)
		{
			break;
		}
		// 若划出目标区域，先跳出循环
		if (hasTarget)
		{
			break;
		}
		// 暂停后没有新图像写入，画面定格，否则正常播放
		if (image.data)
		{
			imshow("vedio", image);
		}
	}
	capture.release();
}

// 图像通道分离
void getChannel(const uchar *input, int width, int height, int inStep, int inChannels, uchar *output, int outStep, int channelToGet)
{
	uchar* row = (uchar*)input;
	for (int y = 0; y < height; ++y, row += inStep)
	{
		uchar* px = row;
		for (int x = 0; x < width; ++x, px += inChannels)
		{
			output[y * outStep + x] = *(px + channelToGet);  // 单通道，表现为灰度图
		}
	}
}

// 计算单通道的直方图
void calc_hist(uchar *data, int width, int height, int step, int H[256])
{
	memset(H, 0, sizeof(H[0]) * 256);
	uchar *row = data;
	for (int y = 0; y < height; ++y, row += step)
	{
		for (int x = 0; x < width; ++x)
		{
			H[row[x]]++;
		}
	}
}

// 计算图像的直方图
void calc_hist(const Mat& img, int hist[256 * 3])
{
	for (int ch = 0; ch < 3; ++ch)  // B G R
	{
		Mat channelImage(img.size(), CV_8UC1);
		getChannel(img.data, img.cols, img.rows, (int)img.step, 3, channelImage.data, (int)channelImage.step, ch);
		calc_hist(channelImage.data, channelImage.cols, channelImage.rows, (int)channelImage.step, hist + 256 * (2 - ch));  // R G B
	}
}

// 尝试转换为HSV后的效果
void calc_hist(const Mat& img, Mat& HSVHist)
{
	Mat HSVImage;
	int channels[] = { 0, 1 };
	int histSize[] = { 30, 32 };  // 将色调量化为30级，将饱和度量化为32级
	float HRanges[] = { 0, 180 };  // hue varies from 0 to 179
	float SRanges[] = { 0, 256 };  // 饱和度从0（黑白）到255（纯光谱色）不等
	const float *ranges[] = { HRanges, SRanges };
	cvtColor(img, HSVImage, COLOR_BGR2HSV);  // 转换到HSV
	calcHist(&HSVImage, 1, channels, Mat(), HSVHist, 2, histSize, ranges, true, false);
	normalize(HSVHist, HSVHist, 0, 1, NORM_MINMAX);  // 归一化
}

// 画直方图
void draw_hist(const Mat& img, int hist[256 * 3])
{
	Mat hist_picture(256, 256 * 3, CV_8UC3, Scalar(0, 0, 0));
	int n = img.rows * img.cols;
	int max_hist = 0;
	for (int i = 0; i < 256 * 3; ++i) max_hist = max(max_hist, hist[i]);
	Scalar color[] = { Scalar(0, 0, 255) , Scalar(0, 255, 0) , Scalar(255, 0, 0) };
	for (int i = 0; i < 256; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			double value = 1.0 * hist[i] / max_hist;//double value = 1.0 * hist[i] / n;
			line(hist_picture, Point(i + 256 * j, 256), Point(i + 256 * j, (1 - value) * 256), color[j]);
		}
	}
	imshow("Histogram", hist_picture);
}

// 计算直方图间的差异
double compare_hist(const int hist1[256 * 3], const int hist2[256 * 3])
{
	int n = targetImage.rows * targetImage.cols;
	double diff = 0.0;
	for (int i = 0; i < 256 * 3; ++i)
	{
		diff += (hist1[i] - hist2[i]) * (hist1[i] - hist2[i]) * 1.0;
	}
	return diff / (n * n); //sqrt(diff);
}

// 使用OpenCV的函数，尝试不同的度量方法
double compare_hist(const Mat hist1, const Mat hist2)
{
	return compareHist(hist1, hist2, 3);
}

// 跟踪目标区域
void find_area_similar_to_target_image(bool isUse)
{
	// 确定搜索区域
	int width = roi.width, height = roi.height;//cout << width << " " << height << endl;
	int x1 = roi.x - width / 2, x2 = roi.x + 1.5 * width;
	int y1 = roi.y - height / 2, y2 = roi.y + 1.5 * height;
	x1 = max(0, x1);
	y1 = max(0, y1);
	x2 = min(image.cols - 1, x2);
	y2 = min(image.rows - 1, y2);

	// 重新读视频
	VideoCapture capture(path);
	int W = capture.get(CAP_PROP_FRAME_WIDTH);
	int H = capture.get(CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(CAP_PROP_FPS);
	int wait_time = 1000.0 / fps;

	VideoWriter writer("video.avi", VideoWriter::fourcc('M', 'P', '4', '2'), fps / 2, Size(W, H), true);
	if (!writer.isOpened())
	{
		cout << "writer open error" << endl;
		return;
	}
	int margin = 10;
	Mat HSVHist;
	double threshold, diff;
	if (isUse) calc_hist(targetImage, HSVHist);
	while (true)
	{
		capture >> image;
		if (!image.data || waitKey(wait_time) == 27)
		{
			break;
		}
		rectangle(image, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0));
		double minDiff = 1;
		Rect rect;
		// 寻找区域
		for (int y = y1; y + height <= y2; y += margin)
		{
			for (int x = x1; x + width <= x2; x += margin)
			{
				startPoint = Point(x, y);
				endPoint = Point(x + width, y + height);
				Mat sameImage = image(Rect(startPoint, endPoint));
				if (isUse)
				{
					Mat tmpHist;
					calc_hist(sameImage, tmpHist);
					diff = compare_hist(HSVHist, tmpHist);
				}
				else
				{
					int tmpHist[256 * 3];
					calc_hist(sameImage, tmpHist);
					diff = compare_hist(hist, tmpHist);
				}
				if (diff < minDiff)
				{
					minDiff = diff;
					rect = Rect(startPoint, endPoint);
				}
			}
		}
		cout << minDiff << endl;
		// 选择一个阈值
		if (isUse) threshold = 0.5;
		else threshold = 0.15;
		// 移动追踪
		if (minDiff < threshold)
		{
			x1 = rect.x - rect.width / 2;
			x2 = rect.x + 1.5 * rect.width;
			y1 = rect.y - rect.height / 2;
			y2 = rect.y + 1.5 * rect.height;
			x1 = max(0, x1);
			y1 = max(0, y1);
			x2 = min(image.cols - 1, x2);
			y2 = min(image.rows - 1, y2);
			rectangle(image, rect, Scalar(0, 0, 255));
		}
		if (image.data)
		{
			writer.write(image);
			imshow("similarImage", image);
		}
	}
	capture.release();
	writer.release();
}

int main()
{
// 第1步：设置鼠标事件，打开视频播放，手动划出目标区域
	namedWindow("vedio");
	setMouseCallback("vedio", onMouse);
	get_target_image();
	
// 第2步：计算目标区域的直方图并显示
	if (!targetImage.data)
	{
		cout << "no target image" << endl;
		return -1;  // 没有划出目标区域的特殊情况
	}
	imshow("targetImage", targetImage);
	calc_hist(targetImage, hist);
	draw_hist(targetImage, hist);
	
// 第3步：在目标区域的附近遍历，寻找与目标区域的直方图最接近的区域并移动到该区域实现追踪
	find_area_similar_to_target_image(false);  // 不使用HSV
	return 0;
}
