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
Rect roi;  // Ŀ������
int hist[256 * 3];  // roi�����ֱ��ͼ
Point startPoint, endPoint;

// ����¼�
void onMouse(int event, int x, int y, int flags, void* ustc)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		startPoint = Point(x, y);
		endPoint = startPoint;  // �����غ�
		canDraw = true;
	}
	if (event == EVENT_MOUSEMOVE && canDraw)
	{
		endPoint = Point(x, y);
		roi = Rect(startPoint, endPoint);
		copyImage = image.clone();  // ��ԭͼ�ϻ������ºۼ�������ʱ�ĸ���Ʒ�ϻ���
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

// ��ȡĿ������
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
		// �������ʱ��ͣ  canDraw == true
		if (!canDraw)
		{
			capture >> image;  // �ȶ���һ֡
		}
		// Esc�����˳�
		if (!image.data || waitKey(wait_time) == 27)
		{
			break;
		}
		// ������Ŀ������������ѭ��
		if (hasTarget)
		{
			break;
		}
		// ��ͣ��û����ͼ��д�룬���涨�񣬷�����������
		if (image.data)
		{
			imshow("vedio", image);
		}
	}
	capture.release();
}

// ͼ��ͨ������
void getChannel(const uchar *input, int width, int height, int inStep, int inChannels, uchar *output, int outStep, int channelToGet)
{
	uchar* row = (uchar*)input;
	for (int y = 0; y < height; ++y, row += inStep)
	{
		uchar* px = row;
		for (int x = 0; x < width; ++x, px += inChannels)
		{
			output[y * outStep + x] = *(px + channelToGet);  // ��ͨ��������Ϊ�Ҷ�ͼ
		}
	}
}

// ���㵥ͨ����ֱ��ͼ
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

// ����ͼ���ֱ��ͼ
void calc_hist(const Mat& img, int hist[256 * 3])
{
	for (int ch = 0; ch < 3; ++ch)  // B G R
	{
		Mat channelImage(img.size(), CV_8UC1);
		getChannel(img.data, img.cols, img.rows, (int)img.step, 3, channelImage.data, (int)channelImage.step, ch);
		calc_hist(channelImage.data, channelImage.cols, channelImage.rows, (int)channelImage.step, hist + 256 * (2 - ch));  // R G B
	}
}

// ����ת��ΪHSV���Ч��
void calc_hist(const Mat& img, Mat& HSVHist)
{
	Mat HSVImage;
	int channels[] = { 0, 1 };
	int histSize[] = { 30, 32 };  // ��ɫ������Ϊ30���������Ͷ�����Ϊ32��
	float HRanges[] = { 0, 180 };  // hue varies from 0 to 179
	float SRanges[] = { 0, 256 };  // ���Ͷȴ�0���ڰף���255��������ɫ������
	const float *ranges[] = { HRanges, SRanges };
	cvtColor(img, HSVImage, COLOR_BGR2HSV);  // ת����HSV
	calcHist(&HSVImage, 1, channels, Mat(), HSVHist, 2, histSize, ranges, true, false);
	normalize(HSVHist, HSVHist, 0, 1, NORM_MINMAX);  // ��һ��
}

// ��ֱ��ͼ
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

// ����ֱ��ͼ��Ĳ���
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

// ʹ��OpenCV�ĺ��������Բ�ͬ�Ķ�������
double compare_hist(const Mat hist1, const Mat hist2)
{
	return compareHist(hist1, hist2, 3);
}

// ����Ŀ������
void find_area_similar_to_target_image(bool isUse)
{
	// ȷ����������
	int width = roi.width, height = roi.height;//cout << width << " " << height << endl;
	int x1 = roi.x - width / 2, x2 = roi.x + 1.5 * width;
	int y1 = roi.y - height / 2, y2 = roi.y + 1.5 * height;
	x1 = max(0, x1);
	y1 = max(0, y1);
	x2 = min(image.cols - 1, x2);
	y2 = min(image.rows - 1, y2);

	// ���¶���Ƶ
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
		// Ѱ������
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
		// ѡ��һ����ֵ
		if (isUse) threshold = 0.5;
		else threshold = 0.15;
		// �ƶ�׷��
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
// ��1������������¼�������Ƶ���ţ��ֶ�����Ŀ������
	namedWindow("vedio");
	setMouseCallback("vedio", onMouse);
	get_target_image();
	
// ��2��������Ŀ�������ֱ��ͼ����ʾ
	if (!targetImage.data)
	{
		cout << "no target image" << endl;
		return -1;  // û�л���Ŀ��������������
	}
	imshow("targetImage", targetImage);
	calc_hist(targetImage, hist);
	draw_hist(targetImage, hist);
	
// ��3������Ŀ������ĸ���������Ѱ����Ŀ�������ֱ��ͼ��ӽ��������ƶ���������ʵ��׷��
	find_area_similar_to_target_image(false);  // ��ʹ��HSV
	return 0;
}
