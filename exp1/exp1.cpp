#include <iostream>
#include <string>
#include <opencv2\opencv.hpp> //加载OpenCV 头文件

using namespace std;
using namespace cv; //opencv的命名空间

uchar* get_pixel(const uchar *img, int x, int y, int step, int nc)
{
	// 返回想访问的像素的地址
	return (uchar*)img + y * step + x * nc;
}

/*
input, width, height, inStep, inChannels分别是输入图像的数据、宽、高、step和通道数。
Output和outStep是输出图像的数据和step，宽高与输入图像相同、通道数为1.
channelToGet是要获取的通道索引，比如如果input是BGR格式的数据，
则channelToGet=0将获取B通道, channelToGet=1将获取G通道……
*/
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
	/*
	若想表现为RGB图，则将格式变为CV_8UC3，
	访问地址修改为output[y*outStep+x*inChannels+channelToGet],
	输出到get_pixel(output, x1, y1, outStep, inChannels)地址处
	*/

	// 相当于另一种实现(3通道RGB格式)
	/*
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			for (int ch = 0; ch < inChannels; ++ch)
			{
				if (ch == channelToGet)
					output[y * outStep + x * inChannels + ch] = input[y * inStep + x * inChannels + ch];
				else
					output[y * outStep + x * inChannels + ch] = 0;
			}
		}
	}
	*/
}

void subRegion(const uchar *input, int width, int height, int inStep, int inChannels, uchar *output, int outStep, int channelToGet, cv::Rect rect)
{
	int x1 = rect.x, y1 = rect.y, x2 = rect.x + rect.width, y2 = rect.y + rect.height;
	// 调用getChannel
	getChannel(get_pixel(input, x1, y1, inStep, inChannels), rect.width, rect.height, inStep, inChannels, get_pixel(output, x1, y1, outStep, 1), outStep, channelToGet);
}

int main()
{
	string path = "C:/Users/20181014/Desktop/";
	string jpg = "jpg.jpg";
	string png = "png.png";
	string bmp = "bmp.bmp";

	// 实验1.1 图像加载显示
	Mat img; //声明一个保存图像的类
	img = imread(path + jpg); //读取图像，根据图片所在位置填写路径即可 3种格式的图片分别显示看看
	if (img.empty()) //判断图像文件是否存在
	{
		cout << "请确认图像文件名称是否正确" << endl;
		return -1;
	}
	imshow("img", img); //显示图像

	int row = img.rows, col = img.cols, channels = (int)img.channels(), channel = 0;
	cout << "总通道数：" << channels << " 读取通道数：" << channel << endl;
	Mat img2(row, col, CV_8UC1);
	Mat img3(row, col, CV_8UC1);

	// 实验1.2 图像通道分离
	getChannel(img.data, col, row, (int)img.step, channels, img2.data, (int)img2.step, channel);
	
	// 实验1.3 图像子区域操作 基于1.2
	cv::Rect rect(row / 3, col / 3, row / 3, col / 3);  // 截取中间区域为例
	subRegion(img.data, col, row, (int)img.step, channels, img3.data, (int)img3.step, channel, rect);
	
	imshow("img2", img2); //显示图像
	imshow("img3", img3); //显示图像
	
	waitKey(0); //等待键盘输入

	return 0;
}
