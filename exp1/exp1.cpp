#include <iostream>
#include <string>
#include <opencv2\opencv.hpp> //����OpenCV ͷ�ļ�

using namespace std;
using namespace cv; //opencv�������ռ�

uchar* get_pixel(const uchar *img, int x, int y, int step, int nc)
{
	// ��������ʵ����صĵ�ַ
	return (uchar*)img + y * step + x * nc;
}

/*
input, width, height, inStep, inChannels�ֱ�������ͼ������ݡ����ߡ�step��ͨ������
Output��outStep�����ͼ������ݺ�step�����������ͼ����ͬ��ͨ����Ϊ1.
channelToGet��Ҫ��ȡ��ͨ���������������input��BGR��ʽ�����ݣ�
��channelToGet=0����ȡBͨ��, channelToGet=1����ȡGͨ������
*/
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
	/*
	�������ΪRGBͼ���򽫸�ʽ��ΪCV_8UC3��
	���ʵ�ַ�޸�Ϊoutput[y*outStep+x*inChannels+channelToGet],
	�����get_pixel(output, x1, y1, outStep, inChannels)��ַ��
	*/

	// �൱����һ��ʵ��(3ͨ��RGB��ʽ)
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
	// ����getChannel
	getChannel(get_pixel(input, x1, y1, inStep, inChannels), rect.width, rect.height, inStep, inChannels, get_pixel(output, x1, y1, outStep, 1), outStep, channelToGet);
}

int main()
{
	string path = "C:/Users/20181014/Desktop/";
	string jpg = "jpg.jpg";
	string png = "png.png";
	string bmp = "bmp.bmp";

	// ʵ��1.1 ͼ�������ʾ
	Mat img; //����һ������ͼ�����
	img = imread(path + jpg); //��ȡͼ�񣬸���ͼƬ����λ����д·������ 3�ָ�ʽ��ͼƬ�ֱ���ʾ����
	if (img.empty()) //�ж�ͼ���ļ��Ƿ����
	{
		cout << "��ȷ��ͼ���ļ������Ƿ���ȷ" << endl;
		return -1;
	}
	imshow("img", img); //��ʾͼ��

	int row = img.rows, col = img.cols, channels = (int)img.channels(), channel = 0;
	cout << "��ͨ������" << channels << " ��ȡͨ������" << channel << endl;
	Mat img2(row, col, CV_8UC1);
	Mat img3(row, col, CV_8UC1);

	// ʵ��1.2 ͼ��ͨ������
	getChannel(img.data, col, row, (int)img.step, channels, img2.data, (int)img2.step, channel);
	
	// ʵ��1.3 ͼ����������� ����1.2
	cv::Rect rect(row / 3, col / 3, row / 3, col / 3);  // ��ȡ�м�����Ϊ��
	subRegion(img.data, col, row, (int)img.step, channels, img3.data, (int)img3.step, channel, rect);
	
	imshow("img2", img2); //��ʾͼ��
	imshow("img3", img3); //��ʾͼ��
	
	waitKey(0); //�ȴ���������

	return 0;
}
