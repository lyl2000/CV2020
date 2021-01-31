/*
�˽�cv::matchTemplate�������÷�����ѡ����ʵĲ���ͼ����в��ԣ�Ҫ��
���TM_SQDIFF�������Զ��������ĺ�������������
���ģ����ͼ��Ŀ�������ɫ�����ȣ����졢�����α��������в��Է�����
�����ص�Ա�TM_SQDIFF��TM_CCOEFF_NORMED���жԱȡ�

cv::TM_SQDIFF���÷���ʹ��ƽ�������ƥ�䣬�����ѵ�ƥ�����ڽ��Ϊ0����ֵԽ��ƥ����Խ�
cv::TM_SQDIFF_NORMED���÷���ʹ�ù�һ����ƽ�������ƥ�䣬���ƥ��Ҳ�ڽ��Ϊ0����
cv::TM_CCORR�������ƥ�䷽�����÷���ʹ��Դͼ����ģ��ͼ��ľ���������ƥ�䣬��ˣ����ƥ��λ����ֵ��󴦣�ֵԽСƥ����Խ�
cv::TM_CCORR_NORMED����һ���������ƥ�䷽�����������ƥ�䷽�����ƣ����ƥ��λ��Ҳ����ֵ��󴦡�
cv::TM_CCOEFF�������ϵ��ƥ�䷽�����÷���ʹ��Դͼ�������ֵ�Ĳģ�������ֵ�Ĳ����֮�������Խ���ƥ�䣬���ƥ������ֵ����1�������ƥ������ֵ����-1����ֵ����0ֱ�ӱ�ʾ���߲���ء�
cv::TM_CCOEFF_NORMED����һ���������ϵ��ƥ�䷽������ֵ��ʾƥ��Ľ���Ϻã���ֵ���ʾƥ���Ч���ϲҲ��ֵԽ��ƥ��Ч��Ҳ�á�
*/
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

Mat img, templ; 
const string image_window = "image";
int match_method;
int max_Trackbar = 5;

void trackbar(int, void*) {
	Mat img_display;
	img.copyTo(img_display);
	//�ȽϽ����ӳ��ͼ��, 32λ�����͵�ͨ��ͼ��,�ߴ�(W - w + 1)*(H - h + 1)
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	Mat result(result_rows, result_cols, CV_32FC1);
	
	cout << match_method << " ";
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	
	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) {
		matchLoc = minLoc; // ԽСԽ��
	}
	else {
		matchLoc = maxLoc;  // ������Խ��Խ��
	}
	cout << minVal << " " << maxVal << endl;
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);
	imshow(image_window, img_display);
	imshow("result", result);
	return;
}

int main() {
	img = imread("exp9ci1.jpg", IMREAD_COLOR);
	templ = imread("exp9ct.jpg", IMREAD_COLOR);

	namedWindow(image_window, WINDOW_AUTOSIZE);
	imshow("templ", templ);

	const string trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, trackbar);
	trackbar(0, 0);
	waitKey(0);
	return 0;
}
