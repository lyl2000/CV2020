/*
了解cv::matchTemplate函数的用法，并选择合适的测试图像进行测试，要求：
理解TM_SQDIFF等相似性度量方法的含义和适用情况。
针对模板与图像目标存在颜色（亮度）差异、几何形变等情况进行测试分析，
可以重点对比TM_SQDIFF和TM_CCOEFF_NORMED进行对比。

cv::TM_SQDIFF：该方法使用平方差进行匹配，因此最佳的匹配结果在结果为0处，值越大匹配结果越差。
cv::TM_SQDIFF_NORMED：该方法使用归一化的平方差进行匹配，最佳匹配也在结果为0处。
cv::TM_CCORR：相关性匹配方法，该方法使用源图像与模板图像的卷积结果进行匹配，因此，最佳匹配位置在值最大处，值越小匹配结果越差。
cv::TM_CCORR_NORMED：归一化的相关性匹配方法，与相关性匹配方法类似，最佳匹配位置也是在值最大处。
cv::TM_CCOEFF：相关性系数匹配方法，该方法使用源图像与其均值的差、模板与其均值的差二者之间的相关性进行匹配，最佳匹配结果在值等于1处，最差匹配结果在值等于-1处，值等于0直接表示二者不相关。
cv::TM_CCOEFF_NORMED：归一化的相关性系数匹配方法，正值表示匹配的结果较好，负值则表示匹配的效果较差，也是值越大，匹配效果也好。
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
	//比较结果的映射图像, 32位浮点型单通道图像,尺寸(W - w + 1)*(H - h + 1)
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
		matchLoc = minLoc; // 越小越好
	}
	else {
		matchLoc = maxLoc;  // 其他的越大越好
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
