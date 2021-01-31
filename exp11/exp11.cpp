//����OpenCV�е�SIFT, SURF, ORB�����������ƥ��ķ�����
//����⵽���������ƥ���ϵ���п��ӻ�������Ƚϲ�ͬ������Ч�ʡ�Ч���ȡ�

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <time.h>
#include <iostream>

using namespace std;
using namespace cv;

void SIFT_detect_match(Mat& src1, Mat& src2) {

	//���ظ���ԭͼƬ
	Mat Src1 = src1.clone();
	Mat Src2 = src2.clone();

	//��ȡ�����㲢���� 
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	Ptr<SIFT> detector = SIFT::create();

	Mat descriptorMat1, descriptorMat2;
	detector->detectAndCompute(Src1, Mat(), keypoints1, descriptorMat1);
	detector->detectAndCompute(Src2, Mat(), keypoints2, descriptorMat2);

	//������������ͼƬ
	Mat dst1, dst2;
	drawKeypoints(Src1, keypoints1, dst1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(Src2, keypoints2, dst2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imwrite("SIFT_detect_src1.jpg", dst1);
	imwrite("SIFT_detect_src2.jpg", dst2);

	//������ƥ��
	cv::BFMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptorMat1, descriptorMat2, matches);

	//��ȡ�õ�ƥ������ɸѡ
	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptorMat1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "SIFT-- Max dist :" << max_dist << endl;
	cout << "SIFT-- Min dist :" << min_dist << endl;

	vector<DMatch> good_matches;
	for (int i = 0; i < descriptorMat1.rows; i++)
	{
		if (matches[i].distance < 0.5*max_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//��������ƥ��ͼ 
	Mat img_matches;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("SIFT_result.jpg", img_matches);
}

void SURF_detect_match(Mat& src1, Mat& src2) {
	//���ظ���ԭͼƬ
	Mat Src1 = src1.clone();
	Mat Src2 = src2.clone();

	//��ȡ�����㲢���� 
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create();

	Mat descriptorMat1, descriptorMat2;
	detector->detectAndCompute(Src1, Mat(), keypoints1, descriptorMat1);
	detector->detectAndCompute(Src2, Mat(), keypoints2, descriptorMat2);

	//������������ͼƬ
	Mat dst1, dst2;
	drawKeypoints(Src1, keypoints1, dst1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(Src2, keypoints2, dst2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imwrite("SURF_detect_src01.jpg", dst1);
	imwrite("SURF_detect_src02.jpg", dst2);

	//������ƥ��
	cv::BFMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptorMat1, descriptorMat2, matches);

	//��ȡ�õ�ƥ������ɸѡ
	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptorMat1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "SURF-- Max dist :" << max_dist << endl;
	cout << "SURF-- Min dist :" << min_dist << endl;

	vector< DMatch > good_matches;
	for (int i = 0; i < descriptorMat1.rows; i++)
	{
		if (matches[i].distance < 0.2*max_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//��������ƥ��ͼ 
	Mat img_matches;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("SURF_result.jpg", img_matches);
}

void ORB_detect_match(Mat& src1, Mat& src2) {
	//���ظ���ԭͼƬ
	Mat Src1 = src1.clone();
	Mat Src2 = src2.clone();

	//��ȡ�����㲢���� 
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	Ptr<ORB> detector = ORB::create();

	Mat descriptorMat1, descriptorMat2;
	detector->detectAndCompute(Src1, Mat(), keypoints1, descriptorMat1);
	detector->detectAndCompute(Src2, Mat(), keypoints2, descriptorMat2);

	//������������ͼƬ
	Mat dst1, dst2;
	drawKeypoints(Src1, keypoints1, dst1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(Src2, keypoints2, dst2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imwrite("ORB_detect_src01.jpg", dst1);
	imwrite("ORB_detect_src02.jpg", dst2);

	//������ƥ��
	cv::BFMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptorMat1, descriptorMat2, matches);

	//��ȡ�õ�ƥ������ɸѡ
	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptorMat1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "ORB-- Max dist :" << max_dist << endl;
	cout << "ORB-- Min dist :" << min_dist << endl;

	vector< DMatch > good_matches;
	for (int i = 0; i < descriptorMat1.rows; i++)
	{
		if (matches[i].distance < 0.5*max_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//��������ƥ��ͼ 
	Mat img_matches;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("ORB_result.jpg", img_matches);
}

int main() {
	Mat img1 = imread("exp9ai5.jpg", 1);
	Mat img2 = imread("exp9ai6.jpg", 1);
	clock_t tic, toc;

	tic = clock();  //��ʼʱ��
	SIFT_detect_match(img1, img2);
	toc = clock();   //����ʱ��
	cout << "SIFT����ʱ��Ϊ" << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;

	tic = clock();  //��ʼʱ��
	ORB_detect_match(img1, img2);
	toc = clock();   //����ʱ��
	cout << "ORB����ʱ��Ϊ" << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;

	tic = clock();  //��ʼʱ��
	SURF_detect_match(img1, img2);
	toc = clock();   //����ʱ��
	cout << "SURF����ʱ��Ϊ" << (double)(toc - tic) / CLOCKS_PER_SEC << "s" << endl;

	return 0;
}
