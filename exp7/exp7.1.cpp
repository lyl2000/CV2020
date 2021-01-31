#include<opencv2\opencv.hpp>
#include<algorithm>
#include<iostream>
using namespace std;
using namespace cv;

void cv_connected_component(const Mat& image) {
	Mat labels, stats, centroids;
	Mat image1(image.size(), CV_8UC3);  // 随即涂色
	Mat image2(image.size(), CV_8UC3);  // 最大前景
	int label_num = connectedComponentsWithStats(image, labels, stats, centroids);
	cout << label_num << endl << stats << endl << centroids << endl << image.rows << " " << image.cols << endl;
	int *area = new int[label_num];
	Vec3b *color = new Vec3b[label_num];
	int idx = -1, max_area = 0;  // 最大前景面积
	for (int i = 0; i < label_num; ++i) {
		area[i] = stats.at<int>(i, CC_STAT_AREA);
		if (i > 0 && area[i] > max_area) {  // 0代表背景
			max_area = area[i];
			idx = i;
		}
		color[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}
	cout << max_area << endl;
	for (int y = 0; y < image.rows; ++y) {
		Vec3b *row1 = image1.ptr<Vec3b>(y);
		Vec3b *row2 = image2.ptr<Vec3b>(y);
		for (int x = 0; x < image.cols; ++x) {
			int label = labels.at<int>(y, x);
			row1[x] = color[label];
			if (label == 0 || label == idx) {  // 背景或最大前景
				row2[x] = color[label];
			}
			else {
				row2[x] = color[0];
			}
		}
	}
	imshow("image1", image1);
	imshow("image2", image2);
	delete[] area;
	delete[] color;
}


//图像一遍扫描  image：二值图像 0为背景,255为前景   image_label：记录每个像素所属连通域
int ScanImage(Mat image, int *image_label) {
	int row = image.rows, col = image.cols;
	int cnt = 1, label = 1;
	// 处理第一行
	uchar *r = image.ptr<uchar>(0);
	image_label[0] = (r[0] == 255 ? label : -1);
	for (int x = 1; x < col; ++x) {
		if (r[x] == r[x - 1]) {
			image_label[x] = image_label[x - 1];  // 背景
		}
		else {
			++cnt;
			image_label[x] = ++label;  // 属于新的连通域
		}
	}
	// 处理下边所有行
	for (int y = 1; y < row; ++y) {
		uchar *lastrow = image.ptr<uchar>(y - 1);  // 上一行指针
		uchar *thisrow = image.ptr<uchar>(y);  // 本行指针
		// 本行首元素
		if (thisrow[0] == lastrow[0]) {
			image_label[y*col] = image_label[(y - 1)*col];
		}
		else {
			++cnt;
			image_label[y*col] = ++label;
		}
		// 右边的元素
		for (int x = 1; x < col; ++x) {
			int up_label = image_label[(y - 1)*col + x];
			int left_label = image_label[y*col + x - 1];
			uchar now = thisrow[x], up = lastrow[x], left = thisrow[x - 1];
			if (now != left && now != up) {
				++cnt;
				image_label[y*col + x] = ++label;
			}
			else if (now == left && now != up) {
				image_label[y*col + x] = left_label;
			}
			else if (now != left && now == up) {
				image_label[y*col + x] = up_label;
			}
			else if (now == left && now == up) {
				if (up_label == left_label) {
					image_label[y*col + x] = up_label;
				}
				else {
					for (int k = 0; k < y*col + x; ++k) {
						if (image_label[k] == left_label) {
							image_label[k] = up_label;
						}
					}
					image_label[y*col + x] = up_label;
					--cnt;
				}
			}
		}
	}
	return cnt;
}

// 随即涂色
void PaintImage(const Mat& image, int *image_label) {
	Mat img(image.size(), CV_8UC3);
	for (int y = 0; y < image.rows; ++y) {
		Vec3b *p = img.ptr<Vec3b>(y);
		for (int x = 0; x < image.cols; ++x) {
			int label = image_label[y*image.cols + x];
			if (label == -1) {
				p[x] = Vec3b(0, 0, 0);
			}
			else {
				p[x] = Vec3b(label * 10 % 255, label * 20 % 255, label * 40 % 255);
			}
		}
	}
	imshow("img", img);
}

// 最大前景
void max_fore_area(const Mat& image, int *image_label, int label_num) {
	map<int, int> areas;
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			int label = image_label[y*image.cols + x];
			if (label == -1) {
				continue;
			}
			areas[label]++;
		}
	}
	int idx = -1, max_area = 0;
	for (auto area : areas) {
		if (area.second > max_area) {
			idx = area.first;
			max_area = area.second;
		}
	}
	Mat img(image.size(), CV_8UC3);
	for (int y = 0; y < image.rows; ++y) {
		Vec3b *p = img.ptr<Vec3b>(y);
		for (int x = 0; x < image.cols; ++x) {
			int label = image_label[y*image.cols + x];
			if (label == idx) {
				p[x] = Vec3b(label * 10 % 255, label * 20 % 255, label * 40 % 255);
			}
			else {
				p[x] = Vec3b(0, 0, 0);
			}
		}
	}
	imshow("max_area", img);
}

void my_connected_component(const Mat& image) {
	int *image_label = new int[image.rows * image.cols];
	int label_num = ScanImage(image, image_label);
	cout << label_num << endl;
	PaintImage(image, image_label);
	max_fore_area(image, image_label, label_num);
	delete[] image_label;
}

int main() {
	Mat image, labels, stats, centroids;
	image = imread("exp7a.jpg", 0);
	threshold(image, image, 0, 255, THRESH_OTSU);
	imshow("image", image);
	cv_connected_component(image);
	//my_connected_component(image);
	waitKey();
	return 0;
}
