// ORB+RANSEC
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Point startPoint, endPoint;
bool canDraw = false;
bool hasTarget = false;
Rect roi;  // 目标区域
Mat image, templ, copyImage, targetImage;

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
		rectangle(copyImage, roi, Scalar(0, 255, 0));
		imshow("frame", copyImage);
	}
	if (event == EVENT_LBUTTONUP)
	{
		rectangle(copyImage, roi, Scalar(0, 255, 0));
		imshow("frame", copyImage);
		targetImage = image(roi);
		imwrite("temp.jpg", targetImage);
		canDraw = false;
		hasTarget = true;
	}
}

void getTarget()
{
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "vedio open error" << endl;
		return;
	}
	int fps = 25, wait_time = 1000 / fps;
	while (true)
	{
		if (!canDraw) capture >> image;  // 先读入一帧
		if (hasTarget || waitKey(wait_time) == 27) break;
		imshow("frame", image);
	}
	capture.release();
}

int version1() {
	templ = imread("temp.jpg");
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "vedio open error" << endl;
		return 1;
	}
	int fps = 25, wait_time = 1000 / fps;
	const int MIN_MATCHES = 20;  // 点对数
	vector<Point2f> sce_corner(4), obj_corner(4);
	obj_corner[0] = Point2f(0, 0);
	obj_corner[1] = Point2f((float)templ.cols, 0);
	obj_corner[2] = Point2f((float)templ.cols, (float)templ.rows);
	obj_corner[3] = Point2f(0, (float)templ.rows);

	Ptr<SIFT> detector = SIFT::create(500);
	// Ptr<ORB> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
	vector<KeyPoint> sce_kp, obj_kp;
	Mat sce_des, obj_des;
	BFMatcher bf_matcher;
	vector<DMatch> matches;
	vector<DMatch> gd_matches;
	Mat img_matches;

	while (true) {
		if (waitKey(wait_time) == 27) break;
		cap >> image;

		clock_t tic = clock();

		detector->detectAndCompute(image, Mat(), sce_kp, sce_des);
		detector->detectAndCompute(templ, Mat(), obj_kp, obj_des);
		bf_matcher.match(sce_des, obj_des, matches);

		/*flann::Index flannIndex(obj_kp, flann::LshIndexParams(20, 10, 2), cvflann::FLANN_DIST_HAMMING);
		Mat matchIndex(sce_des.rows, 2, CV_32SC1), matchDistance(sce_des.rows, 2, CV_32FC1);
		flannIndex.knnSearch(sce_des, matchIndex, matchDistance, 2, flann::SearchParams());
		for (int i = 0; i < matchDistance.rows; i++) {
			if (matchDistance.at<float>(i, 0) < 0.5 * matchDistance.at<float>(i, 1)) {
				DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
				gd_matches.push_back(dmatches);
			}
		}*/

		sort(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {
			return m1.distance < m2.distance;
		});

		gd_matches = vector<DMatch>{ matches.begin(), matches.begin() + min(MIN_MATCHES, (int)matches.size()) };

		drawMatches(image, sce_kp, templ, obj_kp, gd_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		
		vector<Point2f> sce_pts, obj_pts;
		for (int i = 0; i < (int)gd_matches.size(); ++i) {
			sce_pts.push_back(sce_kp[gd_matches[i].queryIdx].pt);
			obj_pts.push_back(obj_kp[gd_matches[i].trainIdx].pt);
		}
		Mat H = findHomography(obj_pts, sce_pts, RANSAC);
		perspectiveTransform(obj_corner, sce_corner, H);
		for (int i = 0; i < 4; ++i) {
			line(img_matches, sce_corner[i], sce_corner[(i + 1) % 4], Scalar(0, 255, 0), 2);
		}

		clock_t toc = clock();

		string text = to_string(CLOCKS_PER_SEC / (double)(toc - tic));
		putText(img_matches, text, Point(50, 50), HersheyFonts::FONT_ITALIC, 1.0, Scalar::all(-1));
		imshow("frame", img_matches);
	}
	cap.release();
	return 0;
}

int version2() {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "vedio open error" << endl;
		return 1;
	}
	templ = imread("temp.jpg");
	Mat gray/*灰度图*/, prevGray/*上一帧灰度图*/;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subpixWinSize(10, 10), winSize(31, 31);
	vector<Point2f> points[2];
	int fps = 25, wait_time = 1000 / fps;
	const int MAX_COUNT = 50;
	bool init = true;
	cvtColor(templ, templ, COLOR_BGR2GRAY);
	goodFeaturesToTrack(templ, points[1], MAX_COUNT, 0.01, 10);  //角点检测
	cornerSubPix(templ, points[1], subpixWinSize, Size(-1, -1), termcrit);  // 亚像素角点检测
	for (size_t i = 0; i < points[1].size(); ++i) {
		points[1][i].x += startPoint.x;
		points[1][i].y += startPoint.y;
	}
	vector<Point2f> sce_corner(4), obj_corner(4);
	sce_corner[0] = Point2f(0 + startPoint.x, 0 + startPoint.y);
	sce_corner[1] = Point2f((float)templ.cols + startPoint.x, 0 + startPoint.y);
	sce_corner[2] = Point2f((float)templ.cols + startPoint.x, (float)templ.rows + startPoint.y);
	sce_corner[3] = Point2f(0 + startPoint.x, (float)templ.rows + startPoint.y);
	while (true) {
		if (waitKey(wait_time) == 27) break;
		cap >> image;
		clock_t tic = clock();
		cvtColor(image, gray, COLOR_BGR2GRAY);
		if (!points[0].empty()) {
			vector<uchar> status;
			vector<float> err;
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
			size_t k = 0;
			for (size_t i = 0; i < points[1].size(); ++i) {
				if (status[i] && norm(points[0][i] - points[1][i]) > 0) {
					points[1][k] = points[1][i];
					points[0][k] = points[0][i];
					++k;
					circle(image, points[1][i], 3, Scalar::all(-1));
				}
			}
			if (k <= 5) break;
			points[1].resize(k);
			points[0].resize(k);
			for (size_t i = 0; i < points[1].size(); ++i) {
				circle(image, points[1][i], 3, Scalar(0, 255, 0));
			}
			Mat H = findHomography(points[0], points[1], RANSAC);
			perspectiveTransform(obj_corner, sce_corner, H);
			for (int i = 0; i < 4; ++i) {
				line(image, obj_corner[i], obj_corner[(i + 1) % 4], Scalar(0, 0, 255), 2);
			}
			for (int i = 0; i < 4; ++i) {
				line(image, sce_corner[i], sce_corner[(i + 1) % 4], Scalar(0, 255, 0), 2);
			}
		}
		clock_t toc = clock();

		string text = to_string(CLOCKS_PER_SEC / (double)(toc - tic));
		putText(image, text, Point(50, 50), HersheyFonts::FONT_HERSHEY_COMPLEX, 1.0, Scalar::all(-1));
		imshow("frame", image);
		swap(obj_corner, sce_corner);
		swap(gray, prevGray);
		swap(points[0], points[1]);
	}
	cap.release();
	return 0;
}

int version3() {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "vedio open error" << endl;
		return 1;
	}
	templ = imread("temp.jpg");
	Mat gray/*灰度图*/, prevGray/*上一帧灰度图*/;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subpixWinSize(10, 10), winSize(31, 31);
	Ptr<SIFT> detector = SIFT::create(500);
	vector<Point2f> points[2];
	vector<KeyPoint> sce_kp, obj_kp;
	Mat sce_des, obj_des;
	BFMatcher bf_matcher;
	vector<DMatch> matches;
	vector<DMatch> gd_matches;
	
	int fps = 25, wait_time = 1000 / fps;
	const int MIN_MATCHES = 20;
	const int MAX_COUNT = 50;
	bool init = true;

	//cvtColor(templ, templ, COLOR_BGR2GRAY);
	//goodFeaturesToTrack(templ, points[1], MAX_COUNT, 0.01, 10);  //角点检测
	//cornerSubPix(templ, points[1], subpixWinSize, Size(-1, -1), termcrit);  // 亚像素角点检测
	//for (size_t i = 0; i < points[1].size(); ++i) {
	//	points[1][i].x += startPoint.x;
	//	points[1][i].y += startPoint.y;
	//}
	vector<Point2f> sce_corner(4), obj_corner(4);
	/*sce_corner[0] = Point2f(0 + startPoint.x, 0 + startPoint.y);
	sce_corner[1] = Point2f((float)templ.cols + startPoint.x, 0 + startPoint.y);
	sce_corner[2] = Point2f((float)templ.cols + startPoint.x, (float)templ.rows + startPoint.y);
	sce_corner[3] = Point2f(0 + startPoint.x, (float)templ.rows + startPoint.y);
	*/
	int timer = 0;
	while (true) {
		if (waitKey(wait_time) == 27) break;
		cap >> image;
		clock_t tic = clock();
		cout << timer << endl;
		if (timer++ == 50) {
			points[1].clear();
			timer = 0;
		}
		if (points[1].empty()) {
			obj_corner[0] = Point2f(0, 0);
			obj_corner[1] = Point2f((float)templ.cols, 0);
			obj_corner[2] = Point2f((float)templ.cols, (float)templ.rows);
			obj_corner[3] = Point2f(0, (float)templ.rows);

			detector->detectAndCompute(image, Mat(), sce_kp, sce_des);
			detector->detectAndCompute(templ, Mat(), obj_kp, obj_des);
			bf_matcher.match(sce_des, obj_des, matches);
			sort(matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {
				return m1.distance < m2.distance;
			});
			gd_matches = vector<DMatch>{ matches.begin(), matches.begin() + min(MIN_MATCHES, (int)matches.size()) };
			points[0].clear();
			for (int i = 0; i < (int)gd_matches.size(); ++i) {
				points[1].push_back(sce_kp[gd_matches[i].queryIdx].pt);
				points[0].push_back(obj_kp[gd_matches[i].trainIdx].pt);
			}
			Mat H = findHomography(points[0], points[1], RANSAC);
			perspectiveTransform(obj_corner, sce_corner, H);
			for (int i = 0; i < 4; ++i) {
				line(image, sce_corner[i], sce_corner[(i + 1) % 4], Scalar(0, 255, 0), 2);
			}
			cvtColor(image, gray, COLOR_BGR2GRAY);
		}
		else if (!points[0].empty()) {
			cvtColor(image, gray, COLOR_BGR2GRAY);
			vector<uchar> status;
			vector<float> err;
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
			size_t k = 0;
			for (size_t i = 0; i < points[1].size(); ++i) {
				if (status[i] && norm(points[0][i] - points[1][i]) > 0) {
					points[1][k] = points[1][i];
					points[0][k] = points[0][i];
					++k;
				}
			}
			if (k <= 5) break;
			points[1].resize(k);
			points[0].resize(k);
			for (size_t i = 0; i < points[1].size(); ++i) {
				circle(image, points[1][i], 3, Scalar(0, 255, 0));
			}
			Mat H = findHomography(points[0], points[1], RANSAC);
			perspectiveTransform(obj_corner, sce_corner, H);
			for (int i = 0; i < 4; ++i) {
				line(image, obj_corner[i], obj_corner[(i + 1) % 4], Scalar(0, 0, 255), 2);
			}
			for (int i = 0; i < 4; ++i) {
				line(image, sce_corner[i], sce_corner[(i + 1) % 4], Scalar(0, 255, 0), 2);
			}
		}
		clock_t toc = clock();
		
		string text = to_string(CLOCKS_PER_SEC / (double)(toc - tic));
		putText(image, text, Point(50, 50), HersheyFonts::FONT_HERSHEY_COMPLEX, 1.0, Scalar::all(-1));
		imshow("frame", image);
		swap(obj_corner, sce_corner);
		swap(gray, prevGray);
		swap(points[0], points[1]);
	}
	cap.release();
	return 0;
}

int main() {
	namedWindow("frame");
	setMouseCallback("frame", onMouse);
	getTarget();
	if (hasTarget)
		version3();
	return 0;
}
