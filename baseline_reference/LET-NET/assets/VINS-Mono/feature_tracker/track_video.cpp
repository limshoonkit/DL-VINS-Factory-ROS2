#include "src/feature_tracker.h"
#include "src/parameters.h"
#define IMG_H 240
#define IMG_W 320

int main() {
    const std::string video_path = "/home/c211/lyc/ncnn/alexnet_demo/demo/img/IMG_0378.MP4";
    cv::VideoCapture capture;
    capture.open(video_path);
    if (!capture.isOpened()) {
        std::cout << "Error opening video file !" << std::endl;
        return -1;
    }

    cv::namedWindow("video", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    int cnt = 0;
    while (true) {
        capture >> frame;
        if (frame.empty()) {
            std::cout << "Video end !" << std::endl;
            break;
        }
        cv::resize(frame, frame, cv::Size(IMG_W, IMG_H));
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // track feature
        FeatureTracker trackerData;
        trackerData.readImage(frame, (cnt + 1) * 0.1);
        cnt ++;
        cv::imshow("video", frame);
        cv::waitKey(5);
    }
}
