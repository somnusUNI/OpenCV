#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <vector>
#include <ctime>

using namespace cv;

std::vector<Rect> detect_faces(Mat& gray, CascadeClassifier& cascade)
{
    std::vector<Rect> faces;
    cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    return faces;
}

void make_edge_frame(Mat& gray_frame, Mat& blurred_frame, Mat& edge_frame, int lower_threshold, int higher_threshold)
{
    GaussianBlur(gray_frame, blurred_frame, Size(3, 3), 3);
    Canny(blurred_frame, edge_frame, lower_threshold, higher_threshold);
}

void draw_faces(Mat& edge, const std::vector<Rect>& faces)
{
    for (size_t i = 0; i < faces.size(); i++) {
        Scalar color = Scalar(255, 0, 0);
        rectangle(edge, faces[i], color, 3);
    }
}

int main()
{   
    clock_t start;
    clock_t end;
    clock_t total_start;
    clock_t total_end;
    long double total_time = 0;
    long double total_input_time = 0;
    long double total_processing_time = 0;
    long double total_output_time = 0;
    size_t frames_counter = 0;

    VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt2.xml");

    total_start = clock();
    while (true) {
        start = clock();
        Mat input_frame;
        capture >> input_frame;
        if (input_frame.empty())
            break;
        end = clock();
        long double input_time = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds


        start = clock();
        Mat mirrored_frame;
        flip(input_frame, mirrored_frame, 1);
        Mat gray_frame, blurred_frame, edge_frame;
        cvtColor(mirrored_frame, gray_frame, COLOR_BGR2GRAY);
        Mat gray_frame_clone = gray_frame.clone();
        const std::vector<Rect>& faces = detect_faces(gray_frame, cascade);
        make_edge_frame(gray_frame, blurred_frame, edge_frame, 75, 100);
        draw_faces(edge_frame, faces);
        end = clock();
        long double processing_time = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds


        start = clock();
        imshow("edge and face detection", edge_frame);
        char c = static_cast<char>(waitKey(1));
        end = clock();
        long double output_time = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds
        
        ++frames_counter;
        total_input_time += input_time;
        total_processing_time += processing_time;
        total_output_time += output_time;
        
        if (c != -1)
            break;
    }
    total_end = clock();
    total_time = 1000.0 * (total_end - total_start) / CLOCKS_PER_SEC; // in milliseconds

    capture.release();

    cv::destroyAllWindows();

    
    std::cout << "Total time: " << total_time / 1000 << std::endl;
    std::cout << "Average FPS: " << frames_counter / (total_time / 1000) << std::endl;
    std::cout << "Percentage of total time spend on frame input: " << total_input_time / total_time * 100  << std::endl;
    std::cout << "Percentage of total time spend on frame processing: " << total_processing_time / total_time * 100 << std::endl;
    std::cout << "Percentage of total time spend on frame output: " << total_output_time / total_time * 100 << std::endl;

    return 0;
}
