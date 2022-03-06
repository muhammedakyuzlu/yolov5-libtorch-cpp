#include <iostream>
#include <memory>
#include <chrono>
#include <regex>

#include "detector.h"
#include "cxxopts.hpp"

std::vector<std::string> LoadNames(const std::string &path)
{
    // load class names
    std::vector<std::string> classes_name;
    std::ifstream infile(path);
    if (infile.is_open())
    {
        std::string line;
        while (getline(infile, line))
        {
            classes_name.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "Error loading the class names!\n";
    }
    return classes_name;
}

bool is_int(const std::string &s)
{
    return std::regex_match(s, std::regex("[0-9]"));
}

cv::Mat draw(cv::Mat &img,
          const std::vector<std::vector<Detection>> &detections,
          const std::vector<std::string> &classes_name,
          bool label = true
          )
{

    if (!detections.empty())
    {
        for (const auto &detection : detections[0])
        {
            const auto &box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label)
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = classes_name[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline = 0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                              cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                              cv::Point(box.tl().x + s_size.width, box.tl().y),
                              cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
            }
            
        }
        return img;
    }
    return img;
}

int main(int argc, const char *argv[])
{
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the yolov5");

    parser.allow_unrecognised_options().add_options()("weights", "path to model.torchscript.pt",
     cxxopts::value<std::string>())("classes", "path to classes name  coco.name",
      cxxopts::value<std::string>())
       ("i", "image or video/stream default:video",cxxopts::value<bool>()->default_value("false"))
       ("source", "path to source", cxxopts::value<std::string>())
       ("conf-thresh", "object confidence threshold", cxxopts::value<float>()->default_value("0.4"))
       ("iou-thresh", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))
       ("show", "display results", cxxopts::value<bool>()->default_value("false"))
       ("h,help", "Print usage");

    auto opt = parser.parse(argc, argv);

    if (opt.count("help"))
    {
        std::cout << parser.help() << std::endl;
        exit(0);
    }

    // load classes name
    std::string classes = opt["classes"].as<std::string>();

    // load classes names
    std::vector<std::string> classes_name = LoadNames(classes);
    if (classes_name.empty())
        return -1;

    // load network
    std::string weights = opt["weights"].as<std::string>();
    auto detector = Detector(weights, torch::kCPU);

    // set up threshold
    float conf_thresh = opt["conf-thresh"].as<float>();
    float iou_thresh = opt["iou-thresh"].as<float>();

    // load source
    std::string source = opt["source"].as<std::string>();

    // is image ?
    bool b_image = opt["i"].as<bool>();

    if (b_image)
    { // image

        cv::Mat img = cv::imread(source);
        if (img.empty())
        {
            std::cerr << "Error loading the image!\n";
            return -1;
        }

        // inference
        auto detections = detector.Run(img, conf_thresh, iou_thresh);

        // final
        cv::Mat result = draw(img, detections, classes_name,true);

        // save
         
        std::string save_path = source.substr(0, source.find("."));
        
        cv::imwrite("_r.jpg", result);

        // // visualize detections
        if (opt["show"].as<bool>()){
            // Display the resulting frame
            cv::imshow("result", result);
            int k = cv::waitKey(0); // Wait for a keystroke in the window
        }
    }
    else
    { // video

        cv::VideoCapture cap;

        // video or camera
        if (is_int(source))
        {
            cap.open(std::stoi(source));
        }
        else
        {
            cap.open(source);
        }

        // Default resolutions of the frame are obtained.The default resolutions are system dependent.
        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
        cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));

        while (true)
        {
            cv::Mat frame;
            // Capture frame-by-frame
            cap >> frame;
            // If the frame is empty, break immediately
            if (frame.empty())
                break;
        
            // inference
            auto detections = detector.Run(frame, conf_thresh, iou_thresh);
            // final
            cv::Mat result = draw(frame, detections, classes_name,true);
            // Write the frame into the file 'outcpp.avi'
            video.write(result);
            if (opt["show"].as<bool>()){
                // Display the resulting frame
                cv::imshow("result", result);
                // Press  ESC on keyboard to  exit
                char c = (char)cv::waitKey(1);
                if (c == 27)
                    break;
            }
        }
        // When everything done, release the video capture and write object
        cap.release();
        video.release();
    }

    cv::destroyAllWindows();
    return 0;
}


