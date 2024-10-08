// ----------------------- OpenPose C++ API Tutorial - Example 4 - Body from images ----------------------
// It reads images, process them, and display them with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <iostream>
#include <openpose/headers.hpp>
#include <Eigen>
#include <boost/format.hpp>
#include <dirent.h>
#include <sys/types.h>
#define M_PI (3.14159265358979323846264338327950288)

// Custom OpenPose flags
// Producer
DEFINE_string(image_dir,                "examples/media/bicep_correct_3",
    "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");
DEFINE_string(store_path, "C:/Users/ziywa/openpose/output/bicep_1p_2.jpg", "store");
DEFINE_string(store_path2, "C:/Users/ziywa/openpose/output/bicep_1p_3.jpg", "store");

// This worker will just read and return all the jpg files in a directory
int getdir(std::string dir, std::vector<std::string>& files, const char* prefix)
{
    DIR* dp;
    struct dirent* dirp;

    if ((dp = opendir(dir.c_str())) == NULL)
    {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
        if (strncmp(dirp->d_name, prefix, strlen(prefix)) == 0) {
            files.push_back(std::string(dirp->d_name));
        }
    }
    closedir(dp);
    return 0;
}

bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, int count)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty()) {
                cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
                const auto npeople = datumsPtr->at(0)->poseKeypoints.getSize(0);
                std::string dir = std::string("C:/Users/ziywa/openpose/output");
                std::vector<std::string> files;
                std::vector<int> filen;
                int mxn;
                const char* prefix = "bicep_";
                getdir(dir, files, prefix);
                for (int i = 0; i < files.size(); i++) {
                    filen.push_back(files[i][9] - '0');
                }
                if (filen.empty()) {
                    mxn = 0;
                }
                else {
                    mxn = *max_element(filen.begin(), filen.end());
                }
                std::string store_path = str(boost::format("C:/Users/ziywa/openpose/output/bicep_%2%p_%1%.jpg") % std::to_string(mxn+1) % std::to_string(npeople));
                //std::string store_path2 = str(boost::format("C:/Users/ziywa/openpose/output/bicep_%2%p_%1%.jpg") % std::to_string(mxn+2) % std::to_string(npeople));
                //if (count == 0)
                //{
                cv::imwrite(store_path, cvMat);
                //}
                //else {
                    //FLAGS_store_path2
                //    cv::imwrite(store_path2, cvMat);
                //}
            }
            else {
                op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        else {
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
        }
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return true;
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
            // op::opLog("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            // op::opLog("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
            // op::opLog("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

static std::string printMatrix(std::vector<Eigen::Vector2f> vec, std::vector<float> conf) {
    std::string s;
    for (int i = 0; i < vec.size(); i ++) {
        std::stringstream ss;
        ss << vec[i].transpose();
        s.append(ss.str());
        s.append(" ");
        s.append(std::to_string(conf[i]));
        s.append("\n");
    }
    return s;
}

void printNormalizedKeypoints(std::vector<Eigen::Vector2f> nkpoints, std::vector<float> conf) {
    if (nkpoints.empty()) {
        op::opLog("No Keypoints.", op::Priority::High);
    }
    else {
        op::opLog("Normalized Body keypoints:\n" + printMatrix(nkpoints, conf), op::Priority::High);
        //op::opLog("Face keypoints: " + nkpoints, op::Priority::High);
        //op::opLog("Left hand keypoints: " + nkpoints, op::Priority::High);
        //op::opLog("Right hand keypoints: " + nkpoints, op::Priority::High);
    }
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // Read frames on directory
        const auto imagePaths = op::getFilesOnDirectory(FLAGS_image_dir, op::Extensions::Images);

        // Process and display images
        // Supposed to input two images on two positions of bicep curl
        std::vector<Eigen::Vector2f> upperarmone;
        std::vector<Eigen::Vector2f> upperarmtwo;
        int count = 0;
        int elbowone = 1;
        int elbowtwo = 1;
        for (const auto& imagePath : imagePaths)
        {
            if (count > 1)
            {
                break;
            }
            const cv::Mat cvImageToProcess = cv::imread(imagePath);
            float width = cvImageToProcess.cols;
            float height = cvImageToProcess.rows;
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            //printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
            {
                const auto userWantsToExit = display(datumProcessed, count);
                if (userWantsToExit)
                {
                    op::opLog("User pressed Esc to exit demo.", op::Priority::High);
                    break;
                }
            }
            if (datumProcessed != nullptr)
            {
                    //find keypoints from image
                    //this image should be the start of the bicep curl
                auto kpoints = datumProcessed->at(0)->poseKeypoints;
                int m = kpoints.getVolume();
                if (m < 75) { 
                    std::cout << "not enough key points detected" << std::endl;
                    throw std::exception("25");
                    }
                const int npoints = m / 3;
                std::vector<Eigen::Vector2f> allkpoints;
                std::vector<float> allconfidences;
                for (int i = 0; i < 25; i++) {
                    Eigen::Vector2f p(2);
                    //normalize data by dividing width and height(did not do it here)
                    p << kpoints[3 * i] / width, kpoints[3 * i + 1] / height;
                    allkpoints.push_back(p);
                    allconfidences.push_back(kpoints[3 * i + 2]);
                }
                printNormalizedKeypoints(allkpoints, allconfidences);
                //find elbow angle give three joint positions[minimize the reaction force]
                if (allconfidences[2] > 0.25 && allconfidences[3] > 0.25 && allconfidences[4] > 0.25) {
                    Eigen::Vector2f v34 = allkpoints[4] - allkpoints[3];
                    Eigen::Vector2f v32 = allkpoints[2] - allkpoints[3];
                    float angle = acos(v34.dot(v32) / (v34.norm() * v32.norm())) * 180.0 / M_PI;
                    /*if (angle < 25.0)
                    {
                        std::cout << "arm angle too small" << std::endl;
                        return 0;
                    }*/
                    std::cout << "arm angle for elbow one:" << angle << "degrees" << std::endl;
                }
                else {
                    std::cout << "warning: no correct elbow one detected" << std::endl;
                    elbowone = 0;
                }
                //the other elbow
                if (allconfidences[5] > 0.25 && allconfidences[6] > 0.25 && allconfidences[7] > 0.25) {
                    Eigen::Vector2f v67 = allkpoints[7] - allkpoints[6];
                    Eigen::Vector2f v65 = allkpoints[5] - allkpoints[6];
                    float angle = acos(v67.dot(v65) / (v67.norm() * v65.norm())) * 180.0 / M_PI;
                    /*if (angle < 25.0)
                    {
                        std::cout << "arm angle too small" << std::endl;
                        return 0;
                    }*/
                    std::cout << "arm angle for elbow two:" << angle << "degrees" << std::endl;
                }
                else {
                    std::cout << "warning: no correct elbow two detected" << std::endl;
                    elbowtwo = 0;
                    if (elbowone == 0) {
                        std::cout << "Error: No elbows are detected.";
                        throw std::exception("20");
                    }
                }
                //upper arm vector
                if (elbowone != 0) {
                    if (allconfidences[2] > 0.25 && allconfidences[3] > 0.25) {
                        Eigen::Vector2f diff = allkpoints[3] - allkpoints[2];
                        upperarmone.push_back(diff);
                    }
                }
                if (elbowtwo != 0) {
                    if (allconfidences[5] > 0.25 && allconfidences[6] > 0.25) {
                        Eigen::Vector2f diff = allkpoints[6] - allkpoints[5];
                        upperarmtwo.push_back(diff);
                    }
                }
                count++;
            }
            else
                op::opLog("Image " + imagePath + " could not be processed.", op::Priority::High);
        }
        if (elbowone != 0) {
            Eigen::Vector2f uao1 = upperarmone[1];
            Eigen::Vector2f uao0 = upperarmone[0];
            float angle1 = acos((uao1.dot(uao0)) / (uao1.norm() * uao0.norm())) * 180.0 / M_PI;
            std::cout << "Upper arm angle one:" << angle1 << "degrees" << std::endl;
            if (angle1 > 15.0) {
                std::cout << "Your upper arm moved" << std::endl;
                return 0;
            }
        }
        if (elbowtwo != 0) {
            Eigen::Vector2f uat1 = upperarmtwo[1];
            Eigen::Vector2f uat0 = upperarmtwo[0];
            float angle2 = acos((uat1.dot(uat0)) / (uat1.norm() * uat0.norm())) * 180.0 / M_PI;
            std::cout << "Upper arm angle two：" << angle2 << "degrees" << std::endl;
            if (angle2 > 15.0) {
                std::cout << "Your upper arm moved" << std::endl;
                return 0;
            }
        }
        // Measuring total time
        op::printTime(opTimer, "Demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 1;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    int x = tutorialApiCpp();
    if (x == 0) {
        std::cout << "False position. You would need to change your arm positions.";
    }
    else if (x == -1) {
        std::cout << "not all keypoints are detected / not enough confidence for keypoints";
    }
    else {
        std::cout << "Correct position. You do not need to change your arm positions.";
    }
    return 0;

}
