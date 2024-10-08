// ----------------------- OpenPose C++ API Tutorial - Example 3 - Body from image -----------------------
// It reads an image, process it, and displays it with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <cmath>
#include <vector>
#include <Eigen>
#define M_PI (3.14159265358979323846264338327950288)

// Custom OpenPose flags
// Producer
/*DEFINE_string(image_path, "examples/media/COCO_val2014_000000000294.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");*/
DEFINE_string(image_path, "examples/media/bicep_curl_2p_1.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty())
            {
                cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
                cv::waitKey(0);
            }
            else
                op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
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

//std::vector<float> subtractvectors

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

        // Process and display image
        const cv::Mat cvImageToProcess = cv::imread(FLAGS_image_path);
        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
        if (datumProcessed != nullptr)
        {
            //find keypoints from image
            //this image should be the start of the bicep curl
            auto kpoints = datumProcessed->at(0)->poseKeypoints;
            int m = kpoints.getVolume();
            const int npoints = m / 3;
            std::vector<Eigen::Vector2f> allkpoints;
            std::vector<float> allconfidences;
            for (int i = 0; i < npoints; i++) {
                Eigen::Vector2f p(2);
                p << kpoints[3 * i], kpoints[3 * i + 1];
                //normalize p to zero to one.
                p = p / p.norm();
                allkpoints.push_back(p);
                allconfidences.push_back(kpoints[3*i + 2]);
            }
            //find elbow angle give three joint positions
            if (allconfidences[2] > 0.2 && allconfidences[3] > 0.2 && allconfidences[4] > 0.2) {
                Eigen::Vector2f v34 = allkpoints[4] - allkpoints[3];
                Eigen::Vector2f v32 = allkpoints[2] - allkpoints[3];
                float angle = acos(v34.dot(v32) / (v34.norm() * v32.norm())) * 180.0 / M_PI;
                if (angle > 25.0)
                {
                    std::cout << "arm angle too small";
                }
                std::cout << "angle:" << angle << "degrees";
            }
            else {
                std::cout << "no correct elbow 3 detected";
            }
            //the other elbow
            if (allconfidences[5] > 0.2 && allconfidences[6] > 0.2 && allconfidences[7] > 0.2) {
                Eigen::Vector2f v67 = allkpoints[7] - allkpoints[6];
                Eigen::Vector2f v65 = allkpoints[5] - allkpoints[6];
                float angle = acos(v67.dot(v65) / (v67.norm() * v65.norm())) * 180.0 / M_PI;
                if (angle < 25.0)
                {
                    std::cout << "arm angle too small";
                }
                std::cout << "angle:" << angle << "degrees";
            }
            else {
                std::cout << "no correct elbow 6 detected";
            }
            //the hip
            if (allconfidences[1] > 0.2 && allconfidences[8] > 0.2 && allconfidences[9] > 0.2 && allconfidences[10]>0.2) {
                Eigen::Vector2f v81 = allkpoints[1] - allkpoints[8];
                Eigen::Vector2f v910 = allkpoints[10] - allkpoints[9];
                float angle = acos(v81.dot(v910) / (v81.norm() * v910.norm())) * 180.0 / M_PI;
                if (angle < 150.0)
                {
                    std::cout << "hip angle too small";
                }
                std::cout << "angle:" << angle << "degrees";
            }
            else {
                std::cout << "no correct hip 9 detected";
            }
            if (allconfidences[1] > 0.2 && allconfidences[8] > 0.2 && allconfidences[11] > 0.2 && allconfidences[12] > 0.2) {
                Eigen::Vector2f v81 = allkpoints[1] - allkpoints[8];
                Eigen::Vector2f v1112 = allkpoints[12] - allkpoints[11];
                float angle = acos(v81.dot(v1112) / (v81.norm() * v1112.norm())) * 180.0 / M_PI;
                if (angle < 150.0)
                {
                    std::cout << "hip angle too small";
                }
                std::cout << "angle:" << angle << "degrees";
            }
            else {
                std::cout << "no correct hip 11 detected";
            }
            //find head angle given joint positions
            printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
                display(datumProcessed);
        }
        else
            op::opLog("Image could not be processed.", op::Priority::High);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
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
    return tutorialApiCpp();
}
