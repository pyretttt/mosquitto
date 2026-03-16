#include "ios/opencv_tools.hpp"

#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv {
namespace opencv {

namespace {
int clampThreshold(int value) {
    return std::min(255, std::max(0, value));
}

SingleFrameInput makeOutputFromMatBGRA(const cv::Mat &mat) {
    if (mat.empty()) {
        return {};
    }

    CVPixelBufferRef outputBuffer = nullptr;
    const int width = mat.cols;
    const int height = mat.rows;
    const OSType pixelFormat = kCVPixelFormatType_32BGRA;

    CVReturn status = CVPixelBufferCreate(
        nullptr,
        width,
        height,
        pixelFormat,
        nullptr,
        &outputBuffer
    );

    if (status != kCVReturnSuccess || !outputBuffer) {
        return {};
    }

    CVPixelBufferLockBaseAddress(outputBuffer, 0);
    auto *base = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(outputBuffer));
    if (!base) {
        CVPixelBufferUnlockBaseAddress(outputBuffer, 0);
        CVPixelBufferRelease(outputBuffer);
        return {};
    }
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(outputBuffer);
    cv::Mat target(height, width, CV_8UC4, base, bytesPerRow);

    if (mat.type() == CV_8UC4) {
        mat.copyTo(target);
    } else if (mat.type() == CV_8UC1) {
        cv::cvtColor(mat, target, cv::COLOR_GRAY2BGRA);
    } else if (mat.type() == CV_8UC3) {
        cv::cvtColor(mat, target, cv::COLOR_BGR2BGRA);
    } else {
        cv::Mat converted;
        mat.convertTo(converted, CV_8UC4);
        converted.copyTo(target);
    }

    CVPixelBufferUnlockBaseAddress(outputBuffer, 0);

    SingleFrameInput output(outputBuffer);
    CVPixelBufferRelease(outputBuffer);
    return output;
}

inline const std::string thresholdOptionKey = "Threshold";
} // namespace

SingleFrameInput binarize(const SingleFrameInput &input, int threshold) {
    cv::Mat bgra = input.toMatBGRA();
    if (bgra.empty()) {
        return {};
    }

    const int clampedThreshold = clampThreshold(threshold);
    cv::Mat gray;
    cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);

    cv::Mat binary;
    cv::threshold(gray, binary, clampedThreshold, 255, cv::THRESH_BINARY);

    cv::Mat out;
    cv::cvtColor(binary, out, cv::COLOR_GRAY2BGRA);
    return makeOutputFromMatBGRA(out);
}

SingleFrameIpTool makeBinarizationTool() {
    auto options = std::vector<std::shared_ptr<BaseOption>>{
        makePtr(IntOption(thresholdOptionKey, 127))
    };
    std::unordered_map<std::string, std::shared_ptr<BaseOption>> optionsMap;
    for (auto const &opt : options) {
        optionsMap.emplace(opt->name, opt);
    }
    
    return SingleFrameIpTool{
        "Binarization",
        options,
        [optionsMap](SingleFrameInput input) {
            auto threshold = cv::asInt(optionsMap.at(thresholdOptionKey));
            return binarize(input, threshold->value);
        },
    };
}

} // namespace opencv
} // namespace cv
