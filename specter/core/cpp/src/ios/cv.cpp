#include "ios/cv.hpp"

#include <opencv2/core.hpp>

namespace cv {

cv::Mat SingleFrameInput::toMatBGRA() const {
    if (!imageBuffer) {
        return {};
    }
    CVPixelBufferRef pixelBuffer = reinterpret_cast<CVPixelBufferRef>(imageBuffer);
    if (CVPixelBufferGetPixelFormatType(pixelBuffer) != kCVPixelFormatType_32BGRA) {
        return {};
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    auto *base = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(pixelBuffer));
    if (!base) {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        return {};
    }
    int width = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
    int height = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
    cv::Mat mat(height, width, CV_8UC4, base, bytesPerRow);
    cv::Mat copy = mat.clone();

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return copy;
}

} // namespace cv
