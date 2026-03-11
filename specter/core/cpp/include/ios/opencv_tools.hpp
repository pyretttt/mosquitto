#pragma once

#include "cv.hpp"

namespace cv {
    namespace opencv {
        SingleFrameInput binarize(
            SingleFrameInput const &input,
            int threshold = 128
        );

        SingleFrameIpTool makeBinarizationTool(int threshold = 128);
    }
}
