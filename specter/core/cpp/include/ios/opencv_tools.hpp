#pragma once

#include "cv.hpp"

namespace cv {
    namespace opencv {
        // Pure
        SingleFrameInput binarize(
            SingleFrameInput const &input,
            int threshold = 128
        );

        // Tools
        SingleFrameIpTool makeBinarizationTool(int threshold = 128);
    }
}
