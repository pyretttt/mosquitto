#pragma once

#include <string>
#include <vector>
#include "option.hpp"

namespace ip_tool {
    struct IpToolDescription {
        std::string const name;
        std::vector<cv::SingleValueOption> options;
        std::vector<cv::MultiValueOption> selectors;
    };
}
