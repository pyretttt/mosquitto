#pragma once

#include <cstdint>
#include <string>
#include <optional>

namespace cv {
    enum class OptionKind : std::uint8_t {
        intOption = 0,
        floatOption = 1,
        boolOption = 2,
        stringOption = 3,
    };

    struct SingleValueOption final {
        OptionKind kind;
        int intOption = 0;
        float floatOption = 0.f;
        bool boolOption = false;
        std::optional<std::string> stringOption = std::nullopt;

        static SingleValueOption makeInt(int value) {
            return SingleValueOption { .kind = OptionKind::intOption } ;
        }

        static SingleValueOption makeFloat(float value) {
            return SingleValueOption { .kind = OptionKind::floatOption } ;
        }

        static SingleValueOption makeBool(bool value) {
            return SingleValueOption { .kind = OptionKind::boolOption } ;
        }

        static SingleValueOption makeString(std::string value) {
            return SingleValueOption { .kind = OptionKind::stringOption, .stringOption = value } ;
        }
    };

    struct MultiValueOption final {
        OptionKind kind;
        std::optional<std::vector<std::string>> stringOptions = std::nullopt;
        std::optional<std::vector<int>> intOptions = std::nullopt;
        std::optional<std::vector<float>> floatOptions = std::nullopt;

        int selectedIndex = 0;

        static MultiValueOption makeString(std::vector<std::string> values) {
            return MultiValueOption { .kind = OptionKind::stringOption, .stringOptions = values } ;
        }

        static MultiValueOption makeInt(std::vector<int> values) {
            return MultiValueOption { .kind = OptionKind::intOption, .intOptions = values } ;
        }

        static MultiValueOption makeFloat(std::vector<float> values) {
            return MultiValueOption { .kind = OptionKind::floatOption, .floatOptions = values } ;
        }
    };
}
