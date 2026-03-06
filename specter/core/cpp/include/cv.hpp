#pragma once

#include <string>
#include <vector>
#include <memory>


namespace cv {
    struct BaseOption {};

    struct IntOption final : public BaseOption {
        int value = 0;

        IntOption() = default;
        IntOption(IntOption &&) = default;
    };

    struct FloatOption final : public BaseOption {
        float value = 0.f;

        FloatOption() = default;
        FloatOption(FloatOption &&) = default;
    };

    struct BoolOption final : public BaseOption {
        bool value = false;

        BoolOption() = default;
        BoolOption(BoolOption &&) = default;
    };

    struct StringOption final : public BaseOption {
        std::string value;

        StringOption() = default;
        StringOption(StringOption &&) = default;
    };

    struct MultiStringOption final : public BaseOption {
        std::vector<std::string> values;
        size_t selected = 0;

        MultiStringOption() = default;
        MultiStringOption(MultiStringOption &&) = default;
    };

    struct MultiIntegerOption final : public BaseOption {
        std::vector<int> values;
        size_t selected = 0;

        MultiIntegerOption() = default;
        MultiIntegerOption(MultiIntegerOption &&) = default;
    };

    struct MultiFloatOption final : public BaseOption {
        std::vector<float> values;
        size_t selected = 0;

        MultiFloatOption() = default;
        MultiFloatOption(MultiFloatOption &&) = default;
    };


    struct IpToolDescription {
        std::string const name;
        std::vector<std::unique_ptr<BaseOption>> options;
    };

    template<typename T>
    inline std::shared_ptr<T> makePtr(T obj) {
        return std::make_shared<T>(std::move(obj));
    }

    inline std::shared_ptr<FloatOption> makeFloatPtr(FloatOption obj) {
        return makePtr<FloatOption>(std::move(obj));
    }

    inline std::shared_ptr<IntOption> makeIntPtr(IntOption obj) {
        return makePtr<IntOption>(std::move(obj));
    }
}
