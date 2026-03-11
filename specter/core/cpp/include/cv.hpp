#pragma once

#include <functional>
#include <string>
#include <vector>
#include <memory>


namespace cv {
    struct BaseOption {};

    struct IntOption final : public BaseOption {
        int value = 0;

        IntOption() = default;
        IntOption(IntOption &&) = default;
        IntOption(int value) {
            this->value = value;
        };
    };

    struct FloatOption final : public BaseOption {
        float value = 0.f;

        FloatOption() = default;
        FloatOption(FloatOption &&) = default;
        FloatOption(float value) {
            this->value = value;
        };
    };

    struct BoolOption final : public BaseOption {
        bool value = false;

        BoolOption() = default;
        BoolOption(BoolOption &&) = default;
        BoolOption(bool value) {
            this->value = value;
        };

    };

    struct StringOption final : public BaseOption {
        std::string value;

        StringOption() = default;
        StringOption(StringOption &&) = default;
        StringOption(std::string value) {
            this->value = std::move(value);
        };
    };

    struct MultiStringOption final : public BaseOption {
        std::vector<std::string> values;
        size_t selected = 0;

        MultiStringOption() = default;
        MultiStringOption(MultiStringOption &&) = default;
        MultiStringOption(
            std::vector<std::string> values,
            size_t selected = 0
        ) {
            this->values = std::move(values);
            this->selected = selected;
        }
    };

    struct MultiIntegerOption final : public BaseOption {
        std::vector<int> values;
        size_t selected = 0;

        MultiIntegerOption() = default;
        MultiIntegerOption(MultiIntegerOption &&) = default;
        MultiIntegerOption(
            std::vector<int> values,
            size_t selected = 0
        ) {
            this->values = std::move(values);
            this->selected = selected;
        }
    };

    struct MultiFloatOption final : public BaseOption {
        std::vector<float> values;
        size_t selected = 0;

        MultiFloatOption() = default;
        MultiFloatOption(MultiFloatOption &&) = default;
        MultiFloatOption(
            std::vector<float> values,
            size_t selected = 0
        ) {
            this->values = std::move(values);
            this->selected = selected;
        }
    };

    inline std::shared_ptr<BaseOption> makePtr(IntOption obj) {
        return std::make_shared<IntOption>(std::move(obj));
    }

    inline std::shared_ptr<BaseOption> makePtr(FloatOption obj) {
        return std::make_shared<FloatOption>(std::move(obj));
    }

    inline std::shared_ptr<BaseOption> makePtr(BoolOption obj) {
        return std::make_shared<BoolOption>(std::move(obj));
    }

    inline std::shared_ptr<BaseOption> makePtr(StringOption obj) {
        return std::make_shared<StringOption>(std::move(obj));
    }

    inline std::shared_ptr<BaseOption> makePtr(MultiStringOption obj) {
        return std::make_shared<MultiStringOption>(std::move(obj));
    }

    inline std::shared_ptr<BaseOption> makePtr(MultiIntegerOption obj) {
        return std::make_shared<MultiIntegerOption>(std::move(obj));
    }

    inline std::shared_ptr<BaseOption> makePtr(MultiFloatOption obj) {
        return std::make_shared<MultiFloatOption>(std::move(obj));
    }

    struct SingleFrameInput {

    };

    template<typename Input, typename Output>
    struct IpTool {
        std::string const name;
        std::vector<std::shared_ptr<BaseOption>> options;
        std::function<Output (Input)> process;
    };

    using SingleFrameIpTool = IpTool<SingleFrameInput, SingleFrameInput>;

    
}
