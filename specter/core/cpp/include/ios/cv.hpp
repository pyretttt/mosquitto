#pragma once

#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#include <CoreVideo/CoreVideo.h>


namespace cv {
    class Mat;

    enum class OptionKind {
        Int, Float, Bool, String, MultiString, MultiInteger, MultiFloat
    };

    struct BaseOption {
        virtual OptionKind kind() const noexcept;
        virtual ~BaseOption() = default;
    };

    struct IntOption final : public BaseOption {
        int value = 0;

        IntOption() = default;
        IntOption(IntOption &&) = default;
        IntOption(int value) {
            this->value = value;
        };

        OptionKind kind() const noexcept override { return OptionKind::Int; }
    };

    struct FloatOption final : public BaseOption {
        float value = 0.f;

        FloatOption() = default;
        FloatOption(FloatOption &&) = default;
        FloatOption(float value) {
            this->value = value;
        };

        OptionKind kind() const noexcept override { return OptionKind::Float; }
    };

    struct BoolOption final : public BaseOption {
        bool value = false;

        BoolOption() = default;
        BoolOption(BoolOption &&) = default;
        BoolOption(bool value) {
            this->value = value;
        };
        OptionKind kind() const noexcept override { return OptionKind::Bool; }
    };

    struct StringOption final : public BaseOption {
        std::string value;

        StringOption() = default;
        StringOption(StringOption &&) = default;
        StringOption(std::string value) {
            this->value = std::move(value);
        };
        OptionKind kind() const noexcept override { return OptionKind::String; }
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
        OptionKind kind() const noexcept override { return OptionKind::MultiString; }
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
        OptionKind kind() const noexcept override { return OptionKind::MultiInteger; }
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
        OptionKind kind() const noexcept override { return OptionKind::MultiFloat; }
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

    inline std::shared_ptr<IntOption> asInt(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<IntOption>>(std::move(ptr));
    }

    inline std::shared_ptr<FloatOption> asFloat(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<FloatOption>>(std::move(ptr));
    }

    inline std::shared_ptr<BoolOption> asBool(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<BoolOption>>(std::move(ptr));
    }

    inline std::shared_ptr<StringOption> asString(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<StringOption>>(std::move(ptr));
    }

    inline std::shared_ptr<MultiStringOption> asMultiString(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<MultiStringOption>>(std::move(ptr));
    }

    inline std::shared_ptr<MultiIntegerOption> asMultiInteger(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<MultiIntegerOption>>(std::move(ptr));
    }

    inline std::shared_ptr<MultiFloatOption> asMultiFloat(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<std::shared_ptr<MultiFloatOption>>(std::move(ptr));
    }

    struct SingleFrameInput {
        CVImageBufferRef imageBuffer = nullptr;

        SingleFrameInput() = default;
        explicit SingleFrameInput(CVImageBufferRef buffer) : imageBuffer(buffer) {
            if (imageBuffer) {
                CVPixelBufferRetain(reinterpret_cast<CVPixelBufferRef>(imageBuffer));
            }
        }

        SingleFrameInput(const SingleFrameInput &other)
            : imageBuffer(other.imageBuffer) {
            if (imageBuffer) {
                CVPixelBufferRetain(reinterpret_cast<CVPixelBufferRef>(imageBuffer));
            }
        }

        SingleFrameInput &operator=(const SingleFrameInput &other) {
            if (this == &other) {
                return *this;
            }
            if (imageBuffer) {
                CVPixelBufferRelease(reinterpret_cast<CVPixelBufferRef>(imageBuffer));
            }
            imageBuffer = other.imageBuffer;
            if (imageBuffer) {
                CVPixelBufferRetain(reinterpret_cast<CVPixelBufferRef>(imageBuffer));
            }
            return *this;
        }

        SingleFrameInput(SingleFrameInput &&other) noexcept
            : imageBuffer(other.imageBuffer) {
            other.imageBuffer = nullptr;
        }

        SingleFrameInput &operator=(SingleFrameInput &&other) noexcept {
            if (this == &other) {
                return *this;
            }
            if (imageBuffer) {
                CVPixelBufferRelease(reinterpret_cast<CVPixelBufferRef>(imageBuffer));
            }
            imageBuffer = other.imageBuffer;
            other.imageBuffer = nullptr;
            return *this;
        }

        ~SingleFrameInput() {
            if (imageBuffer) {
                CVPixelBufferRelease(reinterpret_cast<CVPixelBufferRef>(imageBuffer));
            }
        }

        CVImageBufferRef buffer() const {
            return imageBuffer;
        }

        cv::Mat toMatBGRA() const;
    };

    template<typename Input, typename Output>
    struct IpTool {
        std::string const name;
        std::vector<std::shared_ptr<BaseOption>> options;
        std::function<Output (Input)> process;
    };

    using SingleFrameIpTool = IpTool<SingleFrameInput, SingleFrameInput>;
}
