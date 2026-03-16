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
        std::string name;
        OptionKind kind;

        BaseOption() = default;
        BaseOption(OptionKind kind, std::string name) : name(std::move(name)), kind(kind) {}
        virtual ~BaseOption() = default;
    };

    struct IntOption final : public BaseOption {
        int value = 0;

        IntOption() : BaseOption(OptionKind::Int, {}) {}
        IntOption(IntOption &&) = default;
        IntOption(int value) : BaseOption(OptionKind::Int, {}), value(value) {}
        IntOption(std::string name, int value) : BaseOption(OptionKind::Int, std::move(name)), value(value) {}
    };

    struct FloatOption final : public BaseOption {
        float value = 0.f;

        FloatOption() : BaseOption(OptionKind::Float, {}) {}
        FloatOption(FloatOption &&) = default;
        FloatOption(float value) : BaseOption(OptionKind::Float, {}), value(value) {}
        FloatOption(std::string name, float value) : BaseOption(OptionKind::Float, std::move(name)), value(value) {}
    };

    struct BoolOption final : public BaseOption {
        bool value = false;

        BoolOption() : BaseOption(OptionKind::Bool, {}) {}
        BoolOption(BoolOption &&) = default;
        BoolOption(bool value) : BaseOption(OptionKind::Bool, {}), value(value) {}
        BoolOption(std::string name, bool value) : BaseOption(OptionKind::Bool, std::move(name)), value(value) {}
    };

    struct StringOption final : public BaseOption {
        std::string value;

        StringOption() : BaseOption(OptionKind::String, {}) {}
        StringOption(StringOption &&) = default;
        StringOption(std::string value) : BaseOption(OptionKind::String, {}), value(std::move(value)) {}
        StringOption(std::string name, std::string value) : BaseOption(OptionKind::String, std::move(name)), value(std::move(value)) {}
    };

    struct MultiStringOption final : public BaseOption {
        std::vector<std::string> values;
        size_t selected = 0;

        MultiStringOption() : BaseOption(OptionKind::MultiString, {}) {}
        MultiStringOption(MultiStringOption &&) = default;
        MultiStringOption(std::vector<std::string> values, size_t selected = 0)
            : BaseOption(OptionKind::MultiString, {}), values(std::move(values)), selected(selected) {}
        MultiStringOption(std::string name, std::vector<std::string> values, size_t selected = 0)
            : BaseOption(OptionKind::MultiString, std::move(name)), values(std::move(values)), selected(selected) {}
    };

    struct MultiIntegerOption final : public BaseOption {
        std::vector<int> values;
        size_t selected = 0;

        MultiIntegerOption() : BaseOption(OptionKind::MultiInteger, {}) {}
        MultiIntegerOption(MultiIntegerOption &&) = default;
        MultiIntegerOption(std::vector<int> values, size_t selected = 0)
            : BaseOption(OptionKind::MultiInteger, {}), values(std::move(values)), selected(selected) {}
        MultiIntegerOption(std::string name, std::vector<int> values, size_t selected = 0)
            : BaseOption(OptionKind::MultiInteger, std::move(name)), values(std::move(values)), selected(selected) {}
    };

    struct MultiFloatOption final : public BaseOption {
        std::vector<float> values;
        size_t selected = 0;

        MultiFloatOption() : BaseOption(OptionKind::MultiFloat, {}) {}
        MultiFloatOption(MultiFloatOption &&) = default;
        MultiFloatOption(std::vector<float> values, size_t selected = 0)
            : BaseOption(OptionKind::MultiFloat, {}), values(std::move(values)), selected(selected) {}
        MultiFloatOption(std::string name, std::vector<float> values, size_t selected = 0)
            : BaseOption(OptionKind::MultiFloat, std::move(name)), values(std::move(values)), selected(selected) {}
    };

    using IntOptionPtr = std::shared_ptr<IntOption>;
    using FloatOptionPtr = std::shared_ptr<FloatOption>;
    using BoolOptionPtr = std::shared_ptr<BoolOption>;
    using StringOptionPtr = std::shared_ptr<StringOption>;
    using MultiStringOptionPtr = std::shared_ptr<MultiStringOption>;
    using MultiIntegerOptionPtr = std::shared_ptr<MultiIntegerOption>;
    using MultiFloatOptionPtr = std::shared_ptr<MultiFloatOption>;
    using OptionList = std::vector<std::shared_ptr<BaseOption>>;

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
        return std::static_pointer_cast<IntOption>(std::move(ptr));
    }

    inline std::shared_ptr<FloatOption> asFloat(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<FloatOption>(std::move(ptr));
    }

    inline std::shared_ptr<BoolOption> asBool(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<BoolOption>(std::move(ptr));
    }

    inline std::shared_ptr<StringOption> asString(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<StringOption>(std::move(ptr));
    }

    inline std::shared_ptr<MultiStringOption> asMultiString(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<MultiStringOption>(std::move(ptr));
    }

    inline std::shared_ptr<MultiIntegerOption> asMultiInteger(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<MultiIntegerOption>(std::move(ptr));
    }

    inline std::shared_ptr<MultiFloatOption> asMultiFloat(std::shared_ptr<BaseOption> ptr) {
        return std::static_pointer_cast<MultiFloatOption>(std::move(ptr));
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
