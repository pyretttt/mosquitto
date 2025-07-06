#pragma once

#include <memory>
#include <functional>
#include <optional>

template<typename T>
struct Lazy final {
    Lazy() = default;
    explicit Lazy(std::function<T ()> &&factory) : factory(std::move(factory)) {}

    T &operator()() {
        if (instance) {
            return instance.value();
        }
        instance = factory();
        return instance.value();
    }

private:
    std::optional<T> instance = std::nullopt;
    std::function<T ()> factory;
};