#pragma once

#include <memory>
#include <functional>

template<typename T>
struct Lazy final {
    Lazy() = default;
    explicit Lazy(std::function<std::shared_ptr<T>()> &&factory) : factory(std::move(factory)) {}

    std::shared_ptr<T> &operator()() {
        if (instance) {
            return instance;
        }
        instance = factory();
        return instance;
    }

private:
    std::shared_ptr<T> instance;
    std::function<std::shared_ptr<T>()> factory;
};