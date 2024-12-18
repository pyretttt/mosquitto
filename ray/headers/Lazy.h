#pragma once

#include <memory>
#include <functional>

template<typename T>
struct Lazy final {
    Lazy() = default;
    Lazy(std::function<T()> &&factory) : factory(std::move(factory)) {}

    T &operator()() {
        if (instance) {
            return instance;
        }
        instance = factory();
        return instance;
    }
private:
    T instance;
    std::function<T()> factory;
};