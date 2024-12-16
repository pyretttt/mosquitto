#pragma once

#include <functional>

template<typename T>
struct Lazy final {
    Lazy(std::function<T()> factory) : factory(factory) {}

    T operator()() {
        if (instance) {
            return instance;
        }
        instance = std::make_shared(factory());
    }
private:
    std::shared_ptr<T> instance;
    std::function<T()> factory;
};