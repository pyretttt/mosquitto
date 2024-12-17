#pragma once

#include <functional>

template<typename T>
struct Lazy final {
    Lazy(std::function<T()> &&factory) : factory(std::move(factory)) {}

    std::shared_ptr<T> operator()() {
        if (instance) {
            return instance;
        }
        instance = std::make_shared<T>(factory());
    }
private:
    std::shared_ptr<T> instance;
    std::function<T()> factory;
};