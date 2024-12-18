#pragma once

#include <functional>

#include "boost/signals2.hpp"

using DisposePool = std::vector<boost::signals2::scoped_connection>;

template <typename T>
class ObservableObject {
public:
    explicit ObservableObject(T initialValue) : 
    currentValue(initialValue),
    sig(std::make_shared<boost::signals2::signal<void(T)>>()) {
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
    }
    ObservableObject(ObservableObject<T> const &other) : currentValue(other.currentValue), sig(other.sig) {
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
    }

    ObservableObject(ObservableObject<T> &&other) : currentValue(other.currentValue), sig(std::move(other.sig)) {
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
    }

    ObservableObject<T> operator=(ObservableObject<T> &&other) {
        currentValue = other.currentValue;
        sig = std::move(other.sig);
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
        return *this;
    }

    ObservableObject<T> operator=(ObservableObject<T> const &other) {
        currentValue = other.currentValue;
        sig = other.sig;
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
        return *this;
    }

    T value() const noexcept {
        return currentValue;
    }

    boost::signals2::scoped_connection addObserver(std::function<void(T)> observer) noexcept {
        return boost::signals2::scoped_connection(sig->connect(observer));
    }

protected:
    std::shared_ptr<boost::signals2::signal<void(T)>> sig;
    boost::signals2::scoped_connection connection;
    T currentValue;
};

template <typename T>
class ObservableProperty final : public ObservableObject<T> {
public:
    using ObservableObject<T>::ObservableObject;
    using ObservableObject<T>::operator=;
    using ObservableObject<T>::value;

    void value(T &&val) {
        ObservableObject<T>::sig->operator()(val);
    }
};
