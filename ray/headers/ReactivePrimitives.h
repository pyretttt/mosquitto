#pragma once

#include <functional>

#include "boost/signals2.hpp"

// using DisposePool = std::vector<boost::signals2::scoped_connection>;

// template <typename T>
// class ObservableObject {
// public:
//     explicit ObservableObject(T initialValue) : 
//         currentValue(initialValue),
//         sig(std::make_shared<boost::signals2::signal<void(T)>>()) {
//             connection = sig->connect([this](T value) {
//                 this->currentValue = value;
//             });
//     }

//     ObservableObject(ObservableObject<T> const &other) : currentValue(other.currentValue), sig(other.sig) {
//         connection = sig->connect([this](T value) {
//             this->currentValue = value;
//         });
//     }

//     ObservableObject(ObservableObject<T> &&other) : currentValue(other.currentValue), sig(std::move(other.sig)) {
//         connection = sig->connect([this](T value) {
//             this->currentValue = value;
//         });
//     }

//     ObservableObject<T> operator=(ObservableObject<T> &&other) {
//         currentValue = other.currentValue;
//         sig = std::move(other.sig);
//         connection = sig->connect([this](T value) {
//             this->currentValue = value;
//         });
//         return *this;
//     }

//     ObservableObject<T> operator=(ObservableObject<T> const &other) {
//         currentValue = other.currentValue;
//         sig = other.sig;
//         connection = sig->connect([this](T value) {
//             this->currentValue = value;
//         });
//         return *this;
//     }

//     T value() const noexcept {
//         return currentValue;
//     }

//     boost::signals2::scoped_connection addObserver(std::function<void(T)> observer) noexcept {
//         return boost::signals2::scoped_connection(sig->connect(observer));
//     }

// protected:
//     std::shared_ptr<boost::signals2::signal<void(T)>> sig;
//     boost::signals2::scoped_connection connection;
//     T currentValue;
// };

// template <typename T>
// class ObservableProperty final : public ObservableObject<T> {
// public:
//     using ObservableObject<T>::ObservableObject;
//     using ObservableObject<T>::operator=;
//     using ObservableObject<T>::value;

//     void value(T &&val) {
//         ObservableObject<T>::sig->operator()(val);
//     }
// };


struct Connection {
    Connection(std::function<void()> &&dispose) : dipose(dispose) {}
    Connection(std::function<void()> const &dispose) : dipose(dispose) {}
    
    ~Connection() {
        dispose();
    }

    std::function<void()> dispose;
}

template <typename T>
struct Observer {
    Observer(std::function<T> &&notify) : notify(notify) {};
    std::function<T> notify;
};

template <typename T>
class Signal final {
public:
    Signal() = default;
    Signal(Signal const &other): counter(other.counter), observers(other.observers) {}
    Signal(Signal &&other): counter(other.counter), observers(std::move(other.observers)) {}
    Signal &operator=(Signal const &other) {
        counter = other.counter;
        observers = other.observers;
        return *this;
    }
    Signal &operator=(Signal &&other) {
        counter = std::move(other.counter);
        observers = std::move(other.observers);
        return *this;
    }
     
    Connection subscribe(Observer<T> &&observer) noexcept {
        auto key = counter++;
        observers[key] = std::move(observer);
        keys.push_back(key)
        std::weak_ptr<std::map<Key, Observer<T>> weakObservers(observers);
        return Connection([weakObservers, keys, key]() {
            if (auto spt = weakObservers.lock()) {
                spt->erase(key);
            }
        });
    }

    void send(T const &value) const noexcept {
        auto const &observers = *observers;
        for (auto const &[key, observer] : observers) {
            observer.notify(std::move(value));
        }
    }

private:
    using Key = uint32_t;
    Key counter = 0;
    std::shared_ptr<std::map<Key, Observer<T>>> observers;
};

template <typename UpstreamValue, typename ThisValue>
class ObservableObject {
public:
    ObservableObject(ThisValue initialValue, Signal<UpstreamValue> &upstream, std::function<) : value(std::make_shared(initialValue)) {
        upstream.subscribe(
            Observer<UpstreamValue>([value](T newValue) {
                *value = newValue;
            })
        )
    }

private:
    std::shared_ptr<T> value;    
};


// Decompose:
// Upstream
// Each operator as observer itself?
// Chain of observers