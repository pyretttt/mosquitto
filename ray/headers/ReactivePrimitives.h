#pragma once

#include <functional>

#include "boost/signals2.hpp"

struct Connection final {
    Connection() = default;
    explicit Connection(
        std::function<void()> dispose
    ) : dispose(std::move(dispose)) {}

    ~Connection() {
        dispose();
    }

private:
    std::function<void()> dispose;
};

using Connections = std::vector<Connection>;

template <typename T>
struct Observer final {
    explicit Observer(
        std::function<void(T const &)>
    ) : action(std::move(action)) {}

    std::function<void(T const &)> action;
};

template <typename T>
class Observable final {
public:
    explicit Observable(
        std::function<Connection(Observer<T>)> tieObserver
    ) : tieObserver(std::move(tieObserver)) {}

    Observable<T> static inline values(
        std::vector<T> values
    ) {
        return Observable<T>([values = std::move(values)](auto observer) {
            for (auto const &value : values) {
                observer.action(value);
            }
            return Connection();
        });
    }

    template <typename V>
    Observable<V> map(
        std::function<V(T const &)> conversion
    ) {
        auto tie = tieObserver;
        return Observable([tieObserver = std::move(tie), conversion = std::move(conversion)](Observer<T> observer) {
            observer.action = [action = std::move(observer.action), conversion](T const &value) {
                action(conversion(value));
            };

            return tieObserver(std::move(observer));
        });
    }

    Connection subscribe(
        std::function<void(T const &)> action
    ) {
        return tieObserver(Observer(action));
    }

private:
    std::function<Connection(Observer<T>)> tieObserver;
};

template <typename T>
class Channel final {
public:
    explicit Channel() {
        std::weak_ptr<std::map<Key, Observer<T>>> weakObservers(observers);
        std::shared_ptr<Key> key = this->key;
        observable = Observable<T>([=](Observer<T> observer) {
            auto key_ = *key;
            *key = key_ + 1;
            if (auto spt = weakObservers.lock()) {
                spt->operator()[key_] = std::move(observer);
            }

            return Connection([=]() {
                if (auto spt = weakObservers.lock()) {
                    spt->erase(key_);
                }
            });
        });
    }

    Connection subscribe(
        std::function<void(T const &)> action
    ) {
        return observable.subscribe(std::move(Observer(action)));
    }

    void send(
        T value
    ) const noexcept {
        for (auto const &[key, observer] : *observers) {
            observer.action(std::move(value));
        }
    }

    Observable<T> &asObservable() const noexcept {
        return observable;
    }

private:
    using Key = uint32_t;
    Observable<T> observable;
    std::shared_ptr<Key> key{std::make_shared<Key>(0)};
    std::shared_ptr<std::map<Key, Observer<T>>> observers{std::make_shared<std::map<Key, Observer<T>>>({})};
};

template <typename T>
class ObservableObject {
public:
    explicit ObservableObject(
        T initialValue
    ) : currentValue(std::move(initialValue)),
        sig(std::make_shared<Observable>()) {
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
    }

    ObservableObject(
        ObservableObject<T> const &other
    ) : currentValue(other.currentValue), sig(other.sig) {
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
    }

    ObservableObject(
        ObservableObject<T> &&other
    ) : currentValue(other.currentValue), sig(std::move(other.sig)) {
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
    }

    ObservableObject<T> operator=(
        ObservableObject<T> &&other
    ) {
        currentValue = other.currentValue;
        sig = std::move(other.sig);
        connection = sig->connect([this](T value) {
            this->currentValue = value;
        });
        return *this;
    }

    ObservableObject<T> operator=(
        ObservableObject<T> const &other
    ) {
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

    boost::signals2::scoped_connection addObserver(
        std::function<void(T)> observer
    ) noexcept {
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

    void value(
        T &&val
    ) {
        ObservableObject<T>::sig->operator()(val);
    }
};