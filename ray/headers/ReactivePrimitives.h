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

using Connections = std::vector<std::unique_ptr<Connection>>;

template <typename T>
struct Observer final {
    explicit Observer(
        std::function<void(T const &)> action
    ) : action(std::move(action)) {}

    Observer(
        Observer<T> &&other
    ) : action(std::move(other.action)) {
    }

    Observer(
        Observer<T> const &other
    ) : action(other.action) {}

    Observer<T> &operator=(
        Observer<T> const &other
    ) {
        action = other.action;
        return *this;
    }

    Observer<T> &operator=(
        Observer<T> &&other
    ) {
        action = std::move(other.action);
        return *this;
    }

    std::function<void(T const &)> action;
};

template <typename T>
class Observable final {
public:
    explicit Observable(
        std::function<std::unique_ptr<Connection>(Observer<T>)> connectObserver
    ) : connectObserver(std::move(connectObserver)) {}

    explicit Observable(
        Observable<T> const &other
    ) : connectObserver(other.connectObserver) {}

    explicit Observable(
        Observable<T> &&other
    ) : connectObserver(std::move(other.connectObserver)) {}

    Observable<T> static inline values(
        std::vector<T> values
    ) {
        return Observable<T>([values = std::move(values)](auto observer) {
            for (auto const &value : values) {
                observer.action(value);
            }
            return std::make_unique<Connection>();
        });
    }

    template <typename V>
    Observable<V> map(
        std::function<V(T const &)> conversion
    ) {
        auto connect = connectObserver;
        return Observable([connectObserver = std::move(connect), conversion = std::move(conversion)](Observer<T> observer) {
            observer.action = [action = std::move(observer.action), conversion](T const &value) {
                action(conversion(value));
            };

            return connectObserver(std::move(observer));
        });
    }

    std::unique_ptr<Connection> subscribe(
        std::function<void(T const &)> action
    ) const noexcept {
        return connectObserver(Observer(action));
    }

    std::unique_ptr<Connection> subscribe(
        Observer<T> observer
    ) const noexcept {
        return connectObserver(std::move(observer));
    }

private:
    std::function<std::unique_ptr<Connection>(Observer<T>)> connectObserver;
};

template <typename T>
class Channel final {
public:
    Channel() : observable(makeObservable()) {
    }

    Channel(Channel<T> const &other) = delete;
    Channel<T> &operator=(Channel<T> const &other) = delete;

    Channel(
        Channel<T> &&other
    ) : key(std::move(other.key)), observers(std::move(other.observers)), observable(std::move(other.observable)) {}

    Channel<T> &operator=(
        Channel<T> &&other
    ) {
        key = std::move(other.key);
        observers = std::move(other.observers);
        observable = std::move(observable);
        return *this;
    }

    std::unique_ptr<Connection> subscribe(
        std::function<void(T const &)> action
    ) {
        return observable.subscribe(Observer(action));
    }

    void send(
        T value
    ) const noexcept {
        for (auto const &[key, observer] : *observers) {
            observer.action(std::move(value));
        }
    }

    Observable<T> const &asObservable() const noexcept {
        return observable;
    }

private:
    Observable<T> makeObservable() {
        std::weak_ptr<std::map<Key, Observer<T>>> weakObservers(observers);
        std::shared_ptr<Key> key = this->key;
        return Observable<T>([=](Observer<T> observer) {
            auto key_ = *key;
            *key = key_ + 1;
            if (auto spt = weakObservers.lock()) {
                spt->insert(std::make_pair(key_, observer));
            }

            return std::make_unique<Connection>([=]() {
                if (auto spt = weakObservers.lock()) {
                    spt->erase(key_);
                }
            });
        });
    }

    using Key = uint32_t;
    std::shared_ptr<Key> key{std::make_shared<Key>(0)};
    std::shared_ptr<std::map<Key, Observer<T>>> observers = std::make_shared<std::map<Key, Observer<T>>>();
    Observable<T> observable = makeObservable();
};

template <typename T>
class ObservableObject {
public:
    explicit ObservableObject(
        T const constant
    ) : currentValue(std::make_shared<T>(std::move(constant))),
        observable(Observable<T>([constant](Observer<T> const &observer) {
            observer.action(constant);
            return std::make_unique<Connection>();
        })) {
        bindConnection(observable);
    }

    explicit ObservableObject(
        T const initialValue,
        Observable<T> observable
    ) : currentValue(std::make_shared<T>(std::move(initialValue))),
        observable(std::move(observable)) {
        bindConnection(observable);
    }

    ObservableObject(
        ObservableObject<T> const &other
    ) = delete;

    ObservableObject<T> &operator=(
        ObservableObject<T> const &other
    ) = delete;

    ObservableObject(
        ObservableObject<T> &&other
    ) : currentValue(std::move(other.currentValue)), observable(std::move(other.observable)), connection(std::move(other.connection)) {
    }

    ObservableObject<T> &operator=(
        ObservableObject<T> &&other
    ) {
        currentValue = std::move(other.currentValue);
        observable = std::move(other.observable);
        connection = std::move(other.connection);
        return *this;
    }

    T &value() const noexcept {
        return *currentValue;
    }

    std::unique_ptr<Connection> subscribe(
        std::function<void((T const &))> observer
    ) const noexcept {
        return observable.subscribe(observer);
    }

private:
    void bindConnection(
        Observable<T> const &observable
    ) {
        std::weak_ptr<T> wpt(currentValue);
        connection = observable.subscribe([wpt](T const &value) {
            if (auto spt = wpt.lock()) {
                *spt = value;
            }
        });
    }

protected:
    Observable<T> observable;
    std::unique_ptr<Connection> connection;
    std::shared_ptr<T> currentValue;
};

template <typename T>
class ObservableProperty final {
public:
    ObservableProperty(
        T const value
    ) : currentValue(std::make_shared<T>(std::move(value))) {
        bindConnection(channel.asObservable());
    }

    ObservableProperty(
        ObservableProperty<T> &&other
    ) : currentValue(std::move(other.currentValue)), connection(std::move(other.connection)), channel(std::move(other.channel)) {
    }

    T &value() const noexcept {
        return *currentValue;
    }

    void value(
        T const val
    ) const noexcept {
        channel.send(std::move(val));
    }

    ObservableObject<T> asObservableObject() const noexcept {
        return ObservableObject<T>(*currentValue, channel.asObservable());
    }

private:
    void bindConnection(
        Observable<T> const &observable
    ) {
        std::weak_ptr<T> wpt(currentValue);
        connection = observable.subscribe([wpt](T const &value) {
            if (auto spt = wpt.lock()) {
                *spt = value;
            }
        });
    }

    Channel<T> channel;
    std::unique_ptr<Connection> connection;
    std::shared_ptr<T> currentValue;
};