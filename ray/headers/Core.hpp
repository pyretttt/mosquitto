#pragma once

#include <functional>
#include <utility>

template <typename T, typename Callable>
constexpr inline T &&modified(T &&object, Callable modification) {
    modification(object);
    return object;
}

// Overload
template<class... Ts> 
struct overload: Ts... { 
    using Ts::operator()...;
};
template<class... Ts> overload(Ts...) -> overload<Ts...>;


// Type Id Generator
template <typename T>
class InstanceIdGenerator {
    static size_t instanceId;

public:
    size_t getInstanceId() const noexcept {
        return instanceId++;
    }
};

template <typename T> 
size_t InstanceIdGenerator<T>::instanceId = 0;


// ScopedExit
template <typename F>
struct [[nodiscard]] ScopeExit final : F
{
    ScopeExit(F &&f): F(std::forward<F>(f)) {}

    ~ScopeExit() { static_cast<F&>(*this)(); }
};

#define CONCAT(a, b) a ## b
#define CONCAT2(a, b) CONCAT(a, b)

#define SCOPE_EXIT(...) \
    auto CONCAT2(scope_exit_, __LINE__) = ::ScopeExit{[&] __VA_ARGS__ }