#pragma once

#include <functional>
#include <utility>

template <typename T>
constexpr inline T &&modified(T &&object, std::function<void(T &)> modification) {
    modification(object);
    return object;
}

// Overload
template<class... Ts> 
struct overload: Ts... { 
    using Ts::operator()...;
};
template<class... Ts> overload(Ts...) -> overload<Ts...>;


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