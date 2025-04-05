#pragma once

#include <functional>

template <typename T>
constexpr inline T &&modified(T &&object, std::function<void(T &)> modification) {
    modification(object);
    return object;
}


template<class... Ts> 
struct overload: Ts... { 
    using Ts::operator()...;
};
template<class... Ts> overload(Ts...) -> overload<Ts...>;