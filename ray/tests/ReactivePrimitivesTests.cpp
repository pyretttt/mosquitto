#include <string>

#include <gtest/gtest.h>

#include "ReactivePrimitives.hpp"

TEST(
    ReactivePrimitives, ObservableValuesTest
) {
    auto observable{Observable<float>::values({1.f})};
    float expected = 0.f;
    observable.subscribe([&expected](float const &value) {
        expected = value;
    });
    EXPECT_EQ(expected, 1.f);
}

TEST(
    ReactivePrimitives, ObservableMapTest
) {
    auto observable{Observable<int>::values({1, 2})};
    Observable<std::string> mapped{observable.map<std::string>([](int const &value) {
        return std::to_string(value);
    })};
    std::vector<std::string> strings;
    mapped.subscribe([&strings](std::string const &value) {
        strings.push_back(value);
    });
    std::vector<std::string> expectedResult{"1", "2"};
    EXPECT_EQ(strings, expectedResult);
}