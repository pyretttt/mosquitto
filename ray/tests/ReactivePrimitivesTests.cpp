#include <string>

#include <gtest/gtest.h>

#include "ReactivePrimitives.hpp"

TEST(
    ReactivePrimitives, ObservableValues
) {
    auto observable{Observable<float>::values({1.f})};
    float expected = 0.f;
    observable.subscribe([&expected](float const &value) {
        expected = value;
    });
    EXPECT_EQ(expected, 1.f);
}

TEST(
    ReactivePrimitives, ObservableMap
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

TEST(
    ReactivePrimitives, ObservableCopy
) {
    auto observable{Observable<int>::values({1, 2})};
    auto copy{observable};
    std::vector<int> ints;
    copy.subscribe([&ints](int const &value) {
        ints.push_back(value);
    });
    std::vector<int> expectedResult{1, 2};
    EXPECT_EQ(ints, expectedResult);

    std::vector<int> sourceInts;
    observable.subscribe([&sourceInts](int const &value) {
        sourceInts.push_back(value);
    });
    EXPECT_EQ(sourceInts, expectedResult);
}

TEST(
    ReactivePrimitives, ChannelConnectionDisposition
) {
    auto channel{Channel<int>()};
    channel.subscribe([](int const &value){
        FAIL();
    });
    channel.send(1);
}

TEST(
    ReactivePrimitives, ChannelSendValue
) {
    auto channel{Channel<int>()};
    std::vector<int> result;
    auto connection = channel.subscribe([&result](int const &value){
        result.push_back(value);
    });
    channel.send(1);
    EXPECT_EQ(result, std::vector<int>({1}));
}

TEST(
    ReactivePrimitives, ChannelSendValueAndDisconnect
) {
    auto channel{Channel<int>()};
    std::vector<int> result;
    auto connection = channel.subscribe([&result](int const &value){
        result.push_back(value);
    });
    channel.send(1);
    EXPECT_EQ(result, std::vector<int>({1}));
    connection.reset(nullptr);
    channel.send(2);
    EXPECT_EQ(result, std::vector<int>({1}));
}

TEST(
    ReactivePrimitives, ChannelToObservable
) {
    auto channel{Channel<int>()};
    Observable<int> observable = channel.asObservable();
    std::vector<int> result;
    auto connection = observable.subscribe([&result](int const &value){
        result.push_back(value);
    });
    channel.send(1);
    EXPECT_EQ(result, std::vector<int>({1}));
    connection.reset(nullptr);
    channel.send(2);
    EXPECT_EQ(result, std::vector<int>({1}));
}

TEST(
    ReactivePrimitives, ChannelToObservableRef
) {
    auto channel{Channel<int>()};
    Observable<int> const &observable = channel.asObservable();
    std::vector<int> result;
    auto connection = observable.subscribe([&result](int const &value){
        result.push_back(value);
    });
    channel.send(1);
    EXPECT_EQ(result, std::vector<int>({1}));
    connection.reset(nullptr);
    channel.send(2);
    EXPECT_EQ(result, std::vector<int>({1}));
}

TEST(
    ReactivePrimitives, ChannelMultipleObservers
) {
    auto channel{Channel<int>()};
    Observable<int> const observable = channel.asObservable();
    std::vector<std::string> resultFirst;
    auto connectionFirst = observable.map<std::string>([](int const &value){
        return std::to_string(value);
    })
    .subscribe([&resultFirst](std::string const &value){
        resultFirst.push_back(value);
    });

    std::vector<int> resultSecond;
    auto connectionSecond = channel.subscribe([&resultSecond](int const &value){
        resultSecond.push_back(value);
    });

    channel.send(1);
    EXPECT_EQ(resultFirst, std::vector<std::string>({"1"}));
    EXPECT_EQ(resultSecond, std::vector<int>({1}));
    connectionFirst.reset(nullptr);
    channel.send(2);
    EXPECT_EQ(resultFirst, std::vector<std::string>({"1"}));
    EXPECT_EQ(resultSecond, std::vector<int>({1, 2}));
}