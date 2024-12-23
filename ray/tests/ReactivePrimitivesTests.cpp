#include <gtest/gtest.h>

#include "ReactivePrimitives.hpp"
 
TEST(
    ReactivePrimitives, ObservableValuesTest
) {
    auto observable{Observable<float>::values({1.f})};
    float expected = 0.f;
    observable.subscribe([&expected](float value){
        expected = value;
    });
    EXPECT_EQ(expected, 1.f);
}
