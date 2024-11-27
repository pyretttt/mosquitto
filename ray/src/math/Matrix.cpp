#include "Matrix.h"

#include <cassert>

Matrix4 operator*(Matrix4& lhs, Matrix4& rhs) {
    Matrix4 res;
    for(size_t row = 0; row < 4; row++) {
        for (size_t col = 0; col < 4; col++) {
            float element = 0;
            for (size_t k = 0; k < 4; k++) {
                element += lhs.elements[row][k] * rhs.elements[k][col];
            }
            res.elements[row][col] = element;
        }
    }
    return res;
}