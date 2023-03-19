// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include "ads/lin/tensor.hpp"

#include <iostream>

#include <catch2/catch.hpp>

namespace lin = ads::lin;

TEST_CASE("Tensor") {
    SECTION("Buffer is correctly updated after writing by index") {
        auto t = lin::tensor<double, 2>{{5, 3}};
        t(2, 1) = 17.0;
        int idx = 5 + 2;
        CHECK(t.data()[idx] == 17.0);
    }

    SECTION("Equal tensors are ==") {
        int p = 5;
        int q = 3;
        auto a = lin::tensor<double, 2>{{p, q}};
        auto b = lin::tensor<double, 2>{{p, q}};

        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < q; ++j) {
                a(i, j) = b(i, j) = 7;
            }
        }
        CHECK(a == b);
    }

    SECTION("Reshaping") {
        int p = 5;
        int q = 3;

        double data[] = {
            1,  2,  3,  4,  5,   //
            6,  7,  8,  9,  10,  //
            11, 12, 13, 14, 15   //
        };
        auto tensor2d = lin::tensor_view<double, 2>{data, {p, q}};
        auto tensor1d = lin::tensor_view<double, 1>{data, {p * q}};
        auto reshaped = reshape(tensor2d, p * q);
        CHECK(tensor1d == reshaped);
    }

    SECTION("Cyclic transpose") {
        int k = 2;
        int n = 3;
        int m = 2;

        auto a = lin::tensor<double, 3>{{k, n, m}};
        auto e = lin::tensor<double, 3>{{n, m, k}};

        a(0, 0, 0) = e(0, 0, 0) = 111;
        a(1, 0, 0) = e(0, 0, 1) = 211;
        a(0, 1, 0) = e(1, 0, 0) = 121;
        a(1, 1, 0) = e(1, 0, 1) = 221;
        a(0, 2, 0) = e(2, 0, 0) = 131;
        a(1, 2, 0) = e(2, 0, 1) = 231;

        a(0, 0, 1) = e(0, 1, 0) = 112;
        a(1, 0, 1) = e(0, 1, 1) = 212;
        a(0, 1, 1) = e(1, 1, 0) = 122;
        a(1, 1, 1) = e(1, 1, 1) = 222;
        a(0, 2, 1) = e(2, 1, 0) = 132;
        a(1, 2, 1) = e(2, 1, 1) = 232;

        auto out = lin::tensor<double, 3>{{n, m, k}};
        cyclic_transpose(a, out);
        CHECK(out == e);

        auto a2 = lin::tensor<double, 3>{{m, k, n}};
        auto a3 = lin::tensor<double, 3>{{k, n, m}};
        cyclic_transpose(out, a2);
        cyclic_transpose(a2, a3);
        CHECK(a3 == a);
    }
}
