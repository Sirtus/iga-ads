// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include "ads/bspline/bspline.hpp"

#include <catch2/catch.hpp>

namespace bsp = ads::bspline;
using Catch::Matchers::Equals;

TEST_CASE("B-spline basis", "[splines]") {
    bsp::basis b = bsp::create_basis(0.0, 1.0, 2, 4);

    SECTION("Knot vector is correct after creation") {
        REQUIRE_THAT(b.knot, Equals<double>({0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1}));
    }

    SECTION("Finding span") {
        SECTION("point outside domain") {
            // degree = 2 is the lowest possible result
            CHECK(find_span(-1, b) == 2);
            CHECK(find_span(2.0, b) == 5);
        }
        SECTION("point on domain boundary") {
            CHECK(find_span(0.0, b) == 2);
            CHECK(find_span(1.0, b) == 5);
        }
        SECTION("point on element boundary") {
            CHECK(find_span(0.25, b) == 3);
            CHECK(find_span(0.75, b) == 5);
        }
        SECTION("point in element interior") {
            CHECK(find_span(0.1, b) == 2);
            CHECK(find_span(0.3, b) == 3);
            CHECK(find_span(0.7, b) == 4);
            CHECK(find_span(0.9, b) == 5);
        }
    }

    SECTION("Finding span with repeated nodes") {
        bsp::basis b_rep({0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5}, 3);

        SECTION("point outside domain") {
            CHECK(find_span(-4, b_rep) == 3);
            CHECK(find_span(8, b_rep) == 13);
        }
        SECTION("point on domain boundary") {
            CHECK(find_span(0, b_rep) == 3);
            CHECK(find_span(5, b_rep) == 13);
        }
        SECTION("point on element boundary") {
            CHECK(find_span(1, b_rep) == 5);
            CHECK(find_span(2, b_rep) == 6);
            CHECK(find_span(3, b_rep) == 9);
            CHECK(find_span(4, b_rep) == 13);
        }
        SECTION("point in element interior") {
            CHECK(find_span(0.3, b_rep) == 3);
            CHECK(find_span(1.2, b_rep) == 5);
            CHECK(find_span(2.8, b_rep) == 6);
            CHECK(find_span(3.1, b_rep) == 9);
            CHECK(find_span(4.9, b_rep) == 13);
        }
    }

    SECTION("First non-zero dofs") {
        REQUIRE_THAT(first_nonzero_dofs(b), Equals<int>({0, 1, 2, 3}));
    }
}
