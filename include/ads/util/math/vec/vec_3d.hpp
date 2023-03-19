// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_UTIL_MATH_VEC_VEC_3D_HPP
#define ADS_UTIL_MATH_VEC_VEC_3D_HPP

namespace ads::math {

template <>
struct vec<3> {
    using vec_type = vec<3>;

    double x, y, z;

    constexpr vec_type& operator+=(const vec_type& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    constexpr vec_type& operator-=(const vec_type& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    constexpr vec_type operator-() const { return {-x, -y, -z}; }

    constexpr vec_type& operator*=(double a) {
        x *= a;
        y *= a;
        z *= a;
        return *this;
    }

    constexpr vec_type& operator/=(double a) {
        double inv = 1 / a;
        return (*this) *= inv;
    }

    constexpr double dot(const vec_type& v) const { return x * v.x + y * v.y + z * v.z; }

    constexpr double norm_sq() const { return x * x + y * y + z * z; }
};

}  // namespace ads::math

#endif  // ADS_UTIL_MATH_VEC_VEC_3D_HPP
