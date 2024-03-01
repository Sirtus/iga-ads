// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_UTIL_FUNCTION_VALUE_FUNCTION_VALUE_4d_HPP
#define ADS_UTIL_FUNCTION_VALUE_FUNCTION_VALUE_4d_HPP

namespace ads {

struct function_value_4d {
    double val;
    double dx, dy, dz, dw;

    constexpr function_value_4d(double val, double dx, double dy, double dz, double dw) noexcept
    : val{val}
    , dx{dx}
    , dy{dy}
    , dz{dz}
    , dw{dw} { }

    constexpr function_value_4d() noexcept
    : function_value_4d{0, 0, 0, 0, 0} { }

    function_value_4d& operator+=(const function_value_4d& v) {
        val += v.val;
        dx += v.dx;
        dy += v.dy;
        dz += v.dz;
        dw += v.dw;
        return *this;
    }

    function_value_4d& operator-=(const function_value_4d& v) {
        val -= v.val;
        dx -= v.dx;
        dy -= v.dy;
        dz -= v.dz;
        dw -= v.dw;
        return *this;
    }

    function_value_4d operator-() const { return {-val, -dx, -dy, -dz, -dw}; }

    function_value_4d& operator*=(double a) {
        val *= a;
        dx *= a;
        dy *= a;
        dz *= a;
        dw *= a;
        return *this;
    }

    function_value_4d& operator/=(double a) {
        val /= a;
        dx /= a;
        dy /= a;
        dz /= a;
        dw /= a;
        return *this;
    }
};

inline function_value_4d operator+(function_value_4d x, const function_value_4d& v) {
    x += v;
    return x;
}

inline function_value_4d operator-(function_value_4d x, const function_value_4d& v) {
    x -= v;
    return x;
}

inline function_value_4d operator*(double a, function_value_4d u) {
    u *= a;
    return u;
}

inline function_value_4d operator*(function_value_4d u, double a) {
    u *= a;
    return u;
}

inline function_value_4d operator/(function_value_4d u, double a) {
    u /= a;
    return u;
}

}  // namespace ads

#endif  // ADS_UTIL_FUNCTION_VALUE_function_value_4d_HPP
