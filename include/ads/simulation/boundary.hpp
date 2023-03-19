// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_SIMULATION_BOUNDARY_HPP
#define ADS_SIMULATION_BOUNDARY_HPP

#include <type_traits>

namespace ads {

enum class boundary : unsigned {
    left = 0x1,
    right = 0x2,
    bottom = 0x4,
    top = 0x8,

    vertical = top | bottom,
    horizontal = left | right,
    full = vertical | horizontal,
    none = 0,
};

inline constexpr bool operator!(boundary a) {
    return a == boundary::none;
}

inline constexpr boundary operator&(boundary a, boundary b) {
    using T = std::underlying_type_t<boundary>;
    return boundary(static_cast<T>(a) & static_cast<T>(b));
}

inline constexpr boundary& operator&=(boundary& a, boundary b) {
    return a = a & b;
}

inline constexpr boundary operator|(boundary a, boundary b) {
    using T = std::underlying_type_t<boundary>;
    return boundary(static_cast<T>(a) | static_cast<T>(b));
}

inline constexpr boundary& operator|=(boundary& a, boundary b) {
    return a = a | b;
}

inline constexpr boundary operator^(boundary a, boundary b) {
    using T = std::underlying_type_t<boundary>;
    return boundary(static_cast<T>(a) ^ static_cast<T>(b));
}

inline constexpr boundary& operator^=(boundary& a, boundary b) {
    return a = a ^ b;
}

inline constexpr boundary operator~(boundary a) {
    return (a ^ boundary::full) & boundary::full;
}

inline constexpr bool contains(boundary set, boundary a) {
    return (set & a) == a;
}

inline constexpr bool intersects(boundary a, boundary b) {
    return (a & b) == boundary::none;
}

}  // namespace ads

#endif  // ADS_SIMULATION_BOUNDARY_HPP
