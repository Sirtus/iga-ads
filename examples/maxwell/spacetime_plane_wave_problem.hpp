// SPDX-FileCopyrightText: 2015 - 2021 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ST_MAXWELL_ANTENNA_PROBLEM_HPP
#define ST_MAXWELL_ANTENNA_PROBLEM_HPP

#include <cmath>

#include "excitation.hpp"
#include "problems.hpp"
#include "utils.hpp"

class spacetime_antenna_problem : public maxwell_problem<spacetime_antenna_problem> {
public:
    static constexpr double pi = M_PI;

    static constexpr double eps0 = 8.854e-12;
    static constexpr double mu0 = 12.556e-7;
    static constexpr double c = 1 / std::sqrt(eps0 * mu0);
    static constexpr double eta = c * mu0;

    static constexpr double f = 2 * c;
    static constexpr double omega = 2 * pi * f;
    static constexpr double k = omega / c;
    static constexpr double T0 = 1 / f;
    static constexpr double tau = 2 * T0;
    static constexpr double t0 = 4 * tau;

    static constexpr vec3   E0 = normalized(vec3{0, 0, 1});


    using vec3 = ads::math::vec<3>;

    static constexpr ads::point3_t center{1, 1, 1};

    static constexpr tapered_excitation excitation{tau};

    auto eps(point_type /*x*/) const -> double { return eps0; }
    auto mu(point_type /*x*/) const -> double { return mu0; }


    auto J(point_type p, double t) const -> vec3 {
        auto const z = p[2];

        auto const g = excitation.value(t - z / c);
        auto const dg = excitation.deriv(t - z / c);

        ads::point3_t v;
        v[0] = (-1/eps0) * (E0 / eta) * (((omega / c) * std::sin(omega * (t - z / c)) * g) - ((1 / c) std::cos(omega * (t - z / c)) * dg));
        v[1] = (-E0 / eta) * ((-omega * std::sin(omega * (t - z / c)) * g) + (std::cos(omega * (t - z / c)) * dg));

        return as_vec3(v);
    }

    auto as_vec3(ads::point3_t v) const -> vec3 {
        auto const [x, y, z] = v;
        return {x, y, z};
    }

    auto J1(point_type x, double t) const -> value_type { return {J(x, t).x, 0, 0, 0}; }
    auto J2(point_type x, double t) const -> value_type { return {J(x, t).y, 0, 0, 0}; }
    auto J3(point_type x, double t) const -> value_type { return {J(x, t).z, 0, 0, 0}; }

};

#endif  // ST_MAXWELL_ANTENNA_PROBLEM_HPP
