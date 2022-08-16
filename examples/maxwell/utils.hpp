// SPDX-FileCopyrightText: 2015 - 2021 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef MAXWELL_UTILS_HPP
#define MAXWELL_UTILS_HPP

#include <cmath>
#include <iostream>

#include <fmt/core.h>

#include "ads/experimental/all.hpp"

auto radius(double x, double y, double z) -> double {
    return std::hypot(x, y, z);
}

auto theta(double x, double y, double z) -> double {
    auto const r = std::hypot(x, y, z);
    return std::acos(z / r);
}

auto phi(double x, double y, double /*z*/) -> double {
    return std::atan2(y, x);
}

auto cartesian_to_spherical(ads::point3_t p, ads::point3_t center) -> ads::point3_t {
    auto const [x1, y1, z1] = p;
    auto const [x2, y2, z2] = center;
    auto const [x, y, z] = ads::point3_t{x1 - x2, y1 - y2, z1 - z2};

    auto const r = std::hypot(x, y, z);
    auto const theta = std::acos(z / r);
    auto const phi = std::atan2(y, x);

    return ads::point3_t{r, theta, phi};
}

auto tangent_cartesian_to_spherical(ads::point3_t sph_p, ads::point3_t v) -> ads::point3_t {
    auto const [r, theta, phi] = sph_p;
    auto const [vx, vy, vz] = v;

    auto const sphi = std::sin(phi);
    auto const cphi = std::cos(phi);
    auto const sth = std::sin(theta);
    auto const cth = std::cos(theta);

    double const A[3][3] = {
        {sth * cphi, sth * sphi, cth},
        {cth * cphi, cth * sphi, -sth},
        {-sphi, cphi, 0},
    };
    auto const sr = A[0][0] * vx + A[0][1] * vy + A[0][2] * vz;
    auto const st = A[1][0] * vx + A[1][1] * vy + A[1][2] * vz;
    auto const sp = A[2][0] * vx + A[2][1] * vy + A[2][2] * vz;

    return {sr, st, sp};
}

auto tangent_spherical_to_cartesian(ads::point3_t sph_p, ads::point3_t v) -> ads::point3_t {
    auto const [r, theta, phi] = sph_p;
    auto const [sr, st, sp] = v;

    auto const sphi = std::sin(phi);
    auto const cphi = std::cos(phi);
    auto const sth = std::sin(theta);
    auto const cth = std::cos(theta);

    double const A[3][3] = {
        {sth * cphi, cth * cphi, -sphi},
        {sth * sphi, cth * sphi, cphi},
        {cth, -sth, 0},
    };
    auto const vx = A[0][0] * sr + A[0][1] * st + A[0][2] * sp;
    auto const vy = A[1][0] * sr + A[1][1] * st + A[1][2] * sp;
    auto const vz = A[2][0] * sr + A[2][1] * st + A[2][2] * sp;

    return {vx, vy, vz};
}

auto spherical(ads::point3_t p, ads::point3_t v) -> ads::point3_t {
    auto const sph_p = cartesian_to_spherical(p, {1, 1, 1});
    return tangent_cartesian_to_spherical(sph_p, v);
}

template <typename FEx, typename FEy, typename FEz, typename FHx, typename FHy, typename FHz>
auto maxwell_to_file(const std::string& path,       //
                     FEx&& Ex, FEy&& Ey, FEz&& Ez,  //
                     FHx&& Hx, FHy&& Hy, FHz&& Hz   //
                     ) -> void {
    // constexpr auto res_x = 50 * 3;
    // constexpr auto res_y = 50 * 3;
    // constexpr auto res_z = 50;
    // auto extent = fmt::format("0 {} 0 {} 0 {}", res_x, res_y, res_z);
    // auto const rx = ads::interval{0, 450e3};
    // auto const ry = ads::interval{0, 450e3};
    // auto const rz = ads::interval{0, 150e3};
    constexpr auto res_x = 49;
    constexpr auto res_y = 49;
    constexpr auto res_z = 49;
    auto extent = fmt::format("0 {} 0 {} 0 {}", res_x, res_y, res_z);
    auto const rx = ads::interval{0, 2};
    auto const ry = ads::interval{0, 2};
    auto const rz = ads::interval{0, 2};

    auto const for_all_points = [&](auto&& fun) {
        for (auto z : ads::evenly_spaced(rz, res_z)) {
            for (auto y : ads::evenly_spaced(ry, res_y)) {
                for (auto x : ads::evenly_spaced(rx, res_x)) {
                    const auto X = ads::point3_t{x, y, z};
                    fun(X);
                }
            }
        }
    };

    auto out = fmt::output_file(path);
    out.print("<?xml version=\"1.0\"?>\n");
    out.print("<VTKFile type=\"ImageData\" version=\"0.1\">\n");
    out.print("  <ImageData WholeExtent=\"{}\" origin=\"0 0 0\" spacing=\"1 1 1\">\n", extent);
    out.print("    <Piece Extent=\"{}\">\n", extent);
    out.print("      <PointData Scalars=\"r\" Vectors=\"E\">\n", extent);

    out.print("        <DataArray Name=\"E\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) { out.print("{:.7} {:.7} {:.7}\n", Ex(X), Ey(X), Ez(X)); });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"H\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) { out.print("{:.7} {:.7} {:.7}\n", Hx(X), Hy(X), Hz(X)); });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"E spherical\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) {
        auto const [r, theta, phi] = spherical(X, {Ex(X), Ey(X), Ez(X)});
        out.print("{:.7} {:.7} {:.7}\n", r, theta, phi);
    });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"H spherical\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) {
        auto const [r, theta, phi] = spherical(X, {Hx(X), Hy(X), Hz(X)});
        out.print("{:.7} {:.7} {:.7}\n", r, theta, phi);
    });
    out.print("        </DataArray>\n");

    out.print("      </PointData>\n");
    out.print("    </Piece>\n");
    out.print("  </ImageData>\n");
    out.print("</VTKFile>\n");
}

template <typename FEx, typename FEy, typename FEz, typename FHx, typename FHy, typename FHz,
          typename REx, typename REy, typename REz, typename RHx, typename RHy, typename RHz>
auto maxwell_to_file_with_ref(const std::string& path,                   //
                              FEx&& Ex, FEy&& Ey, FEz&& Ez,              //
                              FHx&& Hx, FHy&& Hy, FHz&& Hz,              //
                              REx&& Ex_ref, REy&& Ey_ref, REz&& Ez_ref,  //
                              RHx&& Hx_ref, RHy&& Hy_ref, RHz&& Hz_ref   //
                              ) -> void {
    constexpr auto res_x = 49;
    constexpr auto res_y = 49;
    constexpr auto res_z = 49;
    auto extent = fmt::format("0 {} 0 {} 0 {}", res_x, res_y, res_z);
    auto const rx = ads::interval{0, 2};
    auto const ry = ads::interval{0, 2};
    auto const rz = ads::interval{0, 2};

    auto const for_all_points = [&](auto&& fun) {
        for (auto z : ads::evenly_spaced(rz, res_z)) {
            for (auto y : ads::evenly_spaced(ry, res_y)) {
                for (auto x : ads::evenly_spaced(rx, res_x)) {
                    const auto X = ads::point3_t{x, y, z};
                    fun(X);
                }
            }
        }
    };

    auto out = fmt::output_file(path);
    out.print("<?xml version=\"1.0\"?>\n");
    out.print("<VTKFile type=\"ImageData\" version=\"0.1\">\n");
    out.print("  <ImageData WholeExtent=\"{}\" origin=\"0 0 0\" spacing=\"1 1 1\">\n", extent);
    out.print("    <Piece Extent=\"{}\">\n", extent);
    out.print("      <PointData Scalars=\"r\" Vectors=\"E\">\n", extent);

    out.print("        <DataArray Name=\"E\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) { out.print("{:.7} {:.7} {:.7}\n", Ex(X), Ey(X), Ez(X)); });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"H\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) { out.print("{:.7} {:.7} {:.7}\n", Hx(X), Hy(X), Hz(X)); });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"E spherical\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) {
        auto const [r, theta, phi] = spherical(X, {Ex(X), Ey(X), Ez(X)});
        out.print("{:.7} {:.7} {:.7}\n", r, theta, phi);
    });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"H spherical\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) {
        auto const [r, theta, phi] = spherical(X, {Hx(X), Hy(X), Hz(X)});
        out.print("{:.7} {:.7} {:.7}\n", r, theta, phi);
    });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"E-ref\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points(
        [&](auto X) { out.print("{:.7} {:.7} {:.7}\n", Ex_ref(X), Ey_ref(X), Ez_ref(X)); });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"H-ref\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points(
        [&](auto X) { out.print("{:.7} {:.7} {:.7}\n", Hx_ref(X), Hy_ref(X), Hz_ref(X)); });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"E-ref spherical\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) {
        auto const [r, theta, phi] = spherical(X, {Ex_ref(X), Ey_ref(X), Ez_ref(X)});
        out.print("{:.7} {:.7} {:.7}\n", r, theta, phi);
    });
    out.print("        </DataArray>\n");

    out.print("        <DataArray Name=\"H-ref spherical\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for_all_points([&](auto X) {
        auto const [r, theta, phi] = spherical(X, {Hx_ref(X), Hy_ref(X), Hz_ref(X)});
        out.print("{:.7} {:.7} {:.7}\n", r, theta, phi);
    });
    out.print("        </DataArray>\n");

    out.print("      </PointData>\n");
    out.print("    </Piece>\n");
    out.print("  </ImageData>\n");
    out.print("</VTKFile>\n");
}

#endif  // MAXWELL_UTILS_HPP
