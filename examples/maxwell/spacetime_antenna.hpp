// SPDX-FileCopyrightText: 2015 - 2021 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef MAXWELL_ANTENNA_HPP
#define MAXWELL_ANTENNA_HPP

#include <array>
#include <cmath>
#include <vector>

#include "ads/bspline/bspline.hpp"
#include "ads/quad/gauss.hpp"
#include "ads/util/function_value.hpp"
#include "spaces.hpp"
#include "state.hpp"

class antenna {
private:
    double omega;
    double tau;
    double J0;

    using point_type = std::array<double, 4>;
    using index_type = std::array<int, 3>;
    using index_1d_iter_type = boost::counting_iterator<int>;
    using index_iter_type = ads::util::iter_product3<index_1d_iter_type, index_type>;
    using index_range = boost::iterator_range<index_iter_type>;
    using value_type = ads::function_value_3d;
    using value_type4 = ads::function_value_4d;

public:
    antenna(double omega, double tau, double J0 = 1)
    : omega{omega}
    , tau{tau}
    , J0{J0} { }

    // auto apply_forcing(double t, state& rhs, space const& V) -> void {
    //     auto const l = 0.1 / 5;
    //     auto const x0 = point_type{1.0, 1.0, 1.0 - l / 2};
    //     auto const x1 = point_type{1.0, 1.0, 1.0 + l / 2};
    //     auto const len = dist(x0, x1);

    //     auto const q = 20;
    //     auto const points = quad_points(x0, x1, q);
    //     auto const* weigths = ads::quad::gauss::Ws[q];
    //     auto const scale = len / 2;

    //     for (int i = 0; i < q; ++i) {
    //         auto const& x = points[i];
    //         auto const W = weigths[i] * scale;
    //         auto const F = forcing(t, x);

    //         for (auto const e : elements(V.x, V.y, V.z)) {
    //             if (!is_inside(x, e, V))
    //                 continue;

    //             for (auto const a : dofs_on_element(e, V.x, V.y, V.z)) {
    //                 auto const v = eval_basis_at(x, a, V);
    //                 rhs.E1(a[0], a[1], a[2]) += F[0] * v.val * W;
    //                 rhs.E2(a[0], a[1], a[2]) += F[1] * v.val * W;
    //                 rhs.E3(a[0], a[1], a[2]) += F[2] * v.val * W;
    //             }
    //         }
    //     }
    // }

    template <typename Space, typename Out>
    auto apply_forcing_spacetime(const Space& space, Out out, double T) -> void {
        auto const l = 0.1 / 5;
        auto const x0 = point_type{1.0, 1.0, 1.0 - l / 2};
        auto const x1 = point_type{1.0, 1.0, 1.0 + l / 2};
        auto const len = dist(x0, x1);

        auto const q = 20;
        auto const points = quad_points(x0, x1, 0, T, q);
        auto const* weigths = ads::quad::gauss::Ws[q];
        auto const scale = len / 2;

        const auto& mesh = space.mesh();
        for (int i = 0; i < q; ++i) {
            auto const& x = points[i];
            auto const W = weigths[i] * scale;
            auto F = forcing(x);
            for (auto e: mesh.elements()) {
                if (!is_inside(x, mesh.element(e)))
                    continue;

                for (auto dof: space.dofs(e)) {
                    auto const v = eval_basis_at(x, dof, space);
                    auto const J = space.global_index(dof);
                    // if (F[2] * v.val * W > 1) {
                    //     std::cout << F[2] * v.val * W << " | F: " << F[2]
                    //     << " | v: " << v.val << " | W: " << W << " " << "\n";
                    // }
                    out(J, F[2] * v.val * W);
                }
            }
        }
    }


private:
    // index_range elements(const ads::dimension& x, const ads::dimension& y,
    //                      const ads::dimension& z) const {
    //     return ads::util::product_range<index_type>(x.element_indices(), y.element_indices(),
    //                                                 z.element_indices());
    // }

    // index_range dofs_on_element(index_type e, const ads::dimension& x, const ads::dimension& y,
    //                             const ads::dimension& z) const {
    //     auto rx = x.basis.dof_range(e[0]);
    //     auto ry = y.basis.dof_range(e[1]);
    //     auto rz = z.basis.dof_range(e[2]);
    //     return ads::util::product_range<index_type>(rx, ry, rz);
    // }

    auto quad_points(point_type const x0, point_type const x1, int q) -> std::vector<point_type> {
        auto points = std::vector<point_type>(q);
        for (int i = 0; i < q; ++i) {
            const auto t = ads::quad::gauss::Xs[q][i];
            const auto s = (t + 1) / 2;
            const auto x = ads::lerp(s, x0[0], x1[0]);
            const auto y = ads::lerp(s, x0[1], x1[1]);
            const auto z = ads::lerp(s, x0[2], x1[2]);
            points[i] = point_type{x, y, z};
        }
        return points;
    }

    auto quad_points(point_type const x0, point_type const x1, double t0, double t1, int q) -> std::vector<point_type> {
        auto points = std::vector<point_type>(q);
        for (int i = 0; i < q; ++i) {
            const auto t = ads::quad::gauss::Xs[q][i];
            const auto s = (t + 1) / 2;
            const auto x = ads::lerp(s, x0[0], x1[0]);
            const auto y = ads::lerp(s, x0[1], x1[1]);
            const auto z = ads::lerp(s, x0[2], x1[2]);
            const auto w = ads::lerp(s, t0,    t1);
            points[i] = point_type{x, y, z, w};
        }
        return points;
    }

    auto dist(point_type a, point_type b) const -> double {
        auto const dx = a[0] - b[0];
        auto const dy = a[1] - b[1];
        auto const dz = a[2] - b[2];
        return std::hypot(dx, dy, dz);
    }

    // auto dist(point_type a, point_type b) const -> double {
    //     auto const dx = a[0] - b[0];
    //     auto const dy = a[1] - b[1];
    //     auto const dz = a[2] - b[2];
    //     auto const dw = a[3] - b[3];
    //     return std::sqrt(std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2) + std::pow(dw, 2));
    // }

    // auto is_inside(point_type x, index_type e, space const& V) const -> bool {
    //     return V.x.B.points[e[0]] <= x[0] && x[0] <= V.x.B.points[e[0] + 1]
    //         && V.y.B.points[e[1]] <= x[1] && x[1] <= V.y.B.points[e[1] + 1]
    //         && V.z.B.points[e[2]] <= x[2] && x[2] <= V.z.B.points[e[2] + 1];
    // }

    template <typename Element>
    auto is_inside(point_type x, Element const& e) const -> bool {
        return e.span_x.left <= x[0] && x[0] <= e.span_x.right
            && e.span_y.left <= x[1] && x[1] <= e.span_y.right
            && e.span_z.left <= x[2] && x[2] <= e.span_z.right;
    }

    template <typename Space, typename Dof>
    auto eval_basis_at(point_type p, Dof dof, Space const& space) const -> value_type4 {
        const auto spanx = ads::bspline::find_span(p[0], space.space_x().basis());
        const auto spany = ads::bspline::find_span(p[1], space.space_y().basis());
        const auto spanz = ads::bspline::find_span(p[2], space.space_z().basis());
        const auto spanw = ads::bspline::find_span(p[3], space.space_w().basis());

        // ads::bspline::eval_ders_ctx cx{space.space_x().degree(), 1};
        // ads::bspline::eval_ders_ctx cy{space.space_y().degree(), 1};
        // ads::bspline::eval_ders_ctx cz{space.space_z().degree(), 1};
        // ads::bspline::eval_ders_ctx cw{space.space_w().degree(), 1};
        auto cx = ads::bspline::eval_ctx{space.space_x().degree()};
        auto cy = ads::bspline::eval_ctx{space.space_y().degree()};
        auto cz = ads::bspline::eval_ctx{space.space_z().degree()};
        auto cw = ads::bspline::eval_ctx{space.space_w().degree()};

        // double** bvx = cx.basis_vals();
        // double** bvy = cy.basis_vals();
        // double** bvz = cz.basis_vals();
        // double** bvw = cw.basis_vals();

        auto val_x = ads::bspline_basis_values{20, space.space_x().dofs_per_element(), 1};
        auto val_y = ads::bspline_basis_values{20, space.space_x().dofs_per_element(), 1};
        auto val_z = ads::bspline_basis_values{20, space.space_x().dofs_per_element(), 1};
        auto val_w = ads::bspline_basis_values{20, space.space_x().dofs_per_element(), 1};

        auto* const bvx = val_x.point_buffer(1);
        auto* const bvy = val_y.point_buffer(1);
        auto* const bvz = val_z.point_buffer(1);
        auto* const bvw = val_w.point_buffer(1);

        eval_basis_with_derivatives(spanx, p[0], space.space_x().basis(), bvx, 1, cx);
        eval_basis_with_derivatives(spany, p[1], space.space_y().basis(), bvy, 1, cy);
        eval_basis_with_derivatives(spanz, p[2], space.space_z().basis(), bvz, 1, cz);
        eval_basis_with_derivatives(spanw, p[3], space.space_w().basis(), bvw, 1, cw);

        int offsetx = spanx - space.space_x().degree();
        int offsety = spany - space.space_y().degree();
        int offsetz = spanz - space.space_z().degree();
        int offsetw = spanw - space.space_w().degree();

        int ix = std::get<0>(dof) - offsetx;
        int iy = std::get<1>(dof) - offsety;
        int iz = std::get<2>(dof) - offsetz;
        int iw = std::get<3>(dof) - offsetw;

        auto value = bvx[0][ix] * bvy[0][iy] * bvz[0][iz] * bvw[0][iw];
        auto dx    = bvx[1][ix] * bvy[0][iy] * bvz[0][iz]  * bvw[0][iw];
        auto dy    = bvx[0][ix] * bvy[1][iy] * bvz[0][iz]  * bvw[0][iw];
        auto dz    = bvx[0][ix] * bvy[0][iy] * bvz[1][iz]  * bvw[0][iw];
        auto dw    = bvx[0][ix] * bvy[0][iy] * bvz[0][iz]  * bvw[1][iw];

        return {value, dx, dy, dz, dw};
    }


    // auto eval_basis_at(point_type p, index_type dof, space const& V) const -> value_type {
    //     const auto spanx = ads::bspline::find_span(p[0], V.x.B);
    //     const auto spany = ads::bspline::find_span(p[1], V.y.B);
    //     const auto spanz = ads::bspline::find_span(p[2], V.z.B);

    //     ads::bspline::eval_ders_ctx cx{V.x.p, 1};
    //     ads::bspline::eval_ders_ctx cy{V.y.p, 1};
    //     ads::bspline::eval_ders_ctx cz{V.z.p, 1};

    //     double** bvx = cx.basis_vals();
    //     double** bvy = cy.basis_vals();
    //     double** bvz = cz.basis_vals();

    //     eval_basis_with_derivatives(spanx, p[0], V.x.B, bvx, 1, cx);
    //     eval_basis_with_derivatives(spany, p[1], V.y.B, bvy, 1, cy);
    //     eval_basis_with_derivatives(spanz, p[2], V.z.B, bvz, 1, cz);

    //     int offsetx = spanx - V.x.p;
    //     int offsety = spany - V.y.p;
    //     int offsetz = spanz - V.z.p;

    //     int ix = dof[0] - offsetx;
    //     int iy = dof[1] - offsety;
    //     int iz = dof[2] - offsetz;

    //     auto value = bvx[0][ix] * bvy[0][iy] * bvz[0][iz];
    //     auto dx = bvx[1][ix] * bvy[0][iy] * bvz[0][iz];
    //     auto dy = bvx[0][ix] * bvy[1][iy] * bvz[0][iz];
    //     auto dz = bvx[0][ix] * bvy[0][iy] * bvz[1][iz];

    //     return {value, dx, dy, dz};
    // }

    auto forcing(point_type x) -> point_type {
        auto const s = excitation(x[3]) * std::sin(omega * x[3]);
        return {0, 0, J0 * s, 0};
    }

    // auto forcing(double t, point_type x) -> point_type {
    //     auto const s = excitation(t) * std::sin(omega * t);
    //     return {0, 0, J0 * s, 0};
    // }

    auto excitation(double t) const -> double {
        return t > 0 ? (1 - std::exp(-t / tau)) : 0;
        // return t > 0 ? std::exp(-0.5 * std::pow((t - t0) / tau, 2)) : 0;
    }
};

#endif  // MAXWELL_ANTENNA_HPP
