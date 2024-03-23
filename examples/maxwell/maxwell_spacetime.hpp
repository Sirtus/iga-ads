// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <fmt/os.h>
#include <lyra/lyra.hpp>

#include "ads/experimental/all.hpp"
#include "ads/experimental/horrible_sparse_matrix.hpp"
#include "ads/experimental/space_factory.hpp"

#include "spacetime_plane_wave_problem.hpp"

template <typename U>
auto save_heat_to_file(std::string const& path, double time, U const& u) -> void {
    constexpr auto res = 50;
    auto extent = fmt::format("0 {0} 0 {0} 0 {0}", res);
    auto spacing = fmt::format("{0} {0} {0}", 1.0 / res);

    auto out = fmt::output_file(path);
    out.print("<?xml version=\"1.0\"?>\n");
    out.print("<VTKFile type=\"ImageData\" version=\"0.1\">\n");
    out.print("  <ImageData WholeExtent=\"{}\" Origin=\"0 0 0\" Spacing=\"{}\">\n", extent,
              spacing);
    out.print("    <Piece Extent=\"{}\">\n", extent);
    out.print("      <PointData Scalars=\"u\">\n");

    out.print("        <DataArray Name=\"u\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"1\">\n");
    for (auto t : ads::evenly_spaced(0.0, time, res)) {
        for (auto y : ads::evenly_spaced(0.0, 1.0, res)) {
            for (auto x : ads::evenly_spaced(0.0, 1.0, res)) {
                const auto X = ads::point3_t{x, y, t};
                out.print("{:.7}\n", u(X));
            }
        }
    }
    out.print("        </DataArray>\n");
    out.print("      </PointData>\n");
    out.print("    </Piece>\n");
    out.print("  </ImageData>\n");
    out.print("</VTKFile>\n");
}

template <typename U>
auto save_heat_to_file4D(std::string const& path, double time, U const& u) -> void {
    constexpr auto res = 50;
    auto extent = fmt::format("0 {0} 0 {0} 0 {0} 0 {0}", res);
    auto spacing = fmt::format("{0} {0} {0} {0}", 1.0 / res);

    auto u_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return u({x, y, z, t});
      };
    };


    int i = 0;
    for (auto t : ads::evenly_spaced(0.0, time, res)) {
        save_heat_to_file(fmt::format("output_{}.vti", i), time, u_at_fixed_t(t)); 
        ++i;
    }
}


constexpr double pi = M_PI;


auto maxwell_spacetime_main(int /*argc*/, char* /*argv*/[]) -> void {
    auto const elems = 4;
    auto const x_elems = 2;
    auto const y_elems = 2;
    auto const z_elems = 32;
    auto const t_elems = 16;
    
    auto const p = 1;
    auto const c = 0;
    auto const T = 1;
    auto const eps = 1.0;
    auto const mu  = 4.0 * pi * 1.0e-7;
    auto const mu_inv = 1/mu;
    
    // auto const s = 1 * 1.0;
    auto const H_0_x = 0.0;
    auto const H_0_y = 0.0;
    auto const H_0_z = 0.0;

    auto const J = 1.0;

    auto const xs = ads::evenly_spaced(0.0, 1.0, x_elems);
    auto const ys = ads::evenly_spaced(0.0, 1.0, y_elems);
    auto const zs = ads::evenly_spaced(0.0, 1.0, z_elems);
    auto const ts = ads::evenly_spaced(0.0, T, t_elems);
    // auto const ts = ads::evenly_spaced(0.0, T, 32);

    auto const bx = ads::make_bspline_basis(xs, p, c);
    auto const by = ads::make_bspline_basis(ys, p, c);
    auto const bz = ads::make_bspline_basis(zs, p, c);
    auto const bt = ads::make_bspline_basis(ts, p, c);

    auto mesh = ads::regular_mesh4{xs, ys, zs, ts};
    auto quad = ads::quadrature4{&mesh, std::max(p + 1, 2)};

    auto spaces = ads::space_factory{};

    auto const U  = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Vx = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Vy = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Vz = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);

    auto const n = spaces.dim();
    fmt::print("Dimension: {}\n", n);

    auto F = std::vector<double>(n);
    auto problem = ads::mumps::problem{F.data(), n};
    auto solver = ads::mumps::solver{};

    auto mat = ads::horrible_sparse_matrix{};
    auto M = [&mat](int row, int col, double val) { mat(row, col) += val; };
    auto rhs = [&F](int row, double val) { F[row] += val; };

    fmt::print("Assembling matrix\n");
    // assemble(U, quad, M, [](auto u, auto v, auto /*x*/) { return u.dw * v.val; });
    // assemble(Vx, quad, M, [](auto sx, auto tx, auto /*x*/) { return sx.val * tx.val; });
    // assemble(Vy, quad, M, [](auto sy, auto ty, auto /*x*/) { return sy.val * ty.val; });
    // assemble(Vz, quad, M, [](auto sz, auto tz, auto /*x*/) { return sz.val * tz.val; });
    // assemble(U, Vx, quad, M,
    //          [=](auto u, auto tx, auto /*x*/) { return (eps * u.dx - beta_x * u.val) * tx.val; }); // operator L
    // assemble(U, Vy, quad, M,
    //          [=](auto u, auto ty, auto /*x*/) { return (eps * u.dy - beta_y * u.val) * ty.val; }); // operator L
    // assemble(U, Vz, quad, M,
    //          [=](auto u, auto tz, auto /*x*/) { return (eps * u.dz - beta_z * u.val) * tz.val; }); // operator L
    // assemble(Vx, U, quad, M, [](auto sx, auto v, auto /*x*/) { return sx.dx * v.val; });
    // assemble(Vy, U, quad, M, [](auto sy, auto v, auto /*x*/) { return sy.dy * v.val; });
    // assemble(Vz, U, quad, M, [](auto sz, auto v, auto /*x*/) { return sz.dz * v.val; });

    ///
    // Ex
    assemble(Vx, quad, M,    [eps](auto sx, auto tx, auto /*x*/) {return  -(eps    * sx.dw * tx.dw); });
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dy * tx.dy); });
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dz * tx.dz); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dx * ty.dy); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dx * tz.dz); });

    // Ey
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dy * tx.dx); });
    assemble(Vy, quad, M,    [eps](auto sy, auto ty, auto /*x*/) {return  -(eps    * sy.dw * ty.dw); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dy * ty.dy); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dz * ty.dz); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dy * tz.dz); });

    // Ez
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dz * tx.dx); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dz * ty.dy); });
    assemble(Vz, quad, M,    [eps](auto sz, auto tz, auto /*x*/) {return  -(eps    * sz.dw * tz.dw); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dx * tz.dx); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dy * tz.dy); });

    fmt::print("Assembling RHS\n");
    assemble_rhs(U, quad, rhs, [](auto v, auto /*x*/) { return 0 * v.val; });

    assemble_rhs(Vx, quad, rhs, [](auto v, auto /*x*/) { return 0; }); //psi

    fmt::print("Collecting BC\n");
    auto is_fixed = std::vector<int>(n);
    for (auto const dof : U.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary
        fix = fix || ix == 0 || ix == U.space_x().dof_count() - 1;
        fix = fix || iy == 0 || iy == U.space_y().dof_count() - 1;
        fix = fix || iz == 0 || iz == U.space_z().dof_count() - 1;

        if (fix) {
            is_fixed[U.global_index(dof)] = 1;
        }
    }

    fmt::print("Applying BC\n");
    for (auto const e : mesh.elements()) {
        for (auto const i : U.dofs(e)) {
            auto const I = U.global_index(i);
            if (is_fixed[I] == 1) {
                for (auto const j : U.dofs(e)) {
                    auto const J = U.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Vx.dofs(e)) {
                    auto const J = Vx.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Vy.dofs(e)) {
                    auto const J = Vy.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Vz.dofs(e)) {
                    auto const J = Vz.global_index(j);
                    mat(I, J) = 0;
                }
                mat(I, I) = 1;
                F[I] = 0;
            }
        }
    }

    //F[U.global_index({elems / 2, elems / 2, 0})] = 1;
    //F[U.global_index({elems / 2 + 1, elems / 2, 0})] = 1;
    //F[U.global_index({elems / 2 + 1, elems / 2 + 1, 0})] = 1;
    //F[U.global_index({elems / 2, elems / 2 + 1, 0})] = 1;


    F[U.global_index({elems / 2, elems / 2, elems / 2, 0})] = 1;

    F[U.global_index({elems / 2 + 1, elems / 2, elems / 2, 0})] = 1;
    F[U.global_index({elems / 2 + 1, elems / 2 + 1, elems / 2, 0})] = 1;
    F[U.global_index({elems / 2 + 1, elems / 2, elems / 2 + 1, 0})] = 1;

    F[U.global_index({elems / 2, elems / 2 + 1, elems / 2, 0})] = 1;
    F[U.global_index({elems / 2, elems / 2 + 1, elems / 2 + 1, 0})] = 1;

    F[U.global_index({elems / 2, elems / 2, elems / 2 + 1, 0})] = 1;

    F[U.global_index({elems / 2 + 1, elems / 2 + 1, elems / 2 + 1, 0})] = 1;

    mat.mumpsify(problem);
    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());

    fmt::print("Solving\n");
    solver.solve(problem);

    auto u = ads::bspline_function4(&U, F.data());
    fmt::print("Saving\n");
    save_heat_to_file4D("dup-full.vti", T, u);
    fmt::print("Done\n");
}