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
#include "plane_wave_problem.hpp"
#include "problems.hpp"

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


template <typename FEx, typename FEy, typename FEz>
auto save_maxwell_to_file(std::string const& path, double time, FEx const& fex, FEy const& fey, FEz const& fez) -> void {
    constexpr auto res_x = 49;
    constexpr auto res_y = 49;
    constexpr auto res_z = 49;
    auto extent = fmt::format("0 {} 0 {} 0 {}", res_x, res_y, res_z);
    auto spacing = fmt::format("{0} {0} {0}", 1.0 / res_x, 1.0 / res_y, 1.0 / res_z);

    auto out = fmt::output_file(path);
    out.print("<?xml version=\"1.0\"?>\n");
    out.print("<VTKFile type=\"ImageData\" version=\"0.1\">\n");
    out.print("  <ImageData WholeExtent=\"{}\" Origin=\"0 0 0\" Spacing=\"{}\">\n", extent,
              spacing);
    out.print("    <Piece Extent=\"{}\">\n", extent);
    out.print("      <PointData Vectors=\"E\">\n");

    out.print("        <DataArray Name=\"E\" type=\"Float32\" format=\"ascii\" "
              "NumberOfComponents=\"3\">\n");
    for (auto z : ads::evenly_spaced(0.0, 1.0, res_z)) {
        for (auto y : ads::evenly_spaced(0.0, 1.0, res_y)) {
            for (auto x : ads::evenly_spaced(0.0, 1.0, res_x)) {
                const auto X = ads::point3_t{x, y, z};
                out.print("{:.7} {:.7} {:.7}\n", fex(X), fey(X), fez(X));
            }
        }
    }
    out.print("        </DataArray>\n");
    out.print("      </PointData>\n");
    out.print("    </Piece>\n");
    out.print("  </ImageData>\n");
    out.print("</VTKFile>\n");
}

template <typename FEx, typename FEy, typename FEz>
auto save_maxwell_to_file4D(std::string const& path, double time, FEx const& fex, FEy const& fey, FEz const& fez) -> void {
    constexpr auto res_time = 100;

    auto fex_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fex({x, y, z, t});
      };
    };

    auto fey_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fey({x, y, z, t});
      };
    };

    auto fez_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fez({x, y, z, t});
      };
    };


    int i = 0;
    for (auto t : ads::evenly_spaced(0.0, time, res_time)) {
        save_maxwell_to_file(fmt::format("output_{}.vti", i), time, fex_at_fixed_t(t), fey_at_fixed_t(t), fez_at_fixed_t(t)); 
        ++i;
    }
}


template <typename FEx, typename FEy, typename FEz>
auto save_init_maxwell_to_file(std::string const& path, double time, FEx const& fex, FEy const& fey, FEz const& fez) -> void {
    constexpr auto res_time = 50;

    auto fex_at_fixed_t = [&]() {
      return [&](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fex({x, y, z});
      };
    };

    auto fey_at_fixed_t = [&]() {
      return [&](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fey({x, y, z});
      };
    };

    auto fez_at_fixed_t = [&]() {
      return [&](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fez({x, y, z});
      };
    };

    save_maxwell_to_file(fmt::format("init_output.vti"), time, fex_at_fixed_t(), fey_at_fixed_t(), fez_at_fixed_t()); 
}



constexpr double pi = M_PI;


auto maxwell_spacetime_main(int /*argc*/, char* /*argv*/[]) -> void {
    auto const elems = 4;
    auto const x_elems = 1;
    auto const y_elems = 1;
    auto const z_elems = 100;
    auto const t_elems = 16;
    
    auto const p = 1;
    auto const c = 0;

    auto const eps = 8.854e-12;
    auto const mu  = 12.556e-7;
    auto const c0  = 1 / std::sqrt(eps * mu);
    auto const T = z_elems / c0;
    auto const mu_inv = 1/mu;
    


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
    auto plane_wave_problem = spacetime_plane_wave_problem{};
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
    assemble(U , quad, M,    [eps](auto sx, auto tx, auto /*x*/) {return  -(eps    * sx.dw * tx.dw); });
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dy * tx.dy); });
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dz * tx.dz); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dx * ty.dy); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dx * tz.dz); });

    // Ey
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dy * tx.dx); });
    assemble(U , quad, M,    [eps](auto sy, auto ty, auto /*x*/) {return  -(eps    * sy.dw * ty.dw); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dy * ty.dy); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dz * ty.dz); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dy * tz.dz); });

    // Ez
    assemble(Vx, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dz * tx.dx); });
    assemble(Vy, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dz * ty.dy); });
    assemble(U , quad, M,    [eps](auto sz, auto tz, auto /*x*/) {return  -(eps    * sz.dw * tz.dw); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dx * tz.dx); });
    assemble(Vz, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dy * tz.dy); });

    fmt::print("Assembling RHS\n");
    assemble_rhs(U, quad, rhs, [=](auto v, auto x) { return (-1) * plane_wave_problem.J1({std::get<0>(x),std::get<1>(x),std::get<2>(x)}, std::get<3>(x)).dw * v.val; });
    assemble_rhs(U, quad, rhs, [=](auto v, auto x) { return (-1) * plane_wave_problem.J2({std::get<0>(x),std::get<1>(x),std::get<2>(x)}, std::get<3>(x)).dw * v.val; });
    assemble_rhs(U, quad, rhs, [=](auto v, auto x) { return (-1) * plane_wave_problem.J3({std::get<0>(x),std::get<1>(x),std::get<2>(x)}, std::get<3>(x)).dw * v.val; });
    // assemble_rhs()

    assemble_rhs(Vx, quad, rhs, [](auto v, auto /*x*/) { return 1; }); //psi
    assemble_rhs(Vy, quad, rhs, [](auto v, auto /*x*/) { return 1; }); //psi
    assemble_rhs(Vz, quad, rhs, [](auto v, auto /*x*/) { return 1; }); //psi

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
                    mat(I, J) = 1;
                }
                for (auto const j : Vx.dofs(e)) {
                    auto const J = Vx.global_index(j);
                    mat(I, J) = 1;
                }
                for (auto const j : Vy.dofs(e)) {
                    auto const J = Vy.global_index(j);
                    mat(I, J) = 1;
                }
                for (auto const j : Vz.dofs(e)) {
                    auto const J = Vz.global_index(j);
                    mat(I, J) = 1;
                }
                // mat(I, I) = 1;
                F[I] = 0;
            }
        }
    }

    //F[U.global_index({elems / 2, elems / 2, 0})] = 1;
    //F[U.global_index({elems / 2 + 1, elems / 2, 0})] = 1;
    //F[U.global_index({elems / 2 + 1, elems / 2 + 1, 0})] = 1;
    //F[U.global_index({elems / 2, elems / 2 + 1, 0})] = 1;


    // F[U.global_index({elems / 2, elems / 2, elems / 2, 0})] = 1;

    // F[U.global_index({elems / 2 + 1, elems / 2, elems / 2, 0})] = 1;
    // F[U.global_index({elems / 2 + 1, elems / 2 + 1, elems / 2, 0})] = 1;
    // F[U.global_index({elems / 2 + 1, elems / 2, elems / 2 + 1, 0})] = 1;

    // F[U.global_index({elems / 2, elems / 2 + 1, elems / 2, 0})] = 1;
    // F[U.global_index({elems / 2, elems / 2 + 1, elems / 2 + 1, 0})] = 1;

    // F[U.global_index({elems / 2, elems / 2, elems / 2 + 1, 0})] = 1;

    // F[U.global_index({elems / 2 + 1, elems / 2 + 1, elems / 2 + 1, 0})] = 1;

    mat.mumpsify(problem);
    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());

    fmt::print("Solving\n");
    solver.solve(problem);

    auto u = ads::bspline_function4(&U, F.data());
    fmt::print("Saving\n");
    save_heat_to_file4D("dup-full.vti", T, u);
    fmt::print("Done\n");
}

auto maxwell_spacetime_main_eigen(int /*argc*/, char* /*argv*/[]) -> void {

    auto const x_elems = 6;
    auto const y_elems = 6;
    auto const z_elems = 6;
    auto const t_elems = 10;
    
    auto const p = 1;
    auto const c = 0;

    auto const eps = 8.854e-12;
    auto const mu  = 12.556e-7;
    auto const c0  = 1 / std::sqrt(eps * mu);
    auto const T = z_elems / c0;
    auto const mu_inv = 1/mu;
    


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

    auto const Ex = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Ey = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Ez = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);

    auto const n = spaces.dim();
    fmt::print("Dimension: {}\n", n);

    auto F = std::vector<double>(n);
    auto problem = ads::eigen::problem{F.data(), n};
    auto pw_problem = spacetime_plane_wave_problem{};
    auto ml_problem = maxwell_manufactured1{1,1};
    auto solver = ads::eigen::solver{1000};


    // L2 projection - begin
    fmt::print("L2 Projection:\n");
    fmt::print("*\tE(x,t=0).x\n");

    auto mesh_init = ads::regular_mesh3{xs, ys, zs};
    auto quad_init = ads::quadrature3{&mesh_init, std::max(p + 1, 2)};
    auto space_init = ads::space3{&mesh_init, bx, by, bz};

    auto Fx_init = std::vector<double>(n);
    auto problem_init = ads::mumps::problem{Fx_init.data(), n};
    auto solver_init = ads::mumps::solver{};

    auto out_init = [&problem_init](int row, int col, double val) {
        if (val != 0) {
            problem_init.add(row + 1, col + 1, val);
        }
    };
    auto rhs_init_x = [&Fx_init](int J, double val) { Fx_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_x, [&ml_problem](auto v, auto x) { return ml_problem.E1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);

    fmt::print("*\tE(x,t=0).y\n");

    auto Fy_init = std::vector<double>(n);
    problem_init = ads::mumps::problem{Fy_init.data(), n};
    auto rhs_init_y = [&Fy_init](int J, double val) { Fy_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_y, [&ml_problem](auto v, auto x) { return ml_problem.E1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);

    fmt::print("*\tE(x,t=0).z\n");
    auto Fz_init = std::vector<double>(n);
    problem_init = ads::mumps::problem{Fz_init.data(), n};
    auto rhs_init_z = [&Fz_init](int J, double val) { Fz_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_z, [&ml_problem](auto v, auto x) { return ml_problem.E1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);



    // L2 projection - end

    auto mat = ads::horrible_sparse_matrix{};
    auto M = [&mat](int row, int col, double val) { mat(row, col) += val; };
    auto rhs = [&F](int row, double val) { F[row] += val; };

    fmt::print("Assembling matrix\n");

    // Ex
    assemble(Ex, quad, M,    [eps](auto sx, auto tx, auto /*x*/) {return  -(eps    * sx.dw * tx.dw); });
    assemble(Ex, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dy * tx.dy); });
    assemble(Ex, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dz * tx.dz); });
    assemble(Ey, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dx * ty.dy); });
    assemble(Ez, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dx * tz.dz); });

    // Ey
    assemble(Ex, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dy * tx.dx); });
    assemble(Ey, quad, M,    [eps](auto sy, auto ty, auto /*x*/) {return  -(eps    * sy.dw * ty.dw); });
    assemble(Ey, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dy * ty.dy); });
    assemble(Ey, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dz * ty.dz); });
    assemble(Ez, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dy * tz.dz); });

    // Ez
    assemble(Ex, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dz * tx.dx); });
    assemble(Ey, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dz * ty.dy); });
    assemble(Ez, quad, M,    [eps](auto sz, auto tz, auto /*x*/) {return  -(eps    * sz.dw * tz.dw); });
    assemble(Ez, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dx * tz.dx); });
    assemble(Ez, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dy * tz.dy); });

    fmt::print("Assembling RHS\n");
    assemble_rhs(Ex, quad, rhs, [=](auto v, auto /*x*/) { return 0 * v.val; });
    assemble_rhs(Ey, quad, rhs, [=](auto v, auto /*x*/) { return 0 * v.val; });
    assemble_rhs(Ez, quad, rhs, [=](auto v, auto /*x*/) { return 0 * v.val; });


    fmt::print("Collecting BC\n");
    auto is_fixed = std::vector<int>(n);
    for (auto const dof : Ex.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary - on y = 0, y = 1, z = 0, z = 1 we force Ex = 0
        fix = fix || iy == 0 || iy == Ex.space_y().dof_count() - 1;
        fix = fix || iz == 0 || iz == Ex.space_z().dof_count() - 1;

        if (fix) {
            is_fixed[Ex.global_index(dof)] = 1;
        }
    }

    for (auto const dof : Ey.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary - on x = 0, x = 1, z = 0, z = 1 we force Ey = 0
        fix = fix || ix == 0 || ix == Ey.space_x().dof_count() - 1;
        fix = fix || iz == 0 || iz == Ey.space_z().dof_count() - 1;

        if (fix) {
            is_fixed[Ey.global_index(dof)] = 1;
        }
    }

    for (auto const dof : Ez.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary - on x = 0, x = 1, y = 0, y = 1 we force Ez = 0
        fix = fix || ix == 0 || ix == Ez.space_x().dof_count() - 1;
        fix = fix || iy == 0 || iy == Ez.space_y().dof_count() - 1;

        if (fix) {
            is_fixed[Ez.global_index(dof)] = 1;
        }
    }

    fmt::print("Applying BC\n");
    for (auto const e : mesh.elements()) {
        // for Ex
        for (auto const i : Ex.dofs(e)) {
            auto const I = Ex.global_index(i);
            if (is_fixed[I] == 1) {
                // Matrix
                for (auto const j : Ex.dofs(e)) {
                    auto const J = Ex.global_index(j);
                    mat(I, J) = 1; // main diagonal
                }
                for (auto const j : Ey.dofs(e)) {
                    auto const J = Ey.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Ez.dofs(e)) {
                    auto const J = Ez.global_index(j);
                    mat(I, J) = 0;
                }
                // RHS
                auto const [ix, iy, iz, it] = i;
                if (it == 0) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F[I] = Fx_init[I_init];
                } else { // boundary
                    F[I] = 0;
                }
            }
        }

        // for Ey
        for (auto const i : Ey.dofs(e)) {
            auto const I = Ey.global_index(i);
            if (is_fixed[I] == 1) {
                // Matrix
                for (auto const j : Ey.dofs(e)) {
                    auto const J = Ey.global_index(j);
                    mat(I, J) = 1; // main diagonal
                }
                for (auto const j : Ex.dofs(e)) {
                    auto const J = Ex.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Ez.dofs(e)) {
                    auto const J = Ez.global_index(j);
                    mat(I, J) = 0;
                }
                // RHS
                auto const [ix, iy, iz, it] = i;
                if (it == 0) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F[I] = Fy_init[I_init];
                } else { // boundary
                    F[I] = 0;
                }
            }
        }

        // for Ez
        for (auto const i : Ez.dofs(e)) {
            auto const I = Ez.global_index(i);
            if (is_fixed[I] == 1) {
                // Matrix
                for (auto const j : Ez.dofs(e)) {
                    auto const J = Ez.global_index(j);
                    mat(I, J) = 1; // main diagonal
                }
                for (auto const j : Ey.dofs(e)) {
                    auto const J = Ey.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Ex.dofs(e)) {
                    auto const J = Ex.global_index(j);
                    mat(I, J) = 0;
                }
                // RHS
                auto const [ix, iy, iz, it] = i;
                if (it == 0) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F[I] = Fz_init[I_init];
                    fmt::print("TEST {} {} {} {} {} {} {}\n", ix, iy, iz, Fz_init[I_init], pw_problem.E({0.001,0.2,0.004},0).x, pw_problem.E({ix,iy,iz},0).y, pw_problem.E({ix,iy,iz},0).z);
                } else { // boundary
                    F[I] = 0;
                }
            }
        }
    }

    // //F[U.global_index({elems / 2, elems / 2, 0})] = 1;
    // //F[U.global_index({elems / 2 + 1, elems / 2, 0})] = 1;
    // //F[U.global_index({elems / 2 + 1, elems / 2 + 1, 0})] = 1;
    // //F[U.global_index({elems / 2, elems / 2 + 1, 0})] = 1;


    // // F[U.global_index({elems / 2, elems / 2, elems / 2, 0})] = 1;

    // // F[U.global_index({elems / 2 + 1, elems / 2, elems / 2, 0})] = 1;
    // // F[U.global_index({elems / 2 + 1, elems / 2 + 1, elems / 2, 0})] = 1;
    // // F[U.global_index({elems / 2 + 1, elems / 2, elems / 2 + 1, 0})] = 1;

    // // F[U.global_index({elems / 2, elems / 2 + 1, elems / 2, 0})] = 1;
    // // F[U.global_index({elems / 2, elems / 2 + 1, elems / 2 + 1, 0})] = 1;

    // // F[U.global_index({elems / 2, elems / 2, elems / 2 + 1, 0})] = 1;

    // // F[U.global_index({elems / 2 + 1, elems / 2 + 1, elems / 2 + 1, 0})] = 1;

    mat.eigenify(problem);

    fmt::print("Equation system preparation\n");
    problem.prepare_data();

    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());

    fmt::print("Solving\n");
    auto eigein_result = solver.solve(problem);

    auto ex = ads::bspline_function4(&Ex, eigein_result);
    fmt::print("Saving\n");
    save_heat_to_file4D("dup-full.vti", T, ex);
    fmt::print("Done\n");
}

auto maxwell_spacetime_main_mumps(int /*argc*/, char* /*argv*/[]) -> void {

    auto const x_elems = 4;
    auto const y_elems = 4;
    auto const z_elems = 4;
    auto const t_elems = 50;
    
    auto const p = 1;
    auto const c = 0;

    auto const eps = 8.854e-12;
    auto const mu  = 12.556e-7;
    auto const c0  = 1 / std::sqrt(eps * mu);
    auto const T = 1;//z_elems / c0;
    auto const mu_inv = 1/mu;
    


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

    auto const Ex = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Ey = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);
    auto const Ez = spaces.next<ads::space4>(&mesh, bx, by, bz, bt);

    auto const n = spaces.dim();
    fmt::print("Dimension: {}\n", n);

    auto F = std::vector<double>(n);
    auto problem = ads::mumps::problem{F.data(), n};
    auto pw_problem = spacetime_plane_wave_problem{};
    auto ml_problem = maxwell_manufactured1{1,1};
    auto solver = ads::mumps::solver{};


    // L2 projection - begin
    fmt::print("L2 Projection:\n");
    fmt::print("*\tE(x,t=0).x\n");

    auto mesh_init = ads::regular_mesh3{xs, ys, zs};
    auto quad_init = ads::quadrature3{&mesh_init, std::max(p + 1, 2)};
    auto space_init = ads::space3{&mesh_init, bx, by, bz};
    auto n_init = space_init.dof_count();

    auto Fx_init = std::vector<double>(n_init);
    auto problem_init = ads::mumps::problem{Fx_init.data(), n_init};
    // auto solver_init = ads::mumps::solver{};

    auto out_init = [&problem_init](int row, int col, double val) {
        if (val != 0) {
            problem_init.add(row + 1, col + 1, val);
        }
    };
    auto rhs_init_x = [&Fx_init](int J, double val) { Fx_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_x, [&ml_problem](auto v, auto x) { return ml_problem.E1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver.solve(problem_init);

    fmt::print("*\tE(x,t=0).y\n");

    auto Fy_init = std::vector<double>(n_init);
    problem_init = ads::mumps::problem{Fy_init.data(), n_init};
    auto rhs_init_y = [&Fy_init](int J, double val) { Fy_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_y, [&ml_problem](auto v, auto x) { return ml_problem.E2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver.solve(problem_init);

    fmt::print("*\tE(x,t=0).z\n");
    auto Fz_init = std::vector<double>(n_init);
    problem_init = ads::mumps::problem{Fz_init.data(), n_init};
    auto rhs_init_z = [&Fz_init](int J, double val) { Fz_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_z, [&ml_problem](auto v, auto x) { return ml_problem.E3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver.solve(problem_init);



    // L2 projection - end

    auto mat = ads::horrible_sparse_matrix{};
    auto M = [&mat](int row, int col, double val) { mat(row, col) += val; };
    auto rhs = [&F](int row, double val) { F[row] += val; };

    fmt::print("Assembling matrix\n");


    // Ex
    assemble(Ex, Ex, quad, M,    [eps](auto sx, auto tx, auto /*x*/) {return  -(eps    * sx.dw * tx.dw); });
    assemble(Ex, Ex, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dy * tx.dy); });
    assemble(Ex, Ex, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return   (mu_inv * sx.dz * tx.dz); });
    assemble(Ey, Ex, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dx * ty.dy); });
    assemble(Ez, Ex, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dx * tz.dz); });

    // Ey
    assemble(Ex, Ey, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dy * tx.dx); });
    assemble(Ey, Ey, quad, M,    [eps](auto sy, auto ty, auto /*x*/) {return  -(eps    * sy.dw * ty.dw); });
    assemble(Ey, Ey, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dy * ty.dy); });
    assemble(Ey, Ey, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return   (mu_inv * sy.dz * ty.dz); });
    assemble(Ez, Ey, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return  -(mu_inv * sz.dy * tz.dz); });

    // Ez
    assemble(Ex, Ez, quad, M, [mu_inv](auto sx, auto tx, auto /*x*/) {return  -(mu_inv * sx.dz * tx.dx); });
    assemble(Ey, Ez, quad, M, [mu_inv](auto sy, auto ty, auto /*x*/) {return  -(mu_inv * sy.dz * ty.dy); });
    assemble(Ez, Ez, quad, M,    [eps](auto sz, auto tz, auto /*x*/) {return  -(eps    * sz.dw * tz.dw); });
    assemble(Ez, Ez, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dx * tz.dx); });
    assemble(Ez, Ez, quad, M, [mu_inv](auto sz, auto tz, auto /*x*/) {return   (mu_inv * sz.dy * tz.dy); });


    fmt::print("Assembling RHS\n");
    assemble_rhs(Ex, quad, rhs, [=](auto v, auto /*x*/) { return 0 * v.val; });


    fmt::print("Collecting BC\n");
    auto is_fixed = std::vector<int>(n);
    for (auto const dof : Ex.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary - on y = 0, y = 1, z = 0, z = 1 we force Ex = 0
        fix = fix || iy == 0 || iy == Ex.space_y().dof_count() - 1;
        fix = fix || iz == 0 || iz == Ex.space_z().dof_count() - 1;

        if (fix) {
            is_fixed[Ex.global_index(dof)] = 1;
        }
    }

    for (auto const dof : Ey.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary - on x = 0, x = 1, z = 0, z = 1 we force Ey = 0
        fix = fix || ix == 0 || ix == Ey.space_x().dof_count() - 1;
        fix = fix || iz == 0 || iz == Ey.space_z().dof_count() - 1;

        if (fix) {
            is_fixed[Ey.global_index(dof)] = 1;
        }
    }

    for (auto const dof : Ez.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // initial condition t = 0
        fix = fix || it == 0;

        // spatial boundary - on x = 0, x = 1, y = 0, y = 1 we force Ez = 0
        fix = fix || ix == 0 || ix == Ez.space_x().dof_count() - 1;
        fix = fix || iy == 0 || iy == Ez.space_y().dof_count() - 1;

        if (fix) {
            is_fixed[Ez.global_index(dof)] = 1;
        }
    }

    fmt::print("Applying BC\n");
    for (auto const e : mesh.elements()) {
        // for Ex
        for (auto const i : Ex.dofs(e)) {
            auto const I = Ex.global_index(i);
            if (is_fixed[I] == 1) {
                // Matrix
                for (auto const j : Ex.dofs(e)) {
                    auto const J = Ex.global_index(j);
                    mat(I, J) = 0; // main diagonal
                }
                for (auto const j : Ey.dofs(e)) {
                    auto const J = Ey.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Ez.dofs(e)) {
                    auto const J = Ez.global_index(j);
                    mat(I, J) = 0;
                }
                mat(I, I) = 1;
                // RHS
                auto const [ix, iy, iz, it] = i;
                if (it == 0) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F[I] = Fx_init[I_init];
                } else { // boundary
                    F[I] = 0;
                }
            }
        }

        // for Ey
        for (auto const i : Ey.dofs(e)) {
            auto const I = Ey.global_index(i);
            if (is_fixed[I] == 1) {
                // Matrix
                for (auto const j : Ey.dofs(e)) {
                    auto const J = Ey.global_index(j);
                    mat(I, J) = 0; // main diagonal
                }
                for (auto const j : Ex.dofs(e)) {
                    auto const J = Ex.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Ez.dofs(e)) {
                    auto const J = Ez.global_index(j);
                    mat(I, J) = 0;
                }
                // RHS
                mat(I, I) = 1;
                auto const [ix, iy, iz, it] = i;
                if (it == 0) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F[I] = Fy_init[I_init];
                } else { // boundary
                    F[I] = 0;
                }
            }
        }

        // for Ez
        for (auto const i : Ez.dofs(e)) {
            auto const I = Ez.global_index(i);
            if (is_fixed[I] == 1) {
                // Matrix
                for (auto const j : Ez.dofs(e)) {
                    auto const J = Ez.global_index(j);
                    mat(I, J) = 0; // main diagonal
                }
                for (auto const j : Ey.dofs(e)) {
                    auto const J = Ey.global_index(j);
                    mat(I, J) = 0;
                }
                for (auto const j : Ex.dofs(e)) {
                    auto const J = Ex.global_index(j);
                    mat(I, J) = 0;
                }
                mat(I, I) = 1;
                // RHS
                auto const [ix, iy, iz, it] = i;
                if (it == 0) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F[I] = Fz_init[I_init];
                } else { // boundary
                    F[I] = 0;
                }
            }
        }
    }

    mat.mumpsify(problem);

    fmt::print("Equation system preparation\n");
    // problem.prepare_data();

    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());

    fmt::print("Solving\n");
    solver.solve(problem);

    auto const i_test4 = ads::index_types4::index{4,4,4,0};
    auto const I_test4 = Ez.global_index(i_test4);
    auto const i_test3 = ads::index_types3::index{4,4,4};
    auto const I_test3 = space_init.global_index(i_test3);
    fmt::print("TEST Ez0: {}  Ez_init: {}\n", F[I_test4], Fz_init[I_test3]);


    auto ex = ads::bspline_function4(&Ex, F.data());
    auto ey = ads::bspline_function4(&Ey, F.data());
    auto ez = ads::bspline_function4(&Ez, F.data());
    fmt::print("Saving\n");
    save_maxwell_to_file4D("dup-full.vti", T, ex, ey, ez);

    auto ex_init = ads::bspline_function3(&space_init, Fx_init.data());
    auto ey_init = ads::bspline_function3(&space_init, Fy_init.data());
    auto ez_init = ads::bspline_function3(&space_init, Fz_init.data());
    fmt::print("Saving Init\n");
    save_init_maxwell_to_file("dup-full.vti", T, ex_init, ey_init, ez_init);
    fmt::print("Done\n");
}