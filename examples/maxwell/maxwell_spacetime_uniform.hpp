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

#include "problems.hpp"

auto as_ms = [](auto d) { return std::chrono::duration_cast<std::chrono::milliseconds>(d); };
auto as_s  = [](auto d) { return static_cast<double>(as_ms(d).count())/1000; };

template <typename Mesh, typename Quad, typename Problem, typename FEx, typename FEy, typename FEz>
auto compute_norms_E(Mesh const& mesh, Quad const& quad, Problem const& problem, FEx const& fex, FEy const& fey, FEz const& fez, double time) {
    auto ex_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fex({x, y, z, t});
      };
    };

    auto ey_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fey({x, y, z, t});
      };
    };

    auto ez_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return fez({x, y, z, t});
      };
    };

    auto refE1 = [&](double t) {
      return [&, problem, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto refE2 = [&](double t) {
      return [&, problem, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto refE3 = [&](double t) {
      return [&, problem, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto max_timesteps = 10;
    auto timestep = 0;
    for (auto it : ads::evenly_spaced(0.0, time, max_timesteps)) {
        auto err_ex = error(mesh, quad, L2{}, ex_at_fixed_t(it), refE1(it));
        auto err_ey = error(mesh, quad, L2{}, ey_at_fixed_t(it), refE2(it));
        auto err_ez = error(mesh, quad, L2{}, ez_at_fixed_t(it), refE3(it));
        auto err    = sum_norms(err_ex, err_ey, err_ez);
        // fmt::print("  Timestep {}\n",timestep);
        // fmt::print("    L2 E.x  = {}\n",err_ex);
        // fmt::print("    L2 E.y  = {}\n",err_ey);
        // fmt::print("    L2 E.z  = {}\n",err_ez);
        // fmt::print("    L2 err  = {}\n",err);
        fmt::print("Timestep {}: {}\n",timestep,err);
        timestep++;
    }

}


template <typename Mesh, typename Quad, typename Problem, typename FEx, typename FEy, typename FEz>
auto compute_spacetime_norm_E(Mesh const& mesh, Quad const& quad, Problem const& problem, FEx const& fex, FEy const& fey, FEz const& fez, double time) {

    auto refE1 = [&]() {
      return [&, problem](ads::point4_t p) {
        auto const [x, y, z, t] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto refE2 = [&]() {
      return [&, problem](ads::point4_t p) {
        auto const [x, y, z, t] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto refE3 = [&]() {
      return [&, problem](ads::point4_t p) {
        auto const [x, y, z, t] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto err_ex = error(mesh, quad, L2{}, fex, refE1());
    auto err_ey = error(mesh, quad, L2{}, fey, refE2());
    auto err_ez = error(mesh, quad, L2{}, fez, refE3());
    auto err    = sum_norms(err_ex, err_ey, err_ez);

    fmt::print("L2 error: {}\n",err);

}



template <typename FEx, typename FEy, typename FEz>
auto save_maxwell_to_file(std::string const& path, FEx const& fex, FEy const& fey, FEz const& fez) -> void {
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
auto save_maxwell_to_file4D(std::string const& path_base, double time, FEx const& fex, FEy const& fey, FEz const& fez) -> void {
    constexpr auto res_time = 50;

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
        save_maxwell_to_file(fmt::format("{}_{}.vti", path_base, i), fex_at_fixed_t(t), fey_at_fixed_t(t), fez_at_fixed_t(t));
        ++i;
    }
}


template <typename Problem>
auto save_maxwell_ref_to_file4D(std::string const& path_base, double time, Problem const& problem) -> void {
    constexpr auto res_time = 50;

    auto fex_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return problem.E1({x, y, z}, t).val;
      };
    };

    auto fey_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return problem.E2({x, y, z}, t).val;
      };
    };

    auto fez_at_fixed_t = [&](double t) {
      return [&, t](ads::point3_t p) {
        auto const [x, y, z] = p;
        return problem.E3({x, y, z}, t).val;
      };
    };


    int i = 0;
    for (auto t : ads::evenly_spaced(0.0, time, res_time)) {
        save_maxwell_to_file(fmt::format("{}_{}.vti", path_base, i), fex_at_fixed_t(t), fey_at_fixed_t(t), fez_at_fixed_t(t));
        ++i;
    }
}


template <typename FEx, typename FEy, typename FEz>
auto save_init_maxwell_to_file(std::string const& path, FEx const& fex, FEy const& fey, FEz const& fez) -> void {

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

    save_maxwell_to_file(path, fex_at_fixed_t(), fey_at_fixed_t(), fez_at_fixed_t());
}



auto maxwell_spacetime_uniform_main_mumps(int /*argc*/, char* /*argv*/[]) -> void {

    auto const x_elems = 8;
    auto const y_elems = 8;
    auto const z_elems = 8;
    auto const t_elems = 8;

    auto const p = 1;
    auto const c = 0;

    auto const eps = 1; //8.854e-12;
    auto const mu  = 1; //12.556e-7;
    auto const T = 1.0;
    auto const mu_inv = 1/mu;



    auto const xs = ads::evenly_spaced(0.0, 1.0, x_elems);
    auto const ys = ads::evenly_spaced(0.0, 1.0, y_elems);
    auto const zs = ads::evenly_spaced(0.0, 1.0, z_elems);
    auto const ts = ads::evenly_spaced(0.0, T,   t_elems);

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

    fmt::print("*\tPsi(x,t=0).x\n");
    auto Fpsi_init_x = std::vector<double>(n_init);
    auto rhs_init_psi_x = [&Fpsi_init_x](int J, double val) { Fpsi_init_x[J] += val; };
    assemble_rhs(space_init, quad_init, rhs_init_psi_x, [&ml_problem, mu_inv](auto v, auto x) { return  mu_inv * (ml_problem.H3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dy - ml_problem.H2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dz) * v.val; });

    fmt::print("*\tPsi(x,t=0).y\n");
    auto Fpsi_init_y = std::vector<double>(n_init);
    auto rhs_init_psi_y = [&Fpsi_init_y](int J, double val) { Fpsi_init_y[J] += val; };
    assemble_rhs(space_init, quad_init, rhs_init_psi_y, [&ml_problem, mu_inv](auto v, auto x) { return  mu_inv * (ml_problem.H1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dz - ml_problem.H3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dx) * v.val; });

    fmt::print("*\tPsi(x,t=0).z\n");
    auto Fpsi_init_z = std::vector<double>(n_init);
    auto rhs_init_psi_z = [&Fpsi_init_z](int J, double val) { Fpsi_init_z[J] += val; };
    assemble_rhs(space_init, quad_init, rhs_init_psi_z, [&ml_problem, mu_inv](auto v, auto x) { return  mu_inv * (ml_problem.H2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dx - ml_problem.H1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dy) * v.val; });



    // L2 projection - end

    auto mat = ads::horrible_sparse_matrix{};
    auto M = [&mat](int row, int col, double val) { mat(row, col) += val; };
    auto rhs = [&F](int row, double val) { F[row] += val; };

    fmt::print("Assembling matrix\n");


    // Ex
    assemble(Ex, Ex, quad, M,    [eps](auto ex, auto vx, auto /*x*/) {return  -(eps    * ex.dw * vx.dw); });
    assemble(Ex, Ex, quad, M, [mu_inv](auto ex, auto vx, auto /*x*/) {return   (mu_inv * ex.dy * vx.dy); });
    assemble(Ex, Ex, quad, M, [mu_inv](auto ex, auto vx, auto /*x*/) {return   (mu_inv * ex.dz * vx.dz); });
    assemble(Ey, Ex, quad, M, [mu_inv](auto ey, auto vx, auto /*x*/) {return  -(mu_inv * ey.dx * vx.dy); });
    assemble(Ez, Ex, quad, M, [mu_inv](auto ez, auto vx, auto /*x*/) {return  -(mu_inv * ez.dx * vx.dz); });

    // Ey
    assemble(Ex, Ey, quad, M, [mu_inv](auto ex, auto vy, auto /*x*/) {return  -(mu_inv * ex.dy * vy.dx); });
    assemble(Ey, Ey, quad, M,    [eps](auto ey, auto vy, auto /*x*/) {return  -(eps    * ey.dw * vy.dw); });
    assemble(Ey, Ey, quad, M, [mu_inv](auto ey, auto vy, auto /*x*/) {return   (mu_inv * ey.dx * vy.dx); });
    assemble(Ey, Ey, quad, M, [mu_inv](auto ey, auto vy, auto /*x*/) {return   (mu_inv * ey.dz * vy.dz); });
    assemble(Ez, Ey, quad, M, [mu_inv](auto ez, auto vy, auto /*x*/) {return  -(mu_inv * ez.dy * vy.dz); });

    // Ez
    assemble(Ex, Ez, quad, M, [mu_inv](auto ex, auto vz, auto /*x*/) {return  -(mu_inv * ex.dz * vz.dx); });
    assemble(Ey, Ez, quad, M, [mu_inv](auto ey, auto vz, auto /*x*/) {return  -(mu_inv * ey.dz * vz.dy); });
    assemble(Ez, Ez, quad, M,    [eps](auto ez, auto vz, auto /*x*/) {return  -(eps    * ez.dw * vz.dw); });
    assemble(Ez, Ez, quad, M, [mu_inv](auto ez, auto vz, auto /*x*/) {return   (mu_inv * ez.dx * vz.dx); });
    assemble(Ez, Ez, quad, M, [mu_inv](auto ez, auto vz, auto /*x*/) {return   (mu_inv * ez.dy * vz.dy); });


    fmt::print("Assembling RHS\n");
    assemble_rhs(Ex, quad, rhs, [=](auto v, auto /*x*/) { return 0 * v.val; });


    fmt::print("Collecting BC\n");
    auto is_fixed = std::vector<int>(n);
    for (auto const dof : Ex.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ex.space_w().dof_count() - 1;

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

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ey.space_w().dof_count() - 1;

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

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ez.space_w().dof_count() - 1;

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
                    mat(I, J) = 0;
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
                if (it == Ex.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    auto const i_new = ads::index_types4::index{ix, iy, iz, 0};
                    auto const I_new = Ex.global_index(i_new);
                    mat(I, I_new) = 1;
                    F[I] = Fx_init[I_init];
                } else { // boundary
                    F[I] = 0;
                    mat(I, I) = 1;
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
                    mat(I, J) = 0;
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
                if (it == Ey.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    auto const i_new = ads::index_types4::index{ix, iy, iz, 0};
                    auto const I_new = Ey.global_index(i_new);
                    mat(I, I_new) = 1;
                    F[I] = Fy_init[I_init];
                } else { // boundary
                    F[I] = 0;
                    mat(I, I) = 1;
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
                    mat(I, J) = 0;
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
                if (it == Ez.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    auto const i_new = ads::index_types4::index{ix, iy, iz, 0};
                    auto const I_new = Ez.global_index(i_new);
                    mat(I, I_new) = 1;
                    F[I] = Fz_init[I_init];
                } else { // boundary
                    F[I] = 0;
                    mat(I, I) = 1;
                }
            }
        }

    }

    // RHS update
    for (auto const i : Ex.dofs()) {
        auto const I = Ex.global_index(i);
        auto const [ix, iy, iz, it] = i;
        if (it == 0) {
            auto const i_init = ads::index_types3::index{ix, iy, iz};
            auto const I_init = space_init.global_index(i_init);
            F[I] -= Fpsi_init_x[I_init];
        }
    }

    for (auto const i : Ey.dofs()) {
        auto const I = Ey.global_index(i);
        auto const [ix, iy, iz, it] = i;
        if (it == 0) {
            auto const i_init = ads::index_types3::index{ix, iy, iz};
            auto const I_init = space_init.global_index(i_init);
            F[I] -= Fpsi_init_y[I_init];
        }
    }

    for (auto const i : Ez.dofs()) {
        auto const I = Ez.global_index(i);
        auto const [ix, iy, iz, it] = i;
        if (it == 0) {
            auto const i_init = ads::index_types3::index{ix, iy, iz};
            auto const I_init = space_init.global_index(i_init);
            F[I] -= Fpsi_init_z[I_init];
        }
    }


    mat.mumpsify(problem);

    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());

    fmt::print("Solving\n");
    solver.solve(problem);


    auto ex = ads::bspline_function4(&Ex, F.data());
    auto ey = ads::bspline_function4(&Ey, F.data());
    auto ez = ads::bspline_function4(&Ez, F.data());


    fmt::print("Saving\n");
    save_maxwell_to_file4D("output", T, ex, ey, ez);

    auto ex_init = ads::bspline_function3(&space_init, Fx_init.data());
    auto ey_init = ads::bspline_function3(&space_init, Fy_init.data());
    auto ez_init = ads::bspline_function3(&space_init, Fz_init.data());


    fmt::print("Saving Init\n");
    save_init_maxwell_to_file("init_output.vti", ex_init, ey_init, ez_init);


    fmt::print("Computing error\n");
    compute_norms_E(mesh_init, quad_init, ml_problem, ex, ey, ez, T);

    fmt::print("Save ref\n");
    save_maxwell_ref_to_file4D("output_ref", 1.0, ml_problem);

    fmt::print("Done\n");
}


auto maxwell_spacetime_uniform_main_eigen(auto const args) -> void {

    auto const x_elems = args.el_x;
    auto const y_elems = args.el_y;
    auto const z_elems = args.el_z;
    auto const t_elems = args.el_t;

    auto const p = 1;
    auto const c = 0;

    auto const eps = 1; //8.854e-12;
    auto const mu  = 1; //12.556e-7;
    auto const T = 1.0; //0.7071067811865475;
    auto const mu_inv = 1/mu;

    auto const save_problem = args.save_problem;



    auto const xs = ads::evenly_spaced(0.0, 1.0, x_elems);
    auto const ys = ads::evenly_spaced(0.0, 1.0, y_elems);
    auto const zs = ads::evenly_spaced(0.0, 1.0, z_elems);
    auto const ts = ads::evenly_spaced(0.0, T,   t_elems);

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
    auto F_guess = std::vector<double>(n);
    auto problem = ads::eigen::problem{F.data(), n};
    auto ml_problem = maxwell_manufactured1{1,1};
    auto solver = ads::eigen::solver{args.max_iter};
    solver.set_tolerance(args.toler);

    auto t_before_sim = std::chrono::steady_clock::now();

    // L2 projection - begin
    fmt::print("L2 Projection:\n");
    fmt::print("*\tE(x,t=0).x\n");

    auto mesh_init = ads::regular_mesh3{xs, ys, zs};
    auto quad_init = ads::quadrature3{&mesh_init, std::max(p + 1, 2)};
    auto space_init = ads::space3{&mesh_init, bx, by, bz};
    auto n_init = space_init.dof_count();
    auto solver_init = ads::mumps::solver{};

    auto Fx_init = std::vector<double>(n_init);
    auto problem_init = ads::mumps::problem{Fx_init.data(), n_init};

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

    auto Fy_init = std::vector<double>(n_init);
    problem_init = ads::mumps::problem{Fy_init.data(), n_init};
    auto rhs_init_y = [&Fy_init](int J, double val) { Fy_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_y, [&ml_problem](auto v, auto x) { return ml_problem.E2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);

    fmt::print("*\tE(x,t=0).z\n");
    auto Fz_init = std::vector<double>(n_init);
    problem_init = ads::mumps::problem{Fz_init.data(), n_init};
    auto rhs_init_z = [&Fz_init](int J, double val) { Fz_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_z, [&ml_problem](auto v, auto x) { return ml_problem.E3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);

    fmt::print("*\tPsi(x,t=0).x\n");
    auto Fpsi_init_x = std::vector<double>(n_init);
    auto rhs_init_psi_x = [&Fpsi_init_x](int J, double val) { Fpsi_init_x[J] += val; };
    assemble_rhs(space_init, quad_init, rhs_init_psi_x, [&ml_problem, mu_inv](auto v, auto x) { return  mu_inv * (ml_problem.H3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dy - ml_problem.H2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dz) * v.val; });

    fmt::print("*\tPsi(x,t=0).y\n");
    auto Fpsi_init_y = std::vector<double>(n_init);
    auto rhs_init_psi_y = [&Fpsi_init_y](int J, double val) { Fpsi_init_y[J] += val; };
    assemble_rhs(space_init, quad_init, rhs_init_psi_y, [&ml_problem, mu_inv](auto v, auto x) { return  mu_inv * (ml_problem.H1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dz - ml_problem.H3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dx) * v.val; });

    fmt::print("*\tPsi(x,t=0).z\n");
    auto Fpsi_init_z = std::vector<double>(n_init);
    auto rhs_init_psi_z = [&Fpsi_init_z](int J, double val) { Fpsi_init_z[J] += val; };
    assemble_rhs(space_init, quad_init, rhs_init_psi_z, [&ml_problem, mu_inv](auto v, auto x) { return  mu_inv * (ml_problem.H2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dx - ml_problem.H1({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).dy) * v.val; });



    // L2 projection - end

    auto mat = ads::horrible_sparse_matrix{};
    auto M = [&mat](int row, int col, double val) { mat(row, col) += val; };
    auto rhs = [&F](int row, double val) { F[row] += val; };

    auto t_before_lhs = std::chrono::steady_clock::now();
    fmt::print("Assembling matrix\n");


    // Ex
    assemble(Ex, Ex, quad, M,    [eps](auto ex, auto vx, auto /*x*/) {return  -(eps    * ex.dw * vx.dw); });
    assemble(Ex, Ex, quad, M, [mu_inv](auto ex, auto vx, auto /*x*/) {return   (mu_inv * ex.dy * vx.dy); });
    assemble(Ex, Ex, quad, M, [mu_inv](auto ex, auto vx, auto /*x*/) {return   (mu_inv * ex.dz * vx.dz); });
    assemble(Ey, Ex, quad, M, [mu_inv](auto ey, auto vx, auto /*x*/) {return  -(mu_inv * ey.dx * vx.dy); });
    assemble(Ez, Ex, quad, M, [mu_inv](auto ez, auto vx, auto /*x*/) {return  -(mu_inv * ez.dx * vx.dz); });

    // Ey
    assemble(Ex, Ey, quad, M, [mu_inv](auto ex, auto vy, auto /*x*/) {return  -(mu_inv * ex.dy * vy.dx); });
    assemble(Ey, Ey, quad, M,    [eps](auto ey, auto vy, auto /*x*/) {return  -(eps    * ey.dw * vy.dw); });
    assemble(Ey, Ey, quad, M, [mu_inv](auto ey, auto vy, auto /*x*/) {return   (mu_inv * ey.dx * vy.dx); });
    assemble(Ey, Ey, quad, M, [mu_inv](auto ey, auto vy, auto /*x*/) {return   (mu_inv * ey.dz * vy.dz); });
    assemble(Ez, Ey, quad, M, [mu_inv](auto ez, auto vy, auto /*x*/) {return  -(mu_inv * ez.dy * vy.dz); });

    // Ez
    assemble(Ex, Ez, quad, M, [mu_inv](auto ex, auto vz, auto /*x*/) {return  -(mu_inv * ex.dz * vz.dx); });
    assemble(Ey, Ez, quad, M, [mu_inv](auto ey, auto vz, auto /*x*/) {return  -(mu_inv * ey.dz * vz.dy); });
    assemble(Ez, Ez, quad, M,    [eps](auto ez, auto vz, auto /*x*/) {return  -(eps    * ez.dw * vz.dw); });
    assemble(Ez, Ez, quad, M, [mu_inv](auto ez, auto vz, auto /*x*/) {return   (mu_inv * ez.dx * vz.dx); });
    assemble(Ez, Ez, quad, M, [mu_inv](auto ez, auto vz, auto /*x*/) {return   (mu_inv * ez.dy * vz.dy); });

    auto t_after_lhs = std::chrono::steady_clock::now();

    auto t_before_rhs = std::chrono::steady_clock::now();
    fmt::print("Assembling RHS\n");
    assemble_rhs(Ex, quad, rhs, [=](auto v, auto /*x*/) { return 0 * v.val; });
    auto t_after_rhs = std::chrono::steady_clock::now();

    auto t_before_bnd = std::chrono::steady_clock::now();
    fmt::print("Collecting BC\n");
    auto is_fixed = std::vector<int>(n);
    for (auto const dof : Ex.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ex.space_w().dof_count() - 1;

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

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ey.space_w().dof_count() - 1;

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

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ez.space_w().dof_count() - 1;

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
                    mat(I, J) = 0;
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
                if (it == Ex.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    auto const i_new = ads::index_types4::index{ix, iy, iz, 0};
                    auto const I_new = Ex.global_index(i_new);
                    mat(I, I_new) = 1;
                    F[I] = Fx_init[I_init];
                    F_guess[I] = Fx_init[I_init];
                } else { // boundary
                    F[I] = 0;
                    mat(I, I) = 1;
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
                    mat(I, J) = 0;
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
                if (it == Ey.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    auto const i_new = ads::index_types4::index{ix, iy, iz, 0};
                    auto const I_new = Ey.global_index(i_new);
                    mat(I, I_new) = 1;
                    F[I] = Fy_init[I_init];
                    F_guess[I] = Fy_init[I_init];
                } else { // boundary
                    F[I] = 0;
                    mat(I, I) = 1;
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
                    mat(I, J) = 0;
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
                if (it == Ez.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    auto const i_new = ads::index_types4::index{ix, iy, iz, 0};
                    auto const I_new = Ez.global_index(i_new);
                    mat(I, I_new) = 1;
                    F[I] = Fz_init[I_init];
                    F_guess[I] = Fz_init[I_init];
                } else { // boundary
                    F[I] = 0;
                    mat(I, I) = 1;
                }
            }
        }

    }

    // RHS update
    for (auto const i : Ex.dofs()) {
        auto const I = Ex.global_index(i);
        auto const [ix, iy, iz, it] = i;
        if (it == 0) {
            auto const i_init = ads::index_types3::index{ix, iy, iz};
            auto const I_init = space_init.global_index(i_init);
            F[I] -= Fpsi_init_x[I_init];
        }
    }

    for (auto const i : Ey.dofs()) {
        auto const I = Ey.global_index(i);
        auto const [ix, iy, iz, it] = i;
        if (it == 0) {
            auto const i_init = ads::index_types3::index{ix, iy, iz};
            auto const I_init = space_init.global_index(i_init);
            F[I] -= Fpsi_init_y[I_init];
        }
    }

    for (auto const i : Ez.dofs()) {
        auto const I = Ez.global_index(i);
        auto const [ix, iy, iz, it] = i;
        if (it == 0) {
            auto const i_init = ads::index_types3::index{ix, iy, iz};
            auto const I_init = space_init.global_index(i_init);
            F[I] -= Fpsi_init_z[I_init];
        }
    }

    auto t_after_bnd = std::chrono::steady_clock::now();
    mat.eigenify(problem);

    fmt::print("Equation system preparation\n");
    problem.prepare_data();

    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());

    auto t_before_matrix_save = std::chrono::steady_clock::now();
    if (save_problem) {
        fmt::print("Saving matrix to file\n");
        std::ostringstream data_files;
        data_files << "MAXWELL_UNIFORM_" << x_elems << "_" << y_elems << "_" << z_elems << "_" << t_elems;
        solver.save_to_file(problem, data_files.str().c_str());
    }
    auto t_after_matrix_save = std::chrono::steady_clock::now();

    auto t_before_solver = std::chrono::steady_clock::now();
    fmt::print("Solving\n");
    auto eigein_result = solver.solveWithGuess(problem, F_guess.data());
    auto t_after_solver = std::chrono::steady_clock::now();

    auto ex = ads::bspline_function4(&Ex, eigein_result);
    auto ey = ads::bspline_function4(&Ey, eigein_result);
    auto ez = ads::bspline_function4(&Ez, eigein_result);


    fmt::print("Saving\n");
    save_maxwell_to_file4D("output", T, ex, ey, ez);

    auto t_after_sim = std::chrono::steady_clock::now();

    auto ex_init = ads::bspline_function3(&space_init, Fx_init.data());
    auto ey_init = ads::bspline_function3(&space_init, Fy_init.data());
    auto ez_init = ads::bspline_function3(&space_init, Fz_init.data());


    // fmt::print("Saving Init\n");
    // save_init_maxwell_to_file("init_output.vti", ex_init, ey_init, ez_init);


    fmt::print("Computing error\n");
    compute_norms_E(mesh_init, quad_init, ml_problem, ex, ey, ez, T);

    fmt::print("Computing space-time L2 error\n");
    compute_spacetime_norm_E(mesh, quad, ml_problem, ex, ey, ez, T);


    // fmt::print("Save ref\n");
    // save_maxwell_ref_to_file4D("output_ref", T, ml_problem);

    fmt::print("Time: \n");
    fmt::print("LHS    :  {:.{}f} s\n", as_s(t_after_lhs - t_before_lhs), 3);
    fmt::print("RHS    :  {:.{}f} s\n", as_s(t_after_rhs - t_before_rhs), 3);
    fmt::print("Bndry  :  {:.{}f} s\n", as_s(t_after_bnd - t_before_bnd), 3);
    fmt::print("Solver :  {:.{}f} s\n", as_s(t_after_solver - t_before_solver), 3);
    fmt::print("Total  :  {:.{}f} s\n", as_s((t_after_sim - t_before_sim) - (t_after_matrix_save - t_before_matrix_save)), 3);


    fmt::print("Done\n");
}

auto maxwell_spacetime_uniform_main_eigen_from_file(auto const args) -> void {

    auto const x_elems = args.el_x;
    auto const y_elems = args.el_y;
    auto const z_elems = args.el_z;
    auto const t_elems = args.el_t;

    auto const p = 1;
    auto const c = 0;

    auto const eps = 1; //8.854e-12;
    auto const mu  = 1; //12.556e-7;
    auto const T = 1.0;
    auto const mu_inv = 1/mu;



    auto const xs = ads::evenly_spaced(0.0, 1.0, x_elems);
    auto const ys = ads::evenly_spaced(0.0, 1.0, y_elems);
    auto const zs = ads::evenly_spaced(0.0, 1.0, z_elems);
    auto const ts = ads::evenly_spaced(0.0, T,   t_elems);

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

    auto F_guess = std::vector<double>(n);
    auto problem = ads::eigen::problem{NULL, n};
    auto ml_problem = maxwell_manufactured1{1,1};
    auto solver = ads::eigen::solver{args.max_iter};
    solver.set_tolerance(args.toler);


    auto mesh_init = ads::regular_mesh3{xs, ys, zs};
    auto quad_init = ads::quadrature3{&mesh_init, std::max(p + 1, 2)};
    auto space_init = ads::space3{&mesh_init, bx, by, bz};
    auto n_init = space_init.dof_count();
    auto solver_init = ads::mumps::solver{};

    auto Fx_init = std::vector<double>(n_init);
    auto problem_init = ads::mumps::problem{Fx_init.data(), n_init};

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

    auto Fy_init = std::vector<double>(n_init);
    problem_init = ads::mumps::problem{Fy_init.data(), n_init};
    auto rhs_init_y = [&Fy_init](int J, double val) { Fy_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_y, [&ml_problem](auto v, auto x) { return ml_problem.E2({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);

    fmt::print("*\tE(x,t=0).z\n");
    auto Fz_init = std::vector<double>(n_init);
    problem_init = ads::mumps::problem{Fz_init.data(), n_init};
    auto rhs_init_z = [&Fz_init](int J, double val) { Fz_init[J] += val; };
    assemble(space_init, quad_init, out_init, [](auto u, auto v, auto /*x*/) { return u.val * v.val; });
    assemble_rhs(space_init, quad_init, rhs_init_z, [&ml_problem](auto v, auto x) { return ml_problem.E3({std::get<0>(x),std::get<1>(x),std::get<2>(x)},0).val * v.val; });
    solver_init.solve(problem_init);



    fmt::print("Collecting BC\n");
    auto is_fixed = std::vector<int>(n);
    for (auto const dof : Ex.dofs()) {
        auto const [ix, iy, iz, it] = dof;
        bool fix = false;

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ex.space_w().dof_count() - 1;

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

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ey.space_w().dof_count() - 1;

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

        // set the initial condition for t = 0 but in the last time step
        fix = fix || it == Ez.space_w().dof_count() - 1;

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
                auto const [ix, iy, iz, it] = i;
                if (it == Ex.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F_guess[I] = Fx_init[I_init];
                }
            }
        }

        // for Ey
        for (auto const i : Ey.dofs(e)) {
            auto const I = Ey.global_index(i);
            if (is_fixed[I] == 1) {
                auto const [ix, iy, iz, it] = i;
                if (it == Ey.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F_guess[I] = Fy_init[I_init];
                }
            }
        }

        // for Ez
        for (auto const i : Ez.dofs(e)) {
            auto const I = Ez.global_index(i);
            if (is_fixed[I] == 1) {
                auto const [ix, iy, iz, it] = i;
                if (it == Ez.space_w().dof_count() - 1) { // initial state
                    auto const i_init = ads::index_types3::index{ix, iy, iz};
                    auto const I_init = space_init.global_index(i_init);
                    F_guess[I] = Fz_init[I_init];
                }
            }
        }

    }

    auto t_before_read_files = std::chrono::steady_clock::now();
    std::ostringstream data_files;
    data_files << "MAXWELL_UNIFORM_" << x_elems << "_" << y_elems << "_" << z_elems << "_" << t_elems;
    solver.read_problem_from_file(problem, data_files.str().c_str());
    auto t_after_read_files = std::chrono::steady_clock::now();



    auto t_before_eq_prep = std::chrono::steady_clock::now();
    fmt::print("Equation system preparation\n");
    problem.prepare_data();

    fmt::print("Non-zeros: {}\n", problem.nonzero_entries());
    auto t_after_eq_prep = std::chrono::steady_clock::now();

    auto t_before_solver = std::chrono::steady_clock::now();
    fmt::print("Solving\n");
    auto eigein_result = solver.solveWithGuess(problem, F_guess.data());
    // auto eigein_result = solver.solve(problem);
    auto t_after_solver = std::chrono::steady_clock::now();

    auto t_before_output = std::chrono::steady_clock::now();

    fmt::print("Saving\n");
    auto ex = ads::bspline_function4(&Ex, eigein_result);
    auto ey = ads::bspline_function4(&Ey, eigein_result);
    auto ez = ads::bspline_function4(&Ez, eigein_result);

    save_maxwell_to_file4D("output", T, ex, ey, ez);

    fmt::print("Computing error\n");
    compute_norms_E(mesh_init, quad_init, ml_problem, ex, ey, ez, T);

    fmt::print("\nComputing space-time L2 error\n");
    compute_spacetime_norm_E(mesh, quad, ml_problem, ex, ey, ez, T);

    auto t_after_output = std::chrono::steady_clock::now();
    fmt::print("Done\n");



    fmt::print("Reading:  {:>8%Q %q}\n", as_s(t_after_read_files - t_before_read_files));
    fmt::print("Eq Sys :  {:>8%Q %q}\n", as_s(t_after_eq_prep - t_before_eq_prep));
    fmt::print("Solver :  {:>8%Q %q}\n", as_s(t_after_solver - t_before_solver));
    fmt::print("Output :  {:>8%Q %q}\n", as_s(t_after_output - t_before_output));

}