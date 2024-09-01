// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>

#include <lyra/lyra.hpp>

#include "maxwell_spacetime_uniform.hpp"
#include "maxwell_base.hpp"

auto parse_args(int argc, char* argv[]) {
    struct {
        int el_x, el_y, el_z, el_t, max_iter, save_problem, from_file;
        double toler;
    } args{};

    bool show_help = false;

    auto const* const desc =
        "Solver for non-stationary space-time Maxwell equations with uniform material data\n"
        "using Eigen";

    auto const cli = lyra::help(show_help).description(desc)  //
         | lyra::cli()
         | lyra::opt(args.el_x, "integer").required() ["-x"] ("number of elements for dimension x")
         | lyra::opt(args.el_y, "integer").required() ["-y"] ("number of elements for dimension y")
         | lyra::opt(args.el_z, "integer").required() ["-z"] ("number of elements for dimension z")
         | lyra::opt(args.el_t, "integer").required() ["-t"] ("number of elements for dimension t")
         | lyra::opt(args.max_iter, "integer").required() ["-i"]["--iter"] ("maximum number of iterations")
         | lyra::opt(args.toler, "double") ["--toler"] ("solver error tolerance")
         | lyra::opt(args.save_problem, "bit")["-s"]["--save"] ("save the problem to file")
         | lyra::opt(args.from_file, "bit") ["--from_file"] ("read problem from file")
        ;

    auto const result = cli.parse({argc, argv});
    validate_args(cli, result, show_help);
    return args;
}

// Example invocation:
// maxwell_spacetime_uniform -x 12 -y 12 -z 12 -t 16 -i 50 -s 0 --toler 0.005 --from_file 0
int main(int argc, char* argv[]) {
    auto const args = parse_args(argc, argv);
    if (args.from_file) maxwell_spacetime_uniform_main_eigen_from_file(args);
    else maxwell_spacetime_uniform_main_eigen(args);
}
