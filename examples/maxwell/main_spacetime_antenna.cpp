// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>

#include <lyra/lyra.hpp>

#include "maxwell_spacetime_antenna.hpp"

auto parse_args(int argc, char* argv[]) {
    struct {
        int n, p, c, step_count;
        double T;
    } args{};

    bool show_help = false;

    auto const* const desc =
        "Solver for non-stationary space-time Maxwell equations with uniform material data\n"
        "using Eigen";

    auto const cli = lyra::help(show_help).description(desc)  //
         | lyra::cli()  
        //  | lyra::arg(args.n, "N")("mesh resolution").required()
        //  | lyra::arg(args.p, "p")("B-spline order").required()
        ;

    auto const result = cli.parse({argc, argv});
    // validate_args(cli, result, show_help);
    return args;
}

// Example invocation:
// <prog> 8 2 1 10 0.1
int main(int argc, char* argv[]) {
    // auto const args = parse_args(argc, argv);
    // auto const dt = args.T / args.step_count;
    // auto const steps = ads::timesteps_config{args.step_count, dt};

    // auto const dim = ads::dim_config{args.p, args.n};
    // auto const cfg = ads::config_3d{dim, dim, dim, steps, 1};

    // auto sim = maxwell_uniform{cfg};
    // sim.run();
    maxwell_spacetime_antenna_main_mumps(argc, argv);
}
