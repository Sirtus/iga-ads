#ifndef ADS_OUTPUT_MANAGER_HPP_
#define ADS_OUTPUT_MANAGER_HPP_

#include <cstddef>
#include <vector>
#include <ostream>

#include "ads/bspline/bspline.hpp"
#include "ads/bspline/eval.hpp"
#include "ads/lin/tensor.hpp"
#include "ads/output/grid.hpp"
#include "ads/output/output_format.hpp"
#include "ads/output/output_manager_base.hpp"
#include "ads/output/axis.hpp"

#include "ads/output/vtk.hpp"
#include "ads/output/gnuplot.hpp"


namespace ads {

const output::output_format DEFAULT_FMT = output::fixed_format(10, 18);


template <std::size_t Dim>
struct output_manager;


template <>
struct output_manager<1> : output_manager_base<output_manager<1>> {
private:
    output::axis x;
    lin::tensor<double, 1> vals;
    output::gnuplot_printer<1> output{ DEFAULT_FMT };

public:
    output_manager(const bspline::basis& bx, std::size_t n)
    : x{ bx, n }
    , vals{{ x.size() }}
    { }

    using output_manager_base::to_file;

    template <typename Solution>
    void write(const Solution& sol, std::ostream& os) {
        for (std::size_t i = 0; i < x.size(); ++ i) {
            vals(i) = bspline::eval(x[i], sol, x.basis, x.ctx);
        }
        auto grid = make_grid(x.range());
        output.print(os, grid, vals);
    }
};



template <>
struct output_manager<2> : output_manager_base<output_manager<2>> {
private:
    output::axis x, y;
    lin::tensor<double, 2> vals;
    output::vtk output{ DEFAULT_FMT };

public:
    output_manager(const bspline::basis& bx, const bspline::basis& by, std::size_t n)
    : x{ bx, n }
    , y{ by, n }
    , vals{{ x.size(), y.size() }}
    { }

    using output_manager_base::to_file;

    template <typename Solution>
    void write(const Solution& sol, std::ostream& os) {
        for (std::size_t i = 0; i < x.size(); ++ i) {
        for (std::size_t j = 0; j < y.size(); ++ j) {
            vals(i, j) = bspline::eval(x[i], y[j], sol, x.basis, y.basis, x.ctx, y.ctx);
        }
        }
        auto grid = make_grid(x.range(), y.range());
        output.print(os, grid, vals);
    }
};


template <>
struct output_manager<3> : output_manager_base<output_manager<3>> {
private:
    output::axis x, y, z;
    lin::tensor<double, 3> vals;
    output::vtk output{ DEFAULT_FMT };

public:
    output_manager(const bspline::basis& bx, const bspline::basis& by, const bspline::basis& bz, std::size_t n)
    : x{ bx, n }
    , y{ by, n }
    , z{ bz, n }
    , vals{{ x.size(), y.size(), z.size() }}
    { }

    using output_manager_base::to_file;

    template <typename Solution>
    void write(const Solution& sol, std::ostream& os) {
        for (std::size_t i = 0; i < x.size(); ++ i) {
        for (std::size_t j = 0; j < y.size(); ++ j) {
        for (std::size_t k = 0; k < z.size(); ++ k) {
            vals(i, j, k) = bspline::eval(x[i], y[j], z[k], sol, x.basis, y.basis, z.basis, x.ctx, y.ctx, z.ctx);
        }
        }
        }
        auto grid = make_grid(x.range(), y.range(), z.range());
        output.print(os, grid, vals);
    }
};


}


#endif /* ADS_OUTPUT_MANAGER_HPP_ */
