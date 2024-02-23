// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_SOLVER_EIGEN_HPP
#define ADS_SOLVER_EIGEN_HPP

#include "ads/config.hpp"

#ifdef ADS_USE_EIGEN

#    include <Eigen/Dense>
#    include <Eigen/Sparse>

#    include <cstdint>
#    include <cstdio>
#    include <iostream>
#    include <vector>

#    include "ads/util.hpp"

namespace ads::eigen {

struct problem {
    problem(double* rhs, int n)
    : rhs_data_{rhs}
    , n{n} { 
        rhs_.resize(n);
        a_.resize(n, n);
    }

    explicit problem(std::vector<double>& rhs)
    : rhs_data_{rhs.data()}
    , n{static_cast<int>(rhs.size())} { }


    void add(int row, int col, double value) {
        cols_.push_back(col);
        rows_.push_back(row);
        values_.push_back(value);   
    }

    int nonzero_entries() const { return  narrow_cast<int> (values_.size()); }

    int dofs() const { return n; }

    Eigen::SparseMatrix<double> a() { return a_; }

    void prepare_data() {
        std::vector<Eigen::Triplet<double>> triplets;

        int triplets_num = static_cast<int> (values_.size());
        for (int i = 0; i < triplets_num; ++i) triplets.push_back(Eigen::Triplet<double>(rows_[i], cols_[i], values_[i]));
        a_.setFromTriplets(triplets.begin(), triplets.end());

        for (int i = 0; i < n; ++i) rhs_(i) = rhs_data_[i];
    }

    Eigen::VectorXd rhs() { return rhs_; }

    void rhs(double* data) { 
        rhs_data_ = data;
    }

private:
    std::vector<int> rows_;
    std::vector<int> cols_;
    std::vector<double> values_;
    double* rhs_data_;
    int n;
    Eigen::VectorXd rhs_;
    Eigen::SparseMatrix<double> a_;
    
    Eigen::Triplet<double> triplet;
};

class solver {
private:
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver_;
    int max_iter_ = 0;
public:
    solver() { }
    solver(int max_iter) : max_iter_{max_iter} { }
    solver(const solver&) = delete;
    solver& operator=(const solver&) = delete;
    solver(solver&&) = delete;
    solver& operator=(solver&&) = delete;

    void set_max_iter(int max_iter) { max_iter_ = max_iter; }

    // void save_to_file(problem& problem, const char* output_path) {
    //     prepare_(problem);
    //     set_output_path(output_path);

    //     analyze_();
    // }

    Eigen::VectorXd solve(problem& problem) {
        prepare_(problem);
        Eigen::VectorXd result = solve_(problem);
        // std::cout << "Estimated error: " << solver_.info() << std::endl;
        std::cout << "#Solver: iterations:     " << solver_.iterations() << std::endl;
        std::cout << "#Solver: estimated error: " << solver_.error()      << std::endl;
        return result;
    }

    // void solve(problem& problem, const char* output_path = nullptr) {

    //     prepare_(problem);

    //     if (output_path != nullptr) {
    //         set_output_path(output_path);
    //     }

    //     analyze_();
    //     // std::cout << "Analysis type: " << infog(32) << std::endl;
    //     // std::cout << "Ordering used: " << infog(7) << std::endl;
    //     // report_after_analysis(std::cout);
    //     factorize_();
    //     // std::cout << "Deficiency: " << infog(28) << std::endl;
    //     solve_();
    // }

    // double flops_assembly() const { return rinfog(2); }

    // double flops_elimination() const { return rinfog(3); }

    ~solver() { }

private:
    void prepare_(problem& problem) {
        if (static_cast<int> (problem.a().nonZeros()) == 0) problem.prepare_data();
        if (max_iter_) solver_.setMaxIterations(max_iter_);
    }

    Eigen::VectorXd solve_(problem& problem) {
        // id.job = 3;
        // dmumps_c(&id);
        // print_state("After solve_()");
        return solver_.compute(problem.a()).solve(problem.rhs());
    }

    // void report_after_analysis(std::ostream& os) const {
    //     os << "MUMPS (" << id.version_number << ") after analysis:" << std::endl;
    //     os << "RINFO:" << std::endl;
    //     os << "  Estimated FLOPS for the elimination:          " << rinfo(1) << std::endl;
    //     os << "  Disk space for out-of-core factorization:     " << rinfo(5) << " MB" << std::endl;
    //     os << "  Size of the file used to save data:           " << rinfo(7) << " MB" << std::endl;
    //     os << "  Size of the MUMPS structure:                  " << rinfo(8) << " MB" << std::endl;
    //     os << "INFO:" << std::endl;
    //     os << "  Success:                                      " << info(1) << std::endl;
    //     auto real_store = handle_neg(info(3));
    //     os << "  Size of the real space to store factors:      " << real_store << " ("
    //        << as_MB(real_store) << " MB)" << std::endl;
    //     os << "  Size of the integer space to store factors:   " << info(4) << std::endl;
    //     os << "  Estimated maximum front size:                 " << info(5) << std::endl;
    //     os << "  Number of nodes in a tree:                    " << info(6) << std::endl;
    //     os << "  Size of the integer space to factorize:       " << info(7) << std::endl;
    //     auto real_factor = handle_neg(info(8));
    //     os << "  Size of the real space to factorize:          " << real_factor << " ("
    //        << as_MB(real_factor) << " MB)" << std::endl;
    //     os << "  Total memory needed:                          " << info(15) << " MB" << std::endl;
    //     os << "  Total memory needed (OoC):                    " << info(17) << " MB" << std::endl;
    //     os << "  Size of the integer space to factorize (OoC): " << info(19) << std::endl;
    //     auto real_factor_ooc = handle_neg(info(20));
    //     os << "  Size of the real space to factorize (OoC):    " << real_factor_ooc << " ("
    //        << as_MB(real_factor_ooc) << " MB)" << std::endl;
    //     os << "  Estimated number of entries in factors:       " << handle_neg(info(24))
    //        << std::endl;
    //     auto low_real_factor = handle_neg(info(29));
    //     os << "  Size of the real space to factorize (low-r):  " << low_real_factor << " ("
    //        << as_MB(low_real_factor) << " MB)" << std::endl;
    //     os << "  Total memory needed (low-rank):               " << info(30) << " MB" << std::endl;
    //     os << "  Total memory needed (low-rank, OoC):          " << info(31) << " MB" << std::endl;
    // }
};

}  // namespace ads::eigen

#endif  // defined(ADS_USE_EIGEN)

#endif  // ADS_SOLVER_EIGEN_HPP
