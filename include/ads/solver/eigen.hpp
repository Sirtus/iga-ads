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
        rhs_.resize(n+1);
        a_.resize(n+1, n+1);
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

    Eigen::SparseMatrix<double>& a() { return a_; }

    void prepare_data() {
        std::vector<Eigen::Triplet<double>> triplets;

        int triplets_num = static_cast<int> (values_.size());
        for (int i = 0; i < triplets_num; ++i) triplets.push_back(Eigen::Triplet<double>(rows_[i], cols_[i], values_[i]));
        a_.setFromTriplets(triplets.begin(), triplets.end());

        for (int i = 0; i < n; ++i) rhs_(i) = rhs_data_[i];
    }

    Eigen::VectorXd& rhs() { return rhs_; }

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
};


class solver {
private:
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver_;
    int max_iter_ = 0;
public:
    solver() { }
    solver(int max_iter) : max_iter_{max_iter} { }
    solver(const solver&) = delete;
    solver& operator=(const solver&) = delete;
    solver(solver&&) = delete;
    solver& operator=(solver&&) = delete;

    void set_max_iter(int max_iter) { max_iter_ = max_iter; }

    void save_to_file(problem& problem, const char* output_path) {
        std::string a_path = output_path;
        std::string rhs_path = output_path;
        a_path += "_a.txt";
        rhs_path += "_rhs.txt";
        
        std::ofstream outputFile(a_path);
        if (outputFile.is_open()) {

            outputFile << "ROW   |   COL   |   VALUE \n";
            for (int k=0; k<3; ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator a_el(problem.a(),k); a_el; ++a_el) {
                    outputFile << a_el.row() << " " << a_el.col() << " " << a_el.value() << std::endl;
                }
            }

            outputFile.close();
            std::cout << "Matrix A saved to: " << a_path << std::endl;
        } else {
            std::cerr << "Cannot open file " << a_path << std::endl;
            return;
        }

        std::ofstream outputFileRhs(rhs_path);
        if (outputFileRhs.is_open()) {
            outputFileRhs << problem.rhs();
            
            outputFileRhs.close();
            std::cout << "Rhs saved to: " << rhs_path << std::endl;
        } else {
            std::cerr << "Cannot open file " << rhs_path << std::endl;
            return;
        }
    }

    double* solve(problem& problem) {
        prepare_(problem);
        double* result = solve_(problem);
        std::cout << "#Solver: iterations:     " << solver_.iterations() << std::endl;
        std::cout << "#Solver: estimated error: " << solver_.error()      << std::endl;
        return result;
    }

    ~solver() { }

private:
    void prepare_(problem& problem) {
        if (static_cast<int> (problem.a().nonZeros()) == 0) problem.prepare_data();
        if (max_iter_) solver_.setMaxIterations(max_iter_);
    }

    double* solve_(problem& problem) {
        Eigen::VectorXd x = solver_.compute(problem.a()).solve(problem.rhs());
        return x.data();
    }

};

}  // namespace ads::eigen

#endif  // defined(ADS_USE_EIGEN)

#endif  // ADS_SOLVER_EIGEN_HPP
