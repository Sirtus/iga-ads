// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_SOLVER_EIGEN_HPP
#define ADS_SOLVER_EIGEN_HPP

#include "ads/config.hpp"

#ifdef ADS_USE_EIGEN

#    include <Eigen/Sparse>
#    include <unsupported/Eigen/IterativeSolvers>

#    include <cstdint>
#    include <cstdio>
#    include <iostream>
#    include <vector>

#    include "ads/util.hpp"

namespace ads::eigen {

struct problem {
    problem(double* rhs, int n, int matrix_space = 0)
    : rhs_data_{rhs}
    , n{n} {
        rhs_.resize(n+1);
        a_.resize(n+1, n+1);
        if (matrix_space) a_.reserve(Eigen::VectorXi::Constant(n+1,1000));
    }

    explicit problem(std::vector<double>& rhs)
    : rhs_data_{rhs.data()}
    , n{static_cast<int>(rhs.size())} { }


    void add(int row, int col, double value) {
        triplets_.push_back(Eigen::Triplet<double>(row, col, value));
    }

    int nonzero_entries() const { return  narrow_cast<int> (a_.nonZeros()); }

    int dofs() const { return n; }

    Eigen::SparseMatrix<double>& a() { return a_; }

    void prepare_data() {

        if (triplets_.size()) a_.setFromTriplets(triplets_.begin(), triplets_.end());

        if (rhs_data_) {
            for (int i = 0; i < n; ++i) rhs_(i) = rhs_data_[i];
        }
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
    std::vector<Eigen::Triplet<double>> triplets_;
};


class solver {
private:
    // Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver_;
    // Eigen::DGMRES<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver_;
    // Eigen::IDRS<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver_;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver_;
    // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::DiagonalPreconditioner<double>> solver_;
    int max_iter_ = 0;
    Eigen::VectorXd result;
public:
    solver() { }
    solver(int max_iter) : max_iter_{max_iter} { }
    solver(const solver&) = delete;
    solver& operator=(const solver&) = delete;
    solver(solver&&) = delete;
    solver& operator=(solver&&) = delete;

    void set_max_iter(int max_iter) { solver_.setMaxIterations(max_iter); }

    void set_tolerance(double tolerance) { solver_.setTolerance(tolerance); }

    void save_to_file(problem& problem, const char* output_path) {
        std::string a_path = output_path;
        std::string rhs_path = output_path;
        a_path += "_a.txt";
        rhs_path += "_rhs.txt";

        std::ofstream outputFile(a_path);
        if (outputFile.is_open()) {

            for (int k=0; k<problem.a().outerSize(); ++k) {
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

    void read_problem_from_file(problem& problem, const char* output_path) {
        std::string a_path = output_path;
        std::string rhs_path = output_path;
        a_path += "_a.txt";
        rhs_path += "_rhs.txt";
        auto& rhs = problem.rhs();
        std::vector<double> rhs_data;

        std::ifstream inputFile(a_path);

        if (inputFile.is_open()) {

            int row, col;
            double value;

            while (inputFile >> row >> col >> value) {
                problem.add(row, col, value);
            }

            inputFile.close();
            std::cout << "Matrix A read from: " << a_path << std::endl;
        } else {
            std::cerr << "Cannot open file " << a_path << std::endl;
            return;
        }

        std::ifstream inputFileRhs(rhs_path);
        if (inputFileRhs.is_open()) {
            double value;
            int ctr = 0;

            while (inputFileRhs >> value) {
                rhs(ctr) = value;
                ctr++;
            }

            inputFileRhs.close();
            std::cout << "Rhs read from: " << rhs_path << std::endl;
        } else {
            std::cerr << "Cannot open file " << rhs_path << std::endl;
            return;
        }
    }

    double* solve(problem& problem) {
        prepare_(problem);
        solve_(problem);
        std::cout << "#Solver: iterations:     " << solver_.iterations() << std::endl;
        std::cout << "#Solver: estimated error: " << solver_.error()      << std::endl;
        return result.data();
    }

    double* solveWithGuess(problem& problem, double* guess_data) {
        prepare_(problem);
        Eigen::VectorXd guess;
        guess.resize(problem.dofs()+1);
        for (int i = 0; i < problem.dofs(); ++i) guess(i) = guess_data[i]; 
        solve_(problem, guess);
        std::cout << "#Solver: iterations:     " << solver_.iterations() << std::endl;
        std::cout << "#Solver: estimated error: " << solver_.error()      << std::endl;
        return result.data();
    }

    ~solver() { }

private:
    void prepare_(problem& problem) {
        if (static_cast<int> (problem.a().nonZeros()) == 0) problem.prepare_data();
        if (max_iter_) solver_.setMaxIterations(max_iter_);
    }

    void solve_(problem& problem) { result = solver_.compute(problem.a()).solve(problem.rhs()); }

    void solve_(problem& problem, Eigen::VectorXd& guess) { result = solver_.compute(problem.a()).solveWithGuess(problem.rhs(), guess); }

};

}  // namespace ads::eigen

#endif  // defined(ADS_USE_EIGEN)

#endif  // ADS_SOLVER_EIGEN_HPP
