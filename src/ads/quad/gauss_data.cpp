// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#include "ads/quad/gauss.hpp"

namespace ads::quad::gauss {

const double* const Xs[] = {
    nullptr,           nullptr,           gauss_data<2>::X,  gauss_data<3>::X,  gauss_data<4>::X,
    gauss_data<5>::X,  gauss_data<6>::X,  gauss_data<7>::X,  gauss_data<8>::X,  gauss_data<9>::X,
    gauss_data<10>::X, gauss_data<11>::X, gauss_data<12>::X, gauss_data<13>::X, gauss_data<14>::X,
    gauss_data<15>::X, gauss_data<16>::X, gauss_data<17>::X, gauss_data<18>::X, gauss_data<19>::X,
    gauss_data<20>::X, gauss_data<21>::X, gauss_data<22>::X, gauss_data<23>::X, gauss_data<24>::X,
    gauss_data<25>::X, gauss_data<26>::X, gauss_data<27>::X, gauss_data<28>::X, gauss_data<29>::X,
    gauss_data<30>::X, gauss_data<31>::X, gauss_data<32>::X, gauss_data<33>::X, gauss_data<34>::X,
    gauss_data<35>::X, gauss_data<36>::X, gauss_data<37>::X, gauss_data<38>::X, gauss_data<39>::X,
    gauss_data<40>::X, gauss_data<41>::X, gauss_data<42>::X, gauss_data<43>::X, gauss_data<44>::X,
    gauss_data<45>::X, gauss_data<46>::X, gauss_data<47>::X, gauss_data<48>::X, gauss_data<49>::X,
    gauss_data<50>::X, gauss_data<51>::X, gauss_data<52>::X, gauss_data<53>::X, gauss_data<54>::X,
    gauss_data<55>::X, gauss_data<56>::X, gauss_data<57>::X, gauss_data<58>::X, gauss_data<59>::X,
    gauss_data<60>::X, gauss_data<61>::X, gauss_data<62>::X, gauss_data<63>::X, gauss_data<64>::X,
};

const double* const Ws[] = {
    nullptr,           nullptr,           gauss_data<2>::W,  gauss_data<3>::W,  gauss_data<4>::W,
    gauss_data<5>::W,  gauss_data<6>::W,  gauss_data<7>::W,  gauss_data<8>::W,  gauss_data<9>::W,
    gauss_data<10>::W, gauss_data<11>::W, gauss_data<12>::W, gauss_data<13>::W, gauss_data<14>::W,
    gauss_data<15>::W, gauss_data<16>::W, gauss_data<17>::W, gauss_data<18>::W, gauss_data<19>::W,
    gauss_data<20>::W, gauss_data<21>::W, gauss_data<22>::W, gauss_data<23>::W, gauss_data<24>::W,
    gauss_data<25>::W, gauss_data<26>::W, gauss_data<27>::W, gauss_data<28>::W, gauss_data<29>::W,
    gauss_data<30>::W, gauss_data<31>::W, gauss_data<32>::W, gauss_data<33>::W, gauss_data<34>::W,
    gauss_data<35>::W, gauss_data<36>::W, gauss_data<37>::W, gauss_data<38>::W, gauss_data<39>::W,
    gauss_data<40>::W, gauss_data<41>::W, gauss_data<42>::W, gauss_data<43>::W, gauss_data<44>::W,
    gauss_data<45>::W, gauss_data<46>::W, gauss_data<47>::W, gauss_data<48>::W, gauss_data<49>::W,
    gauss_data<50>::W, gauss_data<51>::W, gauss_data<52>::W, gauss_data<53>::W, gauss_data<54>::W,
    gauss_data<55>::W, gauss_data<56>::W, gauss_data<57>::W, gauss_data<58>::W, gauss_data<59>::W,
    gauss_data<60>::W, gauss_data<61>::W, gauss_data<62>::W, gauss_data<63>::W, gauss_data<64>::W,
};

}  // namespace ads::quad::gauss
