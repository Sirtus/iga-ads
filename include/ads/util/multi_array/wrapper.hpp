// SPDX-FileCopyrightText: 2015 - 2023 Marcin Łoś <marcin.los.91@gmail.com>
// SPDX-License-Identifier: MIT

#ifndef ADS_UTIL_MULTI_ARRAY_WRAPPER_HPP
#define ADS_UTIL_MULTI_ARRAY_WRAPPER_HPP

#include "ads/util/multi_array/base.hpp"
#include "ads/util/multi_array/ordering/standard.hpp"

namespace ads {

template <                                                  //
    typename T,                                             // element type
    std::size_t Rank,                                       // number of indices
    typename Buffer,                                        // storage type
    template <std::size_t> class Order = standard_ordering  // memory layout
    >
struct multi_array_wrapper
: multi_array_base<T, Rank, multi_array_wrapper<T, Rank, Buffer, Order>> {
    using Self = multi_array_wrapper<T, Rank, Buffer, Order>;
    using Base = multi_array_base<T, Rank, Self, Order>;
    using size_array = typename Base::size_array;

    Buffer data;

    multi_array_wrapper(Buffer s, const size_array& sizes)
    : Base{sizes}
    , data{s} { }

private:
    friend Base;

    T& storage_(std::size_t idx) { return data[idx]; }

    const T& storage_(std::size_t idx) const { return data[idx]; }
};

}  // namespace ads

#endif  // ADS_UTIL_MULTI_ARRAY_WRAPPER_HPP
