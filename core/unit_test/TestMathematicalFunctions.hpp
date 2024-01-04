//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <initializer_list>
#include <type_traits>

#include <cfloat>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||          \
    defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET) || \
    defined(KOKKOS_ENABLE_OPENACC)
#else
#define MATHEMATICAL_FUNCTIONS_HAVE_LONG_DOUBLE_OVERLOADS
#endif

#if defined KOKKOS_COMPILER_INTEL || \
    (defined(KOKKOS_COMPILER_NVCC) && KOKKOS_COMPILER_NVCC >= 1130)
#define MATHEMATICAL_FUNCTIONS_TEST_UNREACHABLE __builtin_unreachable();
#else
#define MATHEMATICAL_FUNCTIONS_TEST_UNREACHABLE
#endif

namespace KE = Kokkos::Experimental;

// clang-format off
template <class>
struct math_unary_function_return_type;
// Floating-point types
template <> struct math_unary_function_return_type<      float> { using type =       float; };
template <> struct math_unary_function_return_type<     double> { using type =      double; };
// Integral types
template <> struct math_unary_function_return_type<              bool> { using type = double; };
template <> struct math_unary_function_return_type<             short> { using type = double; };
template <> struct math_unary_function_return_type<    unsigned short> { using type = double; };
template <> struct math_unary_function_return_type<               int> { using type = double; };
template <> struct math_unary_function_return_type<      unsigned int> { using type = double; };
template <> struct math_unary_function_return_type<              long> { using type = double; };
template <> struct math_unary_function_return_type<     unsigned long> { using type = double; };
template <> struct math_unary_function_return_type<         long long> { using type = double; };
template <> struct math_unary_function_return_type<unsigned long long> { using type = double; };
template <class T>
using math_unary_function_return_type_t = typename math_unary_function_return_type<T>::type;
template <class, class>
struct math_binary_function_return_type;
template <> struct math_binary_function_return_type<             float,              float> { using type =       float; };
template <> struct math_binary_function_return_type<             float,             double> { using type =      double; };
template <> struct math_binary_function_return_type<             float,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<             float,              short> { using type =      double; };
template <> struct math_binary_function_return_type<             float,                int> { using type =      double; };
template <> struct math_binary_function_return_type<             float,               long> { using type =      double; };
template <> struct math_binary_function_return_type<             float,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<             float,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<             float,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<             float,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<             float, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<            double,              float> { using type =      double; };
template <> struct math_binary_function_return_type<            double,             double> { using type =      double; };
template <> struct math_binary_function_return_type<            double,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<            double,              short> { using type =      double; };
template <> struct math_binary_function_return_type<            double,                int> { using type =      double; };
template <> struct math_binary_function_return_type<            double,               long> { using type =      double; };
template <> struct math_binary_function_return_type<            double,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<            double,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<            double,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<            double,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<            double, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<             short,              float> { using type =      double; };
template <> struct math_binary_function_return_type<             short,             double> { using type =      double; };
template <> struct math_binary_function_return_type<             short,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<             short,              short> { using type =      double; };
template <> struct math_binary_function_return_type<             short,                int> { using type =      double; };
template <> struct math_binary_function_return_type<             short,               long> { using type =      double; };
template <> struct math_binary_function_return_type<             short,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<             short,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<             short,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<             short,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<             short, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<               int,              float> { using type =      double; };
template <> struct math_binary_function_return_type<               int,             double> { using type =      double; };
template <> struct math_binary_function_return_type<               int,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<               int,              short> { using type =      double; };
template <> struct math_binary_function_return_type<               int,                int> { using type =      double; };
template <> struct math_binary_function_return_type<               int,               long> { using type =      double; };
template <> struct math_binary_function_return_type<               int,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<               int,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<               int,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<               int,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<               int, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<              long,              float> { using type =      double; };
template <> struct math_binary_function_return_type<              long,             double> { using type =      double; };
template <> struct math_binary_function_return_type<              long,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<              long,              short> { using type =      double; };
template <> struct math_binary_function_return_type<              long,                int> { using type =      double; };
template <> struct math_binary_function_return_type<              long,               long> { using type =      double; };
template <> struct math_binary_function_return_type<              long,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<              long,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<              long,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<              long,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<              long, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,              float> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,             double> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,              short> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,                int> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,               long> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<         long long,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<         long long, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,              float> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,             double> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,              short> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,                int> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,               long> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<    unsigned short, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,              float> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,             double> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,              short> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,                int> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,               long> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<      unsigned int, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,              float> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,             double> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,              short> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,                int> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,               long> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<     unsigned long, unsigned long long> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,              float> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,             double> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,               bool> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,              short> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,                int> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,               long> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,          long long> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,     unsigned short> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,       unsigned int> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long,      unsigned long> { using type =      double; };
template <> struct math_binary_function_return_type<unsigned long long, unsigned long long> { using type =      double; };
template <class T, class U>
using math_binary_function_return_type_t = typename math_binary_function_return_type<T, U>::type;
// clang-format on
template <class T, class U, class V>
using math_ternary_function_return_type_t = math_binary_function_return_type_t<
    T, math_binary_function_return_type_t<U, V>>;

struct FloatingPointComparison {
 private:
  template <class T>
  KOKKOS_FUNCTION double eps(T) const {
    return DBL_EPSILON;
  }
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
  KOKKOS_FUNCTION
  KE::half_t eps(KE::half_t) const {
// FIXME_NVHPC compile-time error
#ifdef KOKKOS_COMPILER_NVHPC
    return 0.0009765625F;
#else
    return KE::epsilon<KE::half_t>::value;
#endif
  }
#endif
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
  KOKKOS_FUNCTION
  KE::bhalf_t eps(KE::bhalf_t) const {
// FIXME_NVHPC compile-time error
#ifdef KOKKOS_COMPILER_NVHPC
    return 0.0078125;
#else
    return KE::epsilon<KE::bhalf_t>::value;
#endif
  }
#endif
  KOKKOS_FUNCTION
  double eps(float) const { return FLT_EPSILON; }
// POWER9 gives unexpected values with LDBL_EPSILON issues
// https://stackoverflow.com/questions/68960416/ppc64-long-doubles-machine-epsilon-calculation
#if defined(KOKKOS_ARCH_POWER9) || defined(KOKKOS_ARCH_POWER8)
  KOKKOS_FUNCTION
  double eps(long double) const { return DBL_EPSILON; }
#else
  KOKKOS_FUNCTION
  double eps(long double) const { return LDBL_EPSILON; }
#endif
  // Using absolute here instead of abs, since we actually test abs ...
  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<std::is_signed<T>::value, T> absolute(
      T val) const {
    return val < T(0) ? -val : val;
  }

  template <class T>
  KOKKOS_FUNCTION std::enable_if_t<!std::is_signed<T>::value, T> absolute(
      T val) const {
    return val;
  }

 public:
  template <class FPT>
  KOKKOS_FUNCTION bool compare_near_zero(FPT const& fpv, double ulp) const {
    auto abs_tol = eps(fpv) * ulp;

    bool ar = absolute(fpv) < abs_tol;
    if (!ar) {
      Kokkos::printf("absolute value exceeds tolerance [|%e| > %e]\n",
                     (double)fpv, abs_tol);
    }

    return ar;
  }

  template <class Lhs, class Rhs>
  KOKKOS_FUNCTION bool compare(Lhs const& lhs, Rhs const& rhs,
                               double ulp) const {
    if (lhs == 0) {
      return compare_near_zero(rhs, ulp);
    } else if (rhs == 0) {
      return compare_near_zero(lhs, ulp);
    } else {
      auto rel_tol     = (eps(lhs) < eps(rhs) ? eps(lhs) : eps(rhs)) * ulp;
      double abs_diff  = static_cast<double>(rhs > lhs ? rhs - lhs : lhs - rhs);
      double min_denom = static_cast<double>(
          absolute(rhs) < absolute(lhs) ? absolute(rhs) : absolute(lhs));
      double rel_diff = abs_diff / min_denom;
      bool ar         = abs_diff == 0 || rel_diff < rel_tol;
      if (!ar) {
        Kokkos::printf("relative difference exceeds tolerance [%e > %e]\n",
                       (double)rel_diff, rel_tol);
      }

      return ar;
    }
  }
};

template <class>
struct math_function_name;

#define DEFINE_UNARY_FUNCTION_EVAL(FUNC, ULP_FACTOR)                    \
  struct MathUnaryFunction_##FUNC {                                     \
    template <typename T>                                               \
    static KOKKOS_FUNCTION auto eval(T x) {                             \
      static_assert(                                                    \
          std::is_same<decltype(Kokkos::FUNC((T)0)),                    \
                       math_unary_function_return_type_t<T>>::value);   \
      return Kokkos::FUNC(x);                                           \
    }                                                                   \
    template <typename T>                                               \
    static auto eval_std(T x) {                                         \
      if constexpr (std::is_same<T, KE::half_t>::value ||               \
                    std::is_same<T, KE::bhalf_t>::value) {              \
        return std::FUNC(static_cast<float>(x));                        \
      } else {                                                          \
        static_assert(                                                  \
            std::is_same<decltype(std::FUNC((T)0)),                     \
                         math_unary_function_return_type_t<T>>::value); \
        return std::FUNC(x);                                            \
      }                                                                 \
      MATHEMATICAL_FUNCTIONS_TEST_UNREACHABLE                           \
    }                                                                   \
    static KOKKOS_FUNCTION double ulp_factor() { return ULP_FACTOR; }   \
  };                                                                    \
  using kk_##FUNC = MathUnaryFunction_##FUNC;                           \
  template <>                                                           \
  struct math_function_name<MathUnaryFunction_##FUNC> {                 \
    static constexpr char name[] = #FUNC;                               \
  };                                                                    \
  constexpr char math_function_name<MathUnaryFunction_##FUNC>::name[]

#ifndef KOKKOS_MATHEMATICAL_FUNCTIONS_SKIP_3
// Generally the expected ULP error should come from here:
// https://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html
// For now 1s largely seem to work ...
DEFINE_UNARY_FUNCTION_EVAL(exp, 2);
DEFINE_UNARY_FUNCTION_EVAL(exp2, 2);
#endif

// clang-format off
template <class>
struct type_helper;
#define DEFINE_TYPE_NAME(T) \
template <> struct type_helper<T> { static char const * name() { return #T; } };
DEFINE_TYPE_NAME(bool)
DEFINE_TYPE_NAME(int)
DEFINE_TYPE_NAME(long)
DEFINE_TYPE_NAME(long long)
DEFINE_TYPE_NAME(unsigned int)
DEFINE_TYPE_NAME(unsigned long)
DEFINE_TYPE_NAME(unsigned long long)
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
DEFINE_TYPE_NAME(KE::half_t)
#endif
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
DEFINE_TYPE_NAME(KE::bhalf_t)
#endif
DEFINE_TYPE_NAME(float)
DEFINE_TYPE_NAME(double)
DEFINE_TYPE_NAME(long double)
#undef DEFINE_TYPE_NAME
// clang-format on

template <class Space, class Func, class Arg, std::size_t N,
          class Ret = math_unary_function_return_type_t<Arg>>
struct TestMathUnaryFunction : FloatingPointComparison {
  Arg val_[N];
  Ret res_[N];
  TestMathUnaryFunction(const Arg (&val)[N], int line = __builtin_LINE()) {
    std::copy(val, val + N, val_);
    std::transform(val, val + N, res_,
                   [](auto x) { return Func::eval_std(x); });
    run(line);
  }
  void run(int line = __builtin_LINE()) {
    int errors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, N), *this, errors);
    ASSERT_EQ(errors, 0) << "Failed check no error for "
                         << math_function_name<Func>::name << "("
                         << type_helper<Arg>::name() << ")" << ", line " << line;
  }
  KOKKOS_FUNCTION void operator()(int i, int& e) const {
    bool ar = compare(Func::eval(val_[i]), res_[i], Func::ulp_factor());
    if (!ar) {
      ++e;
      Kokkos::printf("value at %f which is %f was expected to be %f\n",
                     (double)val_[i], (double)Func::eval(val_[i]),
                     (double)res_[i]);
    }
  }
};

template <class Space, class... Func, class Arg, std::size_t N>
void do_test_math_unary_function(const Arg (&x)[N]) {
  (void)std::initializer_list<int>{
      (TestMathUnaryFunction<Space, Func, Arg, N>(x), 0)...};

  // test if potentially device specific math functions also work on host
  if constexpr (!std::is_same_v<Space, Kokkos::DefaultHostExecutionSpace>)
    (void)std::initializer_list<int>{
        (TestMathUnaryFunction<Kokkos::DefaultHostExecutionSpace, Func, Arg, N>(
             x),
         0)...};
}

#define TEST_MATH_FUNCTION(FUNC) \
  do_test_math_unary_function<TEST_EXECSPACE, MathUnaryFunction_##FUNC>


#ifndef KOKKOS_MATHEMATICAL_FUNCTIONS_SKIP_3
TEST(TEST_CATEGORY, mathematical_functions_exponential_functions) {
  TEST_MATH_FUNCTION(exp)({-9, -8, -7, -6, -5, 4, 3, 2, 1, 0});

  TEST_MATH_FUNCTION(exp2)({-9, -8, -7, -6, -5, 4, 3, 2, 1, 0});

}
#endif
