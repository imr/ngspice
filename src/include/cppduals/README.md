cppduals
========

Header-only dual number library for C++11.  The `dual<>` type can be
used for automatic (forward) differentiation.  It can be used in
conjunction with Eigen to produced very fast vectorized computations
of real and complex matrix functions and their derivatives.

There is a small paper on cppduals here:
[![DOI](https://joss.theoj.org/papers/10.21105/joss.01487/status.svg)](https://doi.org/10.21105/joss.01487)

Documentation
=============

[Full documentation is here](https://tesch1.gitlab.io/cppduals/).

The dual number space is closely related to the complex number space,
and as such, the dual class `duals::dual<>` is similar to
`std::complex<>`.

When compiling with Eigen it is possible to disable the vectorization
templates by `#define CPPDUALS_DONT_VECTORIZE`.  This may be useful if
your compiler is particularly good at optimizing Eigen expressions, I
have had mixed results, sometimes there are differences between the
compiler's best (GCC and Clang) and the vectorized code of 30% or
more, in either direction.

Examples
========

Here we calculate a function $`f(x)`$, with its derivative $`f'(x)`$,
calculated explicitly as `df()`, and calculated by using the dual
class (`1_e` returns the dual number $`0 + 1 \epsilon`$, it is
equivalent to `dual<double>(0,1)`):

```cpp
#include <duals/dual>

using namespace duals::literals;

template <class T> T   f(T x) { return pow(x,pow(x,x)); }
template <class T> T  df(T x) { return pow(x,-1. + x + pow(x,x)) * (1. + x*log(x) + x*pow(log(x),2.)); }
template <class T> T ddf(T x) { return (pow(x,pow(x,x)) * pow(pow(x,x - 1.) + pow(x,x)*log(x)*(log(x) + 1.), 2.) +
                                        pow(x,pow(x,x)) * (pow(x,x - 1.) * log(x) +
                                                           pow(x,x - 1.) * (log(x) + 1.) +
                                                           pow(x,x - 1.) * ((x - 1.)/x + log(x)) +
                                                           pow(x,x) * log(x) * pow(log(x) + 1., 2.) )); }

int main()
{
  std::cout << "  f(2.)            = " << f(2.)    << "\n";
  std::cout << " df(2.)            = " << df(2.)   << "\n";
  std::cout << "ddf(2.)            = " << ddf(2.)  << "\n";
  std::cout << "  f(2+1_e)         = " << f(2+1_e) << "\n";
  std::cout << "  f(2+1_e).dpart() = " << f(2+1_e).dpart() << "\n";
  duals::hyperduald x(2+1_e,1+0_e);
  std::cout << "  f((2+1_e) + (1+0_e)_e).dpart().dpart() = " << f(x).dpart().dpart() << "\n";
}
```

Produces:

```
  f(2.)            = 16
 df(2.)            = 107.11
ddf(2.)            = 958.755
  f(2+1_e)         = (16+107.11_e)
  f(2+1_e).dpart() = 107.11
  f((2+1_e) + (1+0_e)_e).dpart().dpart() = 958.755
```

Installation
============

Copy the [duals/](./duals/) directory (or just [dual](./duals/dual) )
somewhere your `#include`s can find it.  Then just `#include
<duals/dual[_eigen]>` from your source.

Alternatively, `cppduals` supports building with CMake. If using CMake v3.14+,
the ``FetchContent`` pattern is straightforward and enables using CMake targets
to specify library dependencies:

```cmake
  include(FetchContent)

  # Have CMake download the library
  set (CPPDUALS_TAG v0.4.1)
  set (CPPDUALS_MD5 7efe49496b8d0e3d3ffbcd3c68f542f3)
  FetchContent_Declare (cppduals
    URL https://gitlab.com/tesch1/cppduals/-/archive/${CPPDUALS_TAG}/cppduals-${CPPDUALS_TAG}.tar.bz2
    URL_HASH MD5=${CPPDUALS_MD5}
    )
  FetchContent_MakeAvailable (cppduals)

  # Link to cppduals
  target_link_libraries (your_target PRIVATE cppduals::duals)
```

Older versions of CMake can achieve a similar result using the ``ExternalProject``
family of commands and modifying the global preprocessor search path:

```cmake
  include(ExternalProject)

  # Have CMake download the library headers only
  set (CPPDUALS_TAG v0.4.1)
  set (CPPDUALS_MD5 7efe49496b8d0e3d3ffbcd3c68f542f3)
  ExternalProject_Add (cppduals
    URL https://gitlab.com/tesch1/cppduals/-/archive/${CPPDUALS_TAG}/cppduals-${CPPDUALS_TAG}.tar.bz2
    URL_HASH MD5=${CPPDUALS_MD5}
    CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "" )

  # Make include directory globally visible
  ExternalProject_Get_Property (cppduals source_dir)
  include_directories (${source_dir}/)
```

Alternatively, `cppduals` supports installation and discovery via the
`find_package` utility. First, download and install the library to a
location of your choosing:

```sh
  CPPDUALS_PREFIX=<desired_install_location>
  git clone https://gitlab.com/tesch1/cppduals.git && cd cppduals
  mkdir build && cd build
  cmake -DCMAKE_INSTALL_PREFIX="$CPPDUALS_PREFIX" ..
  cmake --build . --target install
```

Then, in your project's `CMakeLists.txt`, find and link to the library in the
standard manner:

```cmake
  find_package(cppduals REQUIRED)
  target_link_libraries(your_target PRIVATE cppduals::cppduals)
```

If you installed `cppduals` to a location that is not on `find_package`'s
default search path, you can specify the location by setting the `cppduals_DIR`
environment variable when configuring your project:

```sh
  cd your_build_dir
  cppduals_DIR="${CPPDUALS_PREFIX}" cmake ..
```


Benchmarks
==========

The benchmark compares cppduals against a local BLAS implementation,
by default OpenBLAS (whose development package is required;
RedHat-flavor: `openblas-devel`, Debian-flavor: `openblas-dev`).  If
you wish to build the benchmarks against a different installation of
BLAS, the following CMake variables can be set at configuration time:

- [BLA_VENDOR](https://cmake.org/cmake/help/latest/module/FindBLAS.html)
- BLAS_DIR
- LAPACK_DIR

For example, to build and run the tests shown below:

```sh
  cmake -Bbuild-bench -H. -DCPPDUALS_BENCHMARK=ON -DBLAS_DIR=/opt/local -DLAPACK_DIR=/opt/local
  cmake --build build-bench --target bench_gemm
  ./build-bench/tests/bench_gemm
```

The first performance goal of this project is to make the
`duals::dual<>` type at least as fast as `std::complex<>`.  This is
considered to be an upper-bound for performance because complex math
operations are usually highly optimized for scientific computing and
have a similar algebraic structure.  The second goal is to make the
compound type `std::complex<duals::dual<>>` as fast as possible for
use in calculation that require the derivative of complex functions
(ie comprising quantum-mechanical wave functions).

The first goal is measured by comparing the speed of matrix-matrix
operations (nominally matrix multiplication) on `duals::dual<>`-valued
Eigen matrices with highly optimtimized BLAS implementations of
equivalent operations on complex-valued matrices.  This can be done by
running the [./tests/bench_gemm](./tests/bench_gemm.cpp) program.  In
the *ideal* case, the results of the `B_MatMat<dual{f,d}>` type should
be nearly as fast, or faster than equivalently sized
`B_MatMat<complex{f,d}>`, and double-sized
`B_MatMatBLAS<{float,double}>` operations.  This is very difficult to
achieve in reality, as the BLAS libraries typically use hand-tuned
assembly, where the Eigen libraries must strive to express the
calculation in a general form that the compiler can turn into optimal
code.

Comparing Eigen 3.3.7 and OpenBLAS 0.3.6 on an `Intel(R) Core(TM)
i7-7700 CPU @ 3.60GHz` is still sub-optimal, only achieving about half
the performance of the BLAS equivalent, and 90% of
`std::complex<float>`:

    B_MatMat<dualf,dualf>/32               5433 ns         5427 ns
    B_MatMat<dualf,dualf>/64              38478 ns        38433 ns
    B_MatMat<dualf,dualf>/128            299450 ns       298981 ns
    B_MatMat<dualf,dualf>/256           2365347 ns      2361566 ns
    B_MatMat<dualf,dualf>/512          18888220 ns     18857342 ns
    B_MatMat<dualf,dualf>/1024        151079955 ns    150856120 ns

    B_MatMat<complexf,complexf>/32         4963 ns         4955 ns
    B_MatMat<complexf,complexf>/64        36716 ns        36671 ns
    B_MatMat<complexf,complexf>/128      280870 ns       280346 ns
    B_MatMat<complexf,complexf>/256     2173791 ns      2170886 ns
    B_MatMat<complexf,complexf>/512    17493222 ns     17459890 ns
    B_MatMat<complexf,complexf>/1024  138498432 ns    138286283 ns

    B_MatMatBLAS<complexf>/32              4877 ns         4870 ns
    B_MatMatBLAS<complexf>/64             27722 ns        27691 ns
    B_MatMatBLAS<complexf>/128           177084 ns       176756 ns
    B_MatMatBLAS<complexf>/256          1268715 ns      1266445 ns
    B_MatMatBLAS<complexf>/512          9772184 ns      9726621 ns
    B_MatMatBLAS<complexf>/1024        75915016 ns     75432354 ns


The second benchmark of interest measures how well the nested
specializations `std::complex<duals::dual<>>` perform as matrix values
relative to using a BLAS library with an extended matrix to compute
the same value function.  This comparison is also made with the
[./tests/bench_gemm](./tests/bench_gemm.cpp) program.  The relevant
measures are `B_MatMat<cdual{f,d}>` and `B_MatMatBLAS<complex{f,d}>`
of twice the size.

On the same machine as above, using `std::complex<duals::dual<float>>`
(`cdualf`) shows a speed advantage over the BLAS approach, while using
only half the memory.  However, notice that the advantage decreases as
the matrices get larger, which ideally should not happen:

    B_MatMat<cdualf,cdualf>/16             2810 ns         2808 ns
    B_MatMat<cdualf,cdualf>/32            19900 ns        19878 ns
    B_MatMat<cdualf,cdualf>/64           151837 ns       151646 ns
    B_MatMat<cdualf,cdualf>/128         1174699 ns      1172931 ns
    B_MatMat<cdualf,cdualf>/256         9122903 ns      9110123 ns
    B_MatMat<cdualf,cdualf>/512        72575352 ns     72467264 ns

    B_MatMatBLAS<complexf>/32              4877 ns         4870 ns
    B_MatMatBLAS<complexf>/64             27722 ns        27691 ns
    B_MatMatBLAS<complexf>/128           177084 ns       176756 ns
    B_MatMatBLAS<complexf>/256          1268715 ns      1266445 ns
    B_MatMatBLAS<complexf>/512          9772184 ns      9726621 ns
    B_MatMatBLAS<complexf>/1024        75915016 ns     75432354 ns


Contributions
=============

Questions, bug reports, bug fixes, and contributions are welcome.
Simply submit an [Issue](https://gitlab.com/tesch1/cppduals/issues)
or [Merge Request](https://gitlab.com/tesch1/cppduals/merge_requests).

Contributors
------------

- [Nestor Demeure](https://gitlab.com/nestordemeure)
- [Jeff](https://github.com/flying-tiger)

Compiler notes
==============

XCode 11 (Apple Clang 11) is known to work.  Also various version of
g++.  Clang 8.0 appears to have some trouble with compiling the
optimized templates for Eigen, as evidenced by its propensity to
segfault when compiling the cppduals test programs.  Please submit
issues if you experience similar problems, with specifics of your
compiler and compilation flags.

License
=======

The primary header file `duals/dual` and testing and benchmarking code
is licensed under the following:

```
 Part of the cppduals project.
 https://tesch1.gitlab.io/cppduals

 (c)2019 Michael Tesch. tesch1@gmail.com

 See https://gitlab.com/tesch1/cppduals/blob/master/LICENSE.txt for
 license information.

 This Source Code Form is subject to the terms of the Mozilla
 Public License v. 2.0. If a copy of the MPL was not distributed
 with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
```

Eigen-derived
-------------

The support for Eigen vectorization, including `duals/dual_eigen` and
the architecture-specific vectorization files under `duals/arch` are
derived from the [Eigen
project](http://eigen.tuxfamily.org/index.php?title=Main_Page), and
thus licensed under [MPL-2](http://www.mozilla.org/MPL/2.0/FAQ.html) .

ChangeLog
=========

v0.4.1
======

- changed constexpr to FMT_CONSTEXPR in the dual<> and complex<>
  formatters to work with more compilers / lang standards.

v0.4.0
======

- cleaned-up release with fixes from v0.3.2.
- improved docs

v0.3.3+
=======

- ignore these, will be, trying to cleanup release tarballs, next
  stable will be v0.4.0

v0.3.2
======

- not actually tagged release
- fixed a bug in the `{fmt}` support, added docs for the same.
- added benchmarking for `{fmt}` vs iostreams.

v0.3.1
======

- forgot to bump the CMakeLists package version number in 0.3.0.

v0.3.0
======

- vastly improved cmake support, thanks to
  [Jeff](https://gitlab.com/flying-tiger).  The improvements required
  changing some CMake target names.
- Added basic optional [libfmt](https://github.com/fmtlib/fmt) support for
  duals::dual<> and std::complex<>, enabled with `#define`\ s

v0.2.0
======

- fixed build on VS2017
- save and restore signam and errno in {t,l}gamma
- fixes from Nestor D. for https://gitlab.com/tesch1/cppduals/issues/5 (spurious nan)

Todo
====

- Add multi-variate differentiation capability.
- Non-x86_64 (CUDA/AltiVec/HIP/NEON/...) vectorization.
- Higher-order derivatives.

