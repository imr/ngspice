/*
 * Copyright 2019 Michael Tesch.  All rights reserved.
 *
 * author(s): michael tesch (tesch1@gmail.com)
 *
 */
/*! \file       bench_fmt.cpp
 * \brief       benchmarking for formatting
 *
 * To disable freq scaling:
 *   cpupower frequency-set --governor performance
 * to re-enable freq scaling:
 *   cpupower frequency-set --governor ondemand
 */

#include <benchmark/benchmark.h>
#include <complex>
#define CPPDUALS_LIBFMT
#define CPPDUALS_LIBFMT_COMPLEX
#include "duals/dual"

template <class T> void B_fmt_1(benchmark::State& state) {
  T c(3.4);
  for (auto _ : state) {
    std::string s;
    benchmark::DoNotOptimize(s = fmt::format("{}", c));
  }
}

template <class T> void B_ios_1(benchmark::State& state) {
  T c(3.4);
  for (auto _ : state) {
    std::string s;
    std::stringstream ss;
    ss << c;
    s = ss.str();
  }
}

template <class T> void B_fmt(benchmark::State& state) {
  T c(3.4, 5.6);
  for (auto _ : state) {
    std::string s;
    benchmark::DoNotOptimize(s = fmt::format("{}", c));
  }
}

template <class T> void B_fmt_g(benchmark::State& state) {
  T c(3.4, 5.6);
  for (auto _ : state) {
    std::string s;
    benchmark::DoNotOptimize(s = fmt::format("{:g}", c));
  }
}

template <class T> void B_fmt_star_g(benchmark::State& state) {
  T c(3.4, 5.6);
  for (auto _ : state) {
    std::string s;
    benchmark::DoNotOptimize(s = fmt::format("{:*g}", c));
  }
}

template <class T> void B_fmt_comma_g(benchmark::State& state) {
  T c(3.4, 5.6);
  for (auto _ : state) {
    std::string s;
    benchmark::DoNotOptimize(s = fmt::format("{:,g}", c));
  }
}

template <class T> void B_ios(benchmark::State& state) {
  T c(3.4, 5.6);
  for (auto _ : state) {
    std::string s;
    std::stringstream ss;
    ss << c;
    s = ss.str();
  }
}

BENCHMARK_TEMPLATE(B_fmt_1, float);
BENCHMARK_TEMPLATE(B_fmt_1, double);
BENCHMARK_TEMPLATE(B_ios_1, float);
BENCHMARK_TEMPLATE(B_ios_1, double);
BENCHMARK_TEMPLATE(B_fmt_g, std::complex<float>);
BENCHMARK_TEMPLATE(B_fmt_star_g, std::complex<float>);
BENCHMARK_TEMPLATE(B_fmt_comma_g, std::complex<float>);

BENCHMARK_TEMPLATE(B_fmt, std::complex<float>);
BENCHMARK_TEMPLATE(B_fmt, std::complex<double>);
BENCHMARK_TEMPLATE(B_fmt, duals::dual<float>);
BENCHMARK_TEMPLATE(B_fmt, duals::dual<double>);

BENCHMARK_TEMPLATE(B_ios, std::complex<float>);
BENCHMARK_TEMPLATE(B_ios, std::complex<double>);
BENCHMARK_TEMPLATE(B_ios, duals::dual<float>);
BENCHMARK_TEMPLATE(B_ios, duals::dual<double>);

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char** argv)
{
  std::ios::sync_with_stdio(false);
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
