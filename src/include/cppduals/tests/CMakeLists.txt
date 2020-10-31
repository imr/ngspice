#
# Part of the cppduals Project
#
# (c)2019 Michael Tesch. tesch1@gmail.com
#
# See https://gitlab.com/tesch1/cppduals/blob/master/LICENSE.txt for
# license information.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# gtest_discover_tests requires 3.10
cmake_minimum_required (VERSION 3.10)

# Configure google-test as a downloadable library.
include (GoogleTest)

if (WIN32)
  add_definitions (-D_USE_MATH_DEFINES)
endif ()
include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("${DEPS_ROOT}/include")
include_directories ("${EIGEN3_INCLUDE_DIRS}")
#include_directories ("${MPFR_INCLUDES}")
include_directories ("${EXPOKIT_INCLUDE_DIR}")

set (IOFORMAT "IOFormat(FullPrecision, DontAlignCols, \", \", \"\\$<SEMICOLON>\\n\", \"\", \"\", \"[\", \"]\")")
add_definitions (-DEIGEN_DEFAULT_IO_FORMAT=${IOFORMAT})
#add_definitions (-DEIGEN_DEFAULT_IO_FORMAT=EIGEN_IO_FORMAT_OCTAVE)

#
# Correctness & Coverage
#
if (NOT MSVC)
  set (OPT_FLAGS "-O2")
  set (OPT_FLAGS "-O3;-msse3")
  set (OPT_FLAGS "-O3;-mavx2;-mfma")
  set (OPT_FLAGS "-O3;-march=native")
else ()
  set (OPT_FLAGS "/arch:IA32")
  set (OPT_FLAGS "/arch:SSE")
  set (OPT_FLAGS "/arch:SSE2")
  set (OPT_FLAGS "/arch:AVX2")
endif ()

#set (OPT_FLAGS "${OPT_FLAGS};-fsanitize=address;-fno-omit-frame-pointer")
set (OPT_FLAGS "${OPT_FLAGS};-DCPPDUALS_VECTORIZE_CDUAL")
#set (OPT_FLAGS "${OPT_FLAGS};-DCPPDUALS_DONT_VECTORIZE")
#set (OPT_FLAGS "${OPT_FLAGS};-DEIGEN_DONT_VECTORIZE")

set (ALL_TESTS
  test_dual test_funcs test_eigen test_packets
  test_vectorize test_solve test_expm test_1 test_fmt
  example
  )
set (ALL_TEST_BINS )
foreach (TEST_ ${ALL_TESTS})
  foreach (PHASE 1 2 3 4 5)
    # check if file has phase defined
    file (READ ${TEST_}.cpp TMPTXT)
    string (FIND "${TMPTXT}" "PHASE_${PHASE}" hasphase)
    if ((${hasphase} EQUAL -1) AND NOT (${PHASE} EQUAL 1))
      continue ()
    endif ()

    set (TEST ${TEST_}_${PHASE})
    message ("Adding test ${TEST}")
    set (ALL_TEST_BINS ${ALL_TEST_BINS} ${TEST})

    add_executable (${TEST} ${TEST_}.cpp)
    set (PHASE_FLAGS ${OPT_FLAGS} -DPHASE_${PHASE})
    string (REPLACE ";" ", " L2 "${CMAKE_CXX_FLAGS}")
    target_compile_options (${TEST} PUBLIC ${PHASE_FLAGS})
    target_compile_definitions (${TEST} PRIVATE "OPT_FLAGS=${L2}")
    target_link_libraries (${TEST} gtest_main cppduals_coverage_config)
    #target_link_libraries (${TEST} -lasan)
    add_dependencies (${TEST} eigenX expokitX)
    gtest_discover_tests (${TEST} TEST_LIST ${TEST}_targets)
    set_tests_properties (${${TEST}_targets} PROPERTIES TIMEOUT 10)
    # -ftest-coverage
  endforeach (PHASE)
endforeach (TEST_)

# special for fmt
target_compile_features (test_fmt_1 PUBLIC cxx_std_14)
target_link_libraries (test_fmt_1 fmt::fmt)

if (CPPDUALS_BENCHMARK)
  #
  # Benchmarks
  #

  message ("searching: ${DEPS_ROOT} for google benchmark libs")
  find_library (BENCHMARK_LIBRARY benchmark PATHS ${DEPS_ROOT}/lib)
  find_library (BENCHMARKM_LIBRARY benchmark_main PATHS ${DEPS_ROOT}/lib)
  #find_library (PTHREAD_LIBRARY pthread)
  message ("BENCHMARK_LIBRARY: ${BENCHMARK_LIBRARY}")
  include_directories ("${BENCHMARK_INC_DIR}")
  if (Boost_FOUND AND NO)
    add_definitions (-DHAVE_BOOST=1)
    include_directories ("${Boost_INCLUDE_DIRS}")
    include_directories ("${PIRANHA_INCLUDE_DIR}")
    include_directories ("${AUDI_INCLUDE_DIR}")
  endif (Boost_FOUND AND NO)

  if (NOT APPLE AND NOT BLA_VENDOR)
    if (NOT BLA_STATIC)
      # default to static
      set (BLA_STATIC OFF)
    endif (NOT BLA_STATIC)
    set (BLA_VENDOR OpenBLAS)
  endif (NOT APPLE AND NOT BLA_VENDOR)
  find_package (BLAS REQUIRED)
  #find_package (LAPACK REQUIRED)
  add_definitions (-DHAVE_BLAS)
  #add_definitions (-DEIGEN_USE_BLAS)

  # find lapacke.h cblas.h
  set (CBLAS_HINTS ${BLAS_DIR} ${LAPACK_DIR} /usr /usr/local /opt /opt/local)
  set (CBLAS_PATHS
    /usr
    /usr/local
    /opt
    /opt/local
    /usr/local/opt
    /System/Library/Frameworks)

  # Finds the include directories for lapacke.h
  find_path (LAPACKE_INCLUDE_DIRS
    NAMES lapacke.h
    HINTS ${CBLAS_HINTS}
    PATH_SUFFIXES
    include inc include/x86_64 include/x64
    openblas/include
#    Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers
    PATHS ${CBLAS_PATHS}
    DOC "LAPACK(E) include header lapacke.h")
  mark_as_advanced (LAPACKE_INCLUDE_DIRS)
  if (LAPACKE_INCLUDE_DIRS)
    include_directories (${LAPACKE_INCLUDE_DIRS})
  else ()
    add_definitions (-DEIGEN_LAPACKE)
  endif (LAPACKE_INCLUDE_DIRS)

  # Finds the include directories for cblas*.h
  find_path (CBLAS_INCLUDE_DIRS
    NAMES cblas.h cblas_openblas.h cblas-openblas.h
    HINTS ${CBLAS_HINTS}
    PATH_SUFFIXES
    include inc include/x86_64 include/x64
    openblas/include
#    Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers
    PATHS ${CBLAS_PATHS}
    DOC "BLAS include header cblas.h")
  mark_as_advanced (CBLAS_INCLUDE_DIRS)
  include_directories (${CBLAS_INCLUDE_DIRS})
  foreach (cblas in cblas.h cblas_openblas.h cblas-openblas.h)
    if (EXISTS "${CBLAS_INCLUDE_DIRS}/${cblas}")
      add_definitions (-DCBLAS_HEADER="${cblas}")
      break()
    endif (EXISTS "${CBLAS_INCLUDE_DIRS}/${cblas}")
  endforeach (cblas)

  message ("Found BLAS    : ${BLAS_LIBRARIES}")
  message ("Found cBLAS   : ${CBLAS_INCLUDE_DIRS}")
  message ("Found lapacke : ${LAPACKE_INCLUDE_DIRS}")

  set (OPT_FLAGS "")
  if (NOT MSVC)
    #set (OPT_FLAGS "-O3;-mavx")
    #set (OPT_FLAGS "-O3;-march=native;-fopenmp")
    set (OPT_FLAGS "-O3;-msse3;-fopenmp")
    set (OPT_FLAGS "-O3")
    set (OPT_FLAGS "-O3;-msse3")
    set (OPT_FLAGS "-O3;-march=native;-funroll-loops")
    set (OPT_FLAGS "-O3;-msse3;-mavx2;-mfma")
    set (OPT_FLAGS "-O3;-march=native")
    #set (OPT_FLAGS "${OPT_FLAGS};-save-temps;-fverbose-asm")
  else ()
    set (OPT_FLAGS "/arch:IA32")
    set (OPT_FLAGS "/arch:SSE")
    set (OPT_FLAGS "/arch:SSE2")
    set (OPT_FLAGS "/arch:AVX2")
  endif ()

  #set (OPT_FLAGS "${OPT_FLAGS};-DEIGEN_DONT_VECTORIZE")
  #set (OPT_FLAGS "${OPT_FLAGS};-DCPPDUALS_DONT_VECTORIZE")
  #set (OPT_FLAGS "${OPT_FLAGS};-DCPPDUALS_DONT_VECTORIZE_CDUAL")

  foreach (BENCH  bench_dual bench_eigen bench_gemm bench_example bench_fmt)
    add_executable (${BENCH} ${BENCH}.cpp)
    target_compile_options (${BENCH} PUBLIC ${OPT_FLAGS})
    #set_target_properties (${BENCH} PROPERTIES LINK_FLAGS -fopenmp)
    #target_link_options (${BENCH} PUBLIC ${OPT_FLAGS})
    string (REPLACE ";" ", " L2 "${OPT_FLAGS} ${CMAKE_CXX_FLAGS}")
    target_compile_definitions (${BENCH} PRIVATE "OPT_FLAGS=${L2}")
    add_dependencies (${BENCH} benchmarkX eigenX expokitX)
    target_link_libraries (${BENCH} ${BENCHMARK_LIBRARY} -lpthread ${BLAS_LIBRARIES})
  endforeach (BENCH)

  target_link_libraries (bench_fmt fmt::fmt)

endif (CPPDUALS_BENCHMARK)

add_executable (sandbox sandbox.cpp)
add_dependencies (sandbox eigenX expokitX ) # mpfrX mprealX
#target_compile_options (sandbox PUBLIC ${OPT_FLAGS})
target_compile_options (sandbox PUBLIC -DCPPDUALS_VECTORIZE_CDUAL)
if (MSVC)
  target_compile_options (sandbox PUBLIC /arch:AVX2)
else ()
  target_compile_options (sandbox PUBLIC -O1 -msse3 -mavx2 -mfma)
endif ()

set_target_properties (sandbox PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries (sandbox PUBLIC cppduals_coverage_config)

#
# Generate coverage reports
#
if (CODE_COVERAGE)
  add_custom_target (cov
    DEPENDS ${ALL_TEST_BINS}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test
    COMMAND $<TARGET_FILE:sandbox>
    COMMAND lcov --capture --directory . --output-file coverage.info
    COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
    COMMAND lcov --remove coverage.info '*/thirdparty/*' --output-file coverage.info
    COMMAND lcov --remove coverage.info '*/googletest/*' --output-file coverage.info
    COMMAND lcov --list coverage.info
    )
  add_custom_target (cov-html
    DEPENDS cov
    COMMAND rm -rf ../coverage
    COMMAND genhtml --ignore-errors source coverage.info --legend --title "make cov"
    --output-directory=../coverage
    COMMAND echo "output in coverage/index.html"
    )
endif ()
