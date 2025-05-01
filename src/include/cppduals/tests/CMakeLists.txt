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

cmake_minimum_required (VERSION 3.14)

if (WIN32)
  add_definitions (-D_USE_MATH_DEFINES)
endif ()
include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("${DEPS_ROOT}/include")
#include_directories ("${MPFR_INCLUDES}")

#
# Correctness & Coverage
#
message ("OSX_ARCHITECTURES: ${CMAKE_SYSTEM_PROCESSOR}")
set (OSX_ARCHITECTURES "arm64;x86_64")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set (X86 TRUE)
else ()
  set (X86 FALSE)
endif ()

# Set optimization flags only if coverage is not enabled
if (NOT CPPDUALS_CODE_COVERAGE)
  if (X86)
    if (NOT MSVC)
      set (OPT_FLAGS "-O3;-march=native")
    else ()
      set (OPT_FLAGS "/arch:AVX2")
    endif ()
  else ()
    set (OPT_FLAGS "-O3")
  endif ()
else()
  if (X86)
    if (NOT MSVC)
      set (OPT_FLAGS "-mavx2")
    else ()
      set (OPT_FLAGS "/arch:AVX2")
    endif ()
  else ()
    set (OPT_FLAGS "")
  endif ()
endif ()

set (OPT_FLAGS "${OPT_FLAGS};-DCPPDUALS_VECTORIZE_CDUAL")

set (ALL_TESTS
  test_dual test_cdual test_funcs test_eigen test_packets
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
    string (REPLACE ";" ", " L2 "${OPT_FLAGS};${CMAKE_CXX_FLAGS}")
    target_compile_options (${TEST} PUBLIC ${PHASE_FLAGS})
    target_compile_definitions (${TEST} PRIVATE "OPT_FLAGS=${L2}")
    
    # Link with coverage config first to ensure flags are used
    target_link_libraries (${TEST} 
      cppduals_coverage_config  # Link coverage first
      gtest_main 
      eigen 
      expokit
    )
    
    gtest_discover_tests (${TEST} TEST_LIST ${TEST}_targets)
    set_tests_properties (${${TEST}_targets} PROPERTIES TIMEOUT 10)
  endforeach (PHASE)
endforeach (TEST_)

# special for fmt
target_link_libraries (test_fmt_1 fmt::fmt)

if (CPPDUALS_BENCHMARK)
  #
  # Benchmarks
  #
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
  endif ()
  find_package (BLAS REQUIRED)
  #find_package (LAPACK REQUIRED)
  add_definitions (-DHAVE_BLAS)
  #add_definitions (-DEIGEN_USE_BLAS)
  if (APPLE)
    add_definitions (-DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
  endif ()

  # find lapacke.h cblas.h
  set (CBLAS_HINTS ${BLAS_DIR} ${LAPACK_DIR} /usr /usr/local /opt /opt/local)
  set (CBLAS_PATHS
    /usr
    /usr/local
    /opt
    /opt/local
    /usr/local/opt
    /System/Library/Frameworks
    ${BLAS_LIBRARIES}
    )

  # Finds the include directories for lapacke.h
  find_path (LAPACKE_INCLUDE_DIRS
    REQUIRED
    NAMES lapacke.h clapack.h
    HINTS ${CBLAS_HINTS}
    PATH_SUFFIXES
    include inc include/x86_64 include/x64
    openblas/include openblas
    Frameworks/vecLib.framework/Headers
    PATHS ${CBLAS_PATHS}
    DOC "LAPACK(E) include header lapacke.h/clapack.h")
  mark_as_advanced (LAPACKE_INCLUDE_DIRS)
  if (LAPACKE_INCLUDE_DIRS)
    include_directories (${LAPACKE_INCLUDE_DIRS})
  else ()
    add_definitions (-DEIGEN_LAPACKE)
  endif (LAPACKE_INCLUDE_DIRS)

  # Finds the include directories for cblas*.h
  find_path (CBLAS_INCLUDE_DIRS
    REQUIRED
    NAMES cblas.h cblas_openblas.h cblas-openblas.h
    HINTS ${CBLAS_HINTS}
    PATH_SUFFIXES
    include inc include/x86_64 include/x64
    openblas/include openblas
    Frameworks/vecLib.framework/Headers
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

  set (BMK_FLAGS "")
  if (NOT MSVC)
    #set (BMK_FLAGS "-O3;-mavx")
    #set (BMK_FLAGS "-O3;-march=native;-fopenmp")
    if (X86)
      set (BMK_FLAGS "-msse3;-fopenmp")
      set (BMK_FLAGS "-msse3")
      set (BMK_FLAGS "-march=native;-funroll-loops")
      set (BMK_FLAGS "-msse3;-mavx2;-mfma")
    else ()
      set (BMK_FLAGS "-march=native")
    endif ()
    #set (BMK_FLAGS "-O3;${BMK_FLAGS}")
    #set (BMK_FLAGS "${BMK_FLAGS};-save-temps;-fverbose-asm")
  else ()
    set (BMK_FLAGS "/arch:IA32")
    set (BMK_FLAGS "/arch:SSE")
    set (BMK_FLAGS "/arch:SSE2")
    set (BMK_FLAGS "/arch:AVX2")
  endif ()

  #set (BMK_FLAGS "${BMK_FLAGS};-DEIGEN_DONT_VECTORIZE")
  #set (BMK_FLAGS "${BMK_FLAGS};-DCPPDUALS_DONT_VECTORIZE")
  #set (BMK_FLAGS "${BMK_FLAGS};-DCPPDUALS_DONT_VECTORIZE_CDUAL")

  foreach (VECTORIZE YES NO)
    foreach (BENCH  bench_dual bench_eigen bench_exp bench_gemm bench_example bench_fmt)
      if (NOT VECTORIZE)
        set (BENCHE ${BENCH})
      else ()
        set (BENCHE ${BENCH}_novec)
      endif ()
      add_executable (${BENCHE} ${BENCH}.cpp)
      target_compile_options (${BENCHE} PUBLIC ${BMK_FLAGS})
      if (NOT VECTORIZE)
        target_compile_options (${BENCHE} PUBLIC "-DEIGEN_DONT_VECTORIZE")
      endif()
      #set_target_properties (${BENCH} PROPERTIES LINK_FLAGS -fopenmp)
      #target_link_options (${BENCH} PUBLIC ${BMK_FLAGS})
      string (REPLACE ";" ", " L2 "${BMK_FLAGS} ${CMAKE_CXX_FLAGS}")
      target_compile_definitions (${BENCHE} PRIVATE "BMK_FLAGS=${L2}")
      target_link_libraries (${BENCHE} benchmark::benchmark ${BLAS_LIBRARIES} eigen expokit)
    endforeach ()
  endforeach ()

  target_link_libraries (bench_fmt fmt::fmt)
  target_link_libraries (bench_fmt_novec fmt::fmt)

endif (CPPDUALS_BENCHMARK)

add_executable (sandbox sandbox.cpp)
#target_compile_options (sandbox PUBLIC ${BMK_FLAGS})
target_compile_options (sandbox PUBLIC -DCPPDUALS_VECTORIZE_CDUAL)
if (MSVC)
  if (X86)
    target_compile_options (sandbox PUBLIC /arch:AVX2)
  endif ()
else ()
  if (X86)
    target_compile_options (sandbox PUBLIC -O1 -msse3 -mavx2 -mfma)
  else ()
    target_compile_options (sandbox PUBLIC -O1 )  # -mfpu=neon
  endif ()
endif ()

set_target_properties (sandbox PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries (sandbox PUBLIC cppduals_coverage_config eigen expokit)

#
# Generate coverage reports
#
if (CPPDUALS_CODE_COVERAGE)
  # Find required tools
  get_filename_component(CMAKE_CXX_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  set(GENHTML_PATH ${CMAKE_CXX_COMPILER_DIR}/genhtml)
  if(NOT EXISTS ${GENHTML_PATH})
    find_program(GENHTML_PATH genhtml REQUIRED)
  endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang
    # often in the same directory as clang - search there too
    get_filename_component(CMAKE_CXX_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(LLVM_COV_PATH ${CMAKE_CXX_COMPILER_DIR}/llvm-cov)
    set(LLVM_PROFDATA_PATH ${CMAKE_CXX_COMPILER_DIR}/llvm-profdata)
    if(NOT EXISTS ${LLVM_COV_PATH})
      find_program(LLVM_COV_PATH llvm-cov REQUIRED)
    endif()
    if(NOT EXISTS ${LLVM_PROFDATA_PATH})
      find_program(LLVM_PROFDATA_PATH llvm-profdata REQUIRED)
    endif()
    
    add_custom_target(cov
      DEPENDS ${ALL_TEST_BINS}
      COMMAND ${CMAKE_COMMAND} -E remove_directory coverage/profraw
      COMMAND ${CMAKE_COMMAND} -E make_directory coverage/profraw
      COMMAND ${CMAKE_COMMAND} --build . --target test "LLVM_PROFILE_FILE=coverage/profraw/%p.profraw"
      COMMAND ${LLVM_PROFDATA_PATH} merge -sparse coverage/profraw/*.profraw -o coverage/coverage.profdata
      COMMAND ${LLVM_COV_PATH} show
        -instr-profile=coverage/coverage.profdata
        -format=html -output-dir=coverage/html
        -show-line-counts-or-regions
        -show-instantiation-summary
        ${ALL_TEST_BINS}
      COMMAND ${LLVM_COV_PATH} report
        -instr-profile=coverage/coverage.profdata
        ${ALL_TEST_BINS}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC
    set(LCOV_PATH ${CMAKE_CXX_COMPILER_DIR}/lcov)
    if(NOT EXISTS ${LCOV_PATH})
      find_program(LCOV_PATH lcov REQUIRED)
    endif()
    # Extract GCC version number and use matching gcov version
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
      OUTPUT_VARIABLE GCC_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(GCOV_PATH ${CMAKE_CXX_COMPILER_DIR}/gcov-${GCC_VERSION})
    if(NOT EXISTS ${GCOV_PATH})
      find_program(GCOV_PATH gcov-${GCC_VERSION} REQUIRED)
    endif()
  
    add_custom_target(cov
      DEPENDS ${ALL_TEST_BINS}
      COMMAND ${CMAKE_COMMAND} -E make_directory coverage
      COMMAND ${LCOV_PATH} --directory . --zerocounters
      # Add initial capture before running tests
      COMMAND ${LCOV_PATH} --directory . --capture --initial --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.base --ignore-errors inconsistent
      # Run tests and capture coverage data
      COMMAND ctest --output-on-failure
      COMMAND ${LCOV_PATH} --directory . --capture --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.info --ignore-errors inconsistent
      # Combine baseline and test coverage data
      COMMAND ${LCOV_PATH} --add-tracefile coverage/coverage.base --add-tracefile coverage/coverage.info --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.total
      # Only look at the coverage of the tests and duals library
      COMMAND ${LCOV_PATH} --extract coverage/coverage.total '*/tests/*' '*/duals/*' --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.info
      # Remove unwanted paths
      #COMMAND ${LCOV_PATH} --remove coverage/coverage.total '/usr/*' --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.info
      #COMMAND ${LCOV_PATH} --remove coverage/coverage.info '*/thirdparty/*' --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.info
      #COMMAND ${LCOV_PATH} --remove coverage/coverage.info '*/googletest/*' --gcov-tool ${GCOV_PATH} --output-file coverage/coverage.info
      # Add --ignore-errors empty to prevent failure on empty coverage data
      COMMAND ${LCOV_PATH} --list coverage/coverage.info --ignore-errors empty --gcov-tool ${GCOV_PATH}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  else()
    message(FATAL_ERROR "No coverage tool found for ${CMAKE_CXX_COMPILER_ID}")
  endif()

  add_custom_target(cov-html
    DEPENDS cov
    COMMAND ${CMAKE_COMMAND} -E remove_directory coverage/html
    COMMAND ${GENHTML_PATH} --ignore-errors source coverage/coverage.info --legend --title "cppduals coverage"
      --output-directory=coverage/html
    COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated at coverage/html/index.html"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

endif ()
