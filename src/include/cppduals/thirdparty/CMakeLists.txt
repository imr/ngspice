
# https://gist.github.com/johnb003/65982fdc7a1274fdb023b0c68664ebe4

# 3.10 adds support for "gtest_discover_tests" which enumerates the tests inside
# of the code and adds them to ctest.
#
cmake_minimum_required (VERSION 3.10)
project (cppduals_thirdparty)
include (ExternalProject)

get_directory_property (hasParent PARENT_DIRECTORY)

set (DEPS_ROOT "${CMAKE_BINARY_DIR}/root")
if (hasParent)
  set (DEPS_ROOT "${CMAKE_BINARY_DIR}/thirdparty/root" PARENT_SCOPE)
endif (hasParent)

if (NOT WIN32)
  set (DOWNLOAD_DIR "$ENV{HOME}/Downloads")
else (NOT WIN32)
  set (DOWNLOAD_DIR "C:/Downloads")
endif (NOT WIN32)

#
# Google test (https://github.com/google/googletest/blob/master/googletest/README.md)
#

# Download and unpack googletest at configure time
configure_file (CMakeLists-gt.txt.in googletest-download/CMakeLists.txt)
execute_process (COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
  message (FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process (COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
  message (FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set (gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory (
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
  ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
  EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories ("${gtest_SOURCE_DIR}/include")
endif ()

# Can simply link against gtest or gtest_main as needed. Eg
#add_executable (example example.cpp)
#target_link_libraries (example gtest_main)
#add_test (NAME example_test COMMAND example)

#
# Eigen
#
if (CPPDUALS_EIGEN_LATEST)
  set (EIGEN_URL http://bitbucket.org/eigen/eigen/get/default.tar.bz2)
  #set (EIGEN_MD5 ffc83130dcd37b694c6cf7e905099af9)
else ()
  set (EIGEN_URL http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2)
  set (EIGEN_MD5 05b1f7511c93980c385ebe11bd3c93fa)
endif ()

ExternalProject_Add (eigenX
  PREFIX eigenX
  URL ${EIGEN_URL}
  #URL_HASH MD5=${EIGEN_MD5}
  DOWNLOAD_DIR "$ENV{HOME}/Downloads"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  )
ExternalProject_Get_Property (eigenX source_dir)
if (hasParent AND NOT EIGEN3_INCLUDE_DIRS)
  set (EIGEN3_INCLUDE_DIRS "${source_dir}" PARENT_SCOPE)
endif ()

#
# Eigen-Expokit
#
set (EEX_SHA 72bf6e445d5ae84218dcbd74580720491e0074db  )
ExternalProject_Add (expokitX
  PREFIX expokitX
  URL https://gitlab.com/api/v4/projects/tesch1%2Feigen-expokit/repository/archive.tbz2?sha=${EEX_SHA}
  #URL_HASH MD5=96b79de1d01547f6d658865b7caa02ee
  DOWNLOAD_DIR "$ENV{HOME}/Downloads"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  )
ExternalProject_Get_Property (expokitX source_dir)
if (hasParent)
  set (EXPOKIT_INCLUDE_DIR "${source_dir}" PARENT_SCOPE)
endif()

#
# fmt
#
ExternalProject_Add (fmtX
  PREFIX fmtX
  URL https://github.com/fmtlib/fmt/archive/6.1.1.tar.gz
  URL_HASH MD5=acfb83d44afdca171ee26c597c931e7c
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  )
ExternalProject_Get_Property (fmtX source_dir)
ExternalProject_Get_Property (fmtX binary_dir)
if (hasParent)
  message (" FMT3_INCLUDE_DIRS: ${source_dir}")
  add_subdirectory (${source_dir} ${binary_dir} EXCLUDE_FROM_ALL)
endif ()

if (CPPDUALS_BENCHMARK)
  #
  # google benchmark
  #
  ExternalProject_Add (benchmarkX
    PREFIX benchmarkX
    URL "http://github.com/google/benchmark/archive/v1.5.0.tar.gz"
    URL_HASH MD5=eb1466370f3ae31e74557baa29729e9e
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    CMAKE_ARGS --target install -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_USE_LIBCXX=${CPPDUALS_USE_LIBCXX}
    "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
    INSTALL_DIR "${DEPS_ROOT}"
    )
  ExternalProject_Get_Property (benchmarkX source_dir)
  ExternalProject_Get_Property (benchmarkX install_dir)
  if (hasParent)
    set (BENCHMARK_SRC_DIR "${source_dir}" PARENT_SCOPE)
    set (BENCHMARK_INC_DIR "${install_dir}/include" PARENT_SCOPE)
    message (" BENCHMARK_SRC_DIR: ${BENCHMARK_SRC_DIR}")
  endif()

  if (Boost_FOUND AND NO)

    #
    # Audi + Piranha - needs boost
    #
    #boost 1.70
    find_package (Boost 1.69)
    if (hasParent)
      set (Boost_FOUND ${Boost_FOUND} PARENT_SCOPE)
      set (Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} PARENT_SCOPE)
    endif ()

    # piranha
    ExternalProject_Add (piranhaX PREFIX piranhaX
      URL https://github.com/bluescarni/piranha/archive/v0.11.tar.gz
      URL_HASH MD5=33482f719f6b8a6a5316f9bd148f5b10
      DOWNLOAD_DIR "$ENV{HOME}/Downloads"
      CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
      )
    ExternalProject_Get_Property (piranhaX source_dir)
    if (hasParent)
      set (PIRANHA_INCLUDE_DIR "${source_dir}/include" PARENT_SCOPE)
    endif ()

    # AuDi
    ExternalProject_Add (audiX PREFIX audiX
      URL https://github.com/darioizzo/audi/archive/v1.6.5.tar.gz
      URL_HASH MD5=a51897469dfce4ceaa25e65519a346b9
      DOWNLOAD_DIR "$ENV{HOME}/Downloads"
      #CONFIGURE_COMMAND ""
      CMAKE_ARGS -DAUDI_BUILD_TESTS=OFF
      BUILD_COMMAND "" INSTALL_COMMAND ""
      )
    ExternalProject_Get_Property (audiX source_dir)
    if (hasParent)
      set (AUDI_INCLUDE_DIR "${source_dir}/include" PARENT_SCOPE)
    endif ()
  endif (Boost_FOUND AND NO)
endif (CPPDUALS_BENCHMARK)

#
# multi-precision math
#
if (0)
ExternalProject_Add (mpfrX
  PREFIX mpfrX
  URL https://www.mpfr.org/mpfr-current/mpfr-4.0.2.tar.bz2
  URL_HASH MD5=6d8a8bb46fe09ff44e21cdbf84f5cdac
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  CONFIGURE_COMMAND "../mpfrX/configure" --prefix=<INSTALL_DIR>
  #BUILD_COMMAND "make install"
  #INSTALL_COMMAND "make install"
  )
# clumsy download link - retrieves "archive.tar"
ExternalProject_Add (mprealX
  PREFIX mprealX
  URL "http://www.holoborodko.com/pavel/wp-content/plugins/download-monitor/download.php?id=4"
  URL_HASH MD5=68ad2258eb4a1c699c407f0e7bee5125
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  )
ExternalProject_Get_Property (mprealX source_dir)
ExternalProject_Get_Property (mpfrX INSTALL_DIR)
if (hasParent)
  set (MPFR_INCLUDES "${INSTALL_DIR}/include" "${source_dir}" PARENT_SCOPE)
  set (MPFR_LIBRARIES "${INSTALL_DIR}/lib/libmpfr.a" -lgmp PARENT_SCOPE)
  message (" MPFR_INCLUDES: ${MPFR_INCLUDES}")
  add_definitions (-DHAVE_MPFR)
endif (hasParent)
endif (0)
