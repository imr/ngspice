
# https://gist.github.com/johnb003/65982fdc7a1274fdb023b0c68664ebe4

# 3.10 adds support for "gtest_discover_tests" which enumerates the tests inside
# of the code and adds them to ctest.
#
cmake_minimum_required (VERSION 3.14)
project (cppduals_thirdparty)
include (ExternalProject)

get_directory_property (hasParent PARENT_DIRECTORY)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
  cmake_policy (SET CMP0135 NEW)
endif ()
if (CMAKE_XCODE_BUILD_SYSTEM VERSION_GREATER_EQUAL 12)
  cmake_policy (SET CMP0114 NEW)
endif ()

set (DEPS_ROOT "${CMAKE_BINARY_DIR}/root")
if (hasParent)
  set (DEPS_ROOT "${CMAKE_BINARY_DIR}/thirdparty/root" PARENT_SCOPE)
endif (hasParent)

if (NOT WIN32)
  set (DOWNLOAD_DIR "$ENV{HOME}/Downloads")
else (NOT WIN32)
  set (DOWNLOAD_DIR "C:/Downloads")
endif (NOT WIN32)

include(FetchContent)

#
# Google test
# https://google.github.io/googletest/quickstart-cmake.html

# 
# Configure google-test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  URL_HASH MD5=eb1c5c237d13ed12bf492d3997ca6b0d
  DOWNLOAD_NAME googletest-v1.15.2.zip
  DOWNLOAD_DIR "$ENV{HOME}/Downloads"
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
include(GoogleTest)

#
# Eigen
#
if (TRUE)
  set (EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar.bz2)
  set (EIGEN_MD5 432ef01499d514f4606343276afa0ec3)
  set (EIGEN_MAX_CXX 17)
else ()
  set (EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2)
  set (EIGEN_MD5 b9e98a200d2455f06db9c661c5610496)
  set (EIGEN_MAX_CXX 17)
endif ()
#
# Eigen
#
ExternalProject_Add (eigenX
  PREFIX eigenX
  URL ${EIGEN_URL}
  URL_HASH MD5=${EIGEN_MD5}
  DOWNLOAD_DIR "$ENV{HOME}/Downloads"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  )
ExternalProject_Get_Property (eigenX source_dir)
if (true) # || hasParent
  add_library (eigen INTERFACE IMPORTED GLOBAL)
  add_dependencies (eigen eigenX)
  set_property (TARGET eigen APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${source_dir}")
  if (XCODE)
    set (IOFORMAT "IOFormat\(FullPrecision, DontAlignCols, \", \", \"\\$<SEMICOLON>\\\\n\", \"\", \"\", \"[\", \"]\"\)")
  else ()
    set (IOFORMAT "IOFormat\(FullPrecision, DontAlignCols, \", \", \"\\$<SEMICOLON>\\n\", \"\", \"\", \"[\", \"]\"\)")
  endif ()
  #target_compile_definitions (eigen INTERFACE EIGEN_DEFAULT_IO_FORMAT=${IOFORMAT})
  #target_compile_definitions (eigen INTERFACE EIGEN_DEFAULT_IO_FORMAT=EIGEN_IO_FORMAT_OCTAVE)
  set_property (TARGET eigen APPEND PROPERTY
    INTERFACE_COMPILE_DEFINITIONS EIGEN_DEFAULT_IO_FORMAT=${IOFORMAT})
endif ()

# if c++20, disable warning -Wdeprecated-enum-enum-conversion to eigen
if (CMAKE_CXX_STANDARD GREATER_EQUAL 20)
  target_compile_options (eigen INTERFACE -Wno-deprecated-enum-enum-conversion)
endif ()

#
# expokit
#
#set (EEX_SHA  ee28baa3bf29561501e17e5c68c2e54c85daae19  ) newer? used by spindropsSDL. md5=cebd15f9b5068c0e327753244ff6d394
set (EEX_SHA c157dec0057be6e183a1ea2a5de353fac7e5e3a7 )
set (EEX_MD5 89484e51f706398284235b96bc805515 )
ExternalProject_Add (expokitX
  PREFIX expokitX
  URL https://gitlab.com/tesch1/eigen-expokit/-/archive/${EEX_SHA}/eigen-expokit.tar.bz2
  URL_HASH MD5=${EEX_MD5}
  DOWNLOAD_NAME eigen-expokit-${EEX_SHA}.tar.bz2
  DOWNLOAD_DIR "$ENV{HOME}/Downloads"
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  )
ExternalProject_Get_Property (expokitX source_dir)
if (hasParent)
  add_library (expokit INTERFACE IMPORTED GLOBAL)
  add_dependencies (expokit expokitX)
  set_property (TARGET expokit APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${source_dir}")
endif (hasParent)


#
# fmt
#
ExternalProject_Add (fmtX
  PREFIX fmtX
  URL https://github.com/fmtlib/fmt/archive/11.1.4.tar.gz
  URL_HASH MD5=10c2ae163accd3b82e6b8b4dff877645
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  )
ExternalProject_Get_Property (fmtX source_dir)
ExternalProject_Get_Property (fmtX binary_dir)
if (hasParent)
  add_subdirectory (${source_dir} ${binary_dir} EXCLUDE_FROM_ALL)
endif ()

if (CPPDUALS_BENCHMARK)
  #
  # google benchmark
  #
  ExternalProject_Add (benchmarkX
    PREFIX benchmarkX
    URL "http://github.com/google/benchmark/archive/v1.9.1.tar.gz"
    URL_HASH MD5=92000ef8b4a7b1e9229972f8943070a7
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
  ExternalProject_Get_Property (benchmarkX source_dir)
  ExternalProject_Get_Property (benchmarkX binary_dir)
  if (hasParent)
    # https://github.com/google/benchmark#requirements
    set (BENCHMARK_ENABLE_GTEST_TESTS OFF)
    set (BENCHMARK_USE_LIBCXX ${CPPDUALS_USE_LIBCXX})
    add_subdirectory (${source_dir} ${binary_dir} EXCLUDE_FROM_ALL)
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
