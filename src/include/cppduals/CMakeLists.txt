#
# CMake for cppduals
#
cmake_minimum_required (VERSION 3.1)
project (cppduals
  VERSION 0.3.1
  LANGUAGES C CXX
  )
include (GNUInstallDirs)

set (CMAKE_CXX_STANDARD 11 CACHE STRING "Which C++ standard to test against.")
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_DISABLE_IN_SOURCE_BUILD ON)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message (STATUS "Configuring cppduals for standalone build")
    set (CPPDUALS_STANDALONE TRUE)
elseif ()
    message (STATUS "Configuring cppduals for subproject build")
    set (CPPDUALS_STANDALONE FALSE)
endif ()
if (CPPDUALS_STANDALONE AND
    NOT CMAKE_CONFIGURATION_TYPES AND
    NOT CMAKE_NO_BUILD_TYPE AND
    NOT CMAKE_BUILD_TYPE
    )
  message (STATUS "Setting build type to 'RelWithDebInfo' since none specified")
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY VALUE RelWithDebInfo)
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY
    STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
    )
endif ()
if (CPPDUALS_STANDALONE AND
    CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
       CACHE PATH "cppduals installation directory"
       FORCE
       )
  message (STATUS "No install prefix specified; using '${CMAKE_INSTALL_PREFIX}'")
endif ()

set_property (CACHE CMAKE_CXX_STANDARD PROPERTY STRINGS 11 14 17 20)

option (CPPDUALS_TESTING "Enable testing" OFF)
option (CPPDUALS_BENCHMARK "Enable benchmarking" OFF)
option (CPPDUALS_EIGEN_LATEST "Eigen latest" OFF)
option (CPPDUALS_USE_LIBCXX "When testing use flags for libc++" OFF)
set (EIGEN3_INCLUDE_DIRS "" CACHE PATH "Path to Eigen includes" )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wpedantic")
endif ()

if (CPPDUALS_BENCHMARK)
  set (CPPDUALS_TESTING ON)
endif (CPPDUALS_BENCHMARK)

if (CPPDUALS_USE_LIBCXX)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options ("-stdlib=libc++")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ ") # -lc++abi
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # this is unfinished...
    set (CPPDUALS_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++")
    message (WARNING "libc++ header path must be manually specified using CMAKE_CXX_FLAGS")
    list (APPEND CPPDUALS_CXX_LINKER_FLAGS -nodefaultlibs)
    list (APPEND CPPDUALS_CXX_LIBRARIES c++)
  else ()
    message(FATAL_ERROR "-DCPPDUALS_USE_LIBCXX:BOOL=ON is not supported for this compiler")
  endif ()
endif (CPPDUALS_USE_LIBCXX)

#
# Primary Library Target
#
add_library (cppduals_duals INTERFACE)
target_include_directories (cppduals_duals
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
set_target_properties (cppduals_duals
  PROPERTIES
    EXPORT_NAME duals
    )
add_library (cppduals::duals ALIAS cppduals_duals)

#
# Build external dependencies for testing & benchmarking
#
if (CPPDUALS_TESTING)

  cmake_minimum_required (VERSION 3.10) # need gtest_discover_tests
  file (MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/thirdparty")

  # generator name
  if (NOT "${CMAKE_EXTRA_GENERATOR}" STREQUAL "")
    set (GENERATOR_STRING "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
  else ()
    set (GENERATOR_STRING "${CMAKE_GENERATOR}")
  endif ()

  # configure the thirdparty build dir
  execute_process (
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/thirdparty"
    COMMAND ${CMAKE_COMMAND} "-G${GENERATOR_STRING}"
                             "-DCMAKE_CONFIGURATION_TYPES=${CMAKE_CONFIGURATION_TYPES}"
                             "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                             "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
                             "-DCPPDUALS_BENCHMARK=${CPPDUALS_BENCHMARK}"
                             "-DCPPDUALS_EIGEN_LATEST=${CPPDUALS_EIGEN_LATEST}"
                             "-DCPPDUALS_USE_LIBCXX=${CPPDUALS_USE_LIBCXX}"
                             "${PROJECT_SOURCE_DIR}/thirdparty"
    RESULT_VARIABLE DEPS_CONFIG_RESULT
    )
  if (DEPS_CONFIG_RESULT)
    message (FATAL_ERROR "Configuring dependencies failed: ${DEPS_CONFIG_RESULT}")
  endif ()

  # build the thirdparty
  message ("***************************************************************")
  message ("** Building '${PROJECT_BINARY_DIR}/thirdparty'...")
  message ("***************************************************************")
  execute_process (
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/thirdparty"
    COMMAND "${CMAKE_COMMAND}" --build .
    RESULT_VARIABLE DEPS_BUILD_RESULT
    )
  message ("***************************************************************")
  message ("*** Building thirdparty/ done.")
  message ("***************************************************************")
  if (DEPS_BUILD_RESULT)
    message ("***************************************************************")
    message (FATAL_ERROR "Building dependencies failed: ${DEPS_BUILD_RESULT}")
    message ("***************************************************************")
  endif ()

  add_subdirectory (thirdparty)

endif ()

#
# Code Coverage Configuration
#
add_library (cppduals_coverage_config INTERFACE)
option (CODE_COVERAGE "Enable coverage reporting" OFF)
if (CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options (cppduals_coverage_config INTERFACE
    -O0        # no optimization
    -g         # generate debug info
    --coverage # sets all required flags
    )
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    target_link_options (cppduals_coverage_config INTERFACE --coverage)
  else ()
    target_link_libraries (cppduals_coverage_config INTERFACE --coverage)
  endif ()
endif ()

#
# Testing
#
if (CPPDUALS_TESTING)
  enable_testing ()
  include (CTest)
  add_subdirectory (tests)
endif (CPPDUALS_TESTING)

#
# Documentation
#
find_program (DOXYGEN doxygen)
if (DOXYGEN)
  configure_file (${PROJECT_SOURCE_DIR}/doc/Doxyfile.in       ${PROJECT_BINARY_DIR}/doc/Doxyfile)
  configure_file (${PROJECT_SOURCE_DIR}/doc/DoxygenLayout.xml ${PROJECT_BINARY_DIR}/doc/DoxygenLayout.xml COPYONLY)
  configure_file (${PROJECT_SOURCE_DIR}/doc/header.html       ${PROJECT_BINARY_DIR}/doc/header.html COPYONLY)
  configure_file (${PROJECT_SOURCE_DIR}/doc/footer.html       ${PROJECT_BINARY_DIR}/doc/footer.html COPYONLY)
  configure_file (${PROJECT_SOURCE_DIR}/doc/favicon.ico       ${PROJECT_BINARY_DIR}/doc/favicon.ico COPYONLY)
  configure_file (${PROJECT_SOURCE_DIR}/doc/customdoxygen.css ${PROJECT_BINARY_DIR}/doc/customdoxygen.css)
  add_custom_target (cppduals_docs
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc
    COMMAND cmake -E remove_directory ${PROJECT_BINARY_DIR}/docs
    COMMAND ${DOXYGEN} Doxyfile
    )
else ()
  add_custom_target (cppduals_docs
    COMMAND echo "Please install doxygen and reconfigure to build the docs"
    )
endif ()

#
# Installation
#
install (TARGETS cppduals_duals EXPORT cppduals_export)
install (EXPORT cppduals_export
  FILE cppduals-config.cmake
  NAMESPACE cppduals::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cppduals
  )
install (
  DIRECTORY ${PROJECT_SOURCE_DIR}/duals
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN "*~" EXCLUDE
  )
install (
  DIRECTORY ${PROJECT_BINARY_DIR}/docs/   # Trailing slash triggers rename
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/cppduals
  OPTIONAL
  )

#
# etags
#
find_program (ETAGS etags)
if (ETAGS)
  add_custom_target (cppduals_etags
    COMMAND ${ETAGS} --language=c++ ${PROJECT_SOURCE_DIR}/duals/*
    COMMAND ${ETAGS} --language=c++ --append ${PROJECT_SOURCE_DIR}/duals/arch/*/*
    COMMAND ${ETAGS} --language=c++ --append `find ${PROJECT_BINARY_DIR}/thirdparty/eigenX/src/eigenX -type f`
    )
endif (ETAGS)
