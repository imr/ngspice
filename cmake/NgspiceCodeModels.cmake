# NgspiceCodeModels.cmake
#
# Builds the XSPICE code models (analog.cm, digital.cm, ...).
#
# This replaces src/xspice/icm/GNUmakefile.in, which is a hand written
# GNU make file outside the automake tree.  The pipeline per code model
# directory <cm> is:
#
#   cmpp -lst   (CMPP_IDIR=<src>/<cm>  CMPP_ODIR=<bin>/<cm>)
#       -> cmextrn.h cminfo.h cminfo2.h udnextrn.h udninfo.h udninfo2.h
#          objects.inc                     (from modpath.lst / udnpath.lst)
#
#   for every model m listed in modpath.lst:
#       cmpp -ifs   -> <bin>/<cm>/<m>/ifspec.c   (from ifspec.ifs)
#       cmpp -mod   -> <bin>/<cm>/<m>/cfunc.c    (from cfunc.mod)
#
#   for every user defined node type u in udnpath.lst:
#       <src>/<cm>/<u>/udnfunc.c is compiled as-is
#
#   link dlmain.c + all of the above + the shared helpers into <cm>.cm
#
# The models resolve simulator functions through the coreInfo_t pointer
# table (see src/xspice/icm/dlmain.c), so the modules do not have to be
# linked against ngspice itself -- which is why this also works with MSVC.

set(NGSPICE_ICM_SRC_DIR "${CMAKE_SOURCE_DIR}/src/xspice/icm")
set(NGSPICE_ICM_BIN_DIR "${CMAKE_BINARY_DIR}/codemodels")

# The set of code models, mirroring CMDIRS in src/xspice/icm/GNUmakefile.in
set(NGSPICE_CODE_MODELS spice2poly digital analog xtradev xtraevt table tlines
    CACHE STRING "XSPICE code models to build")


# Read a modpath.lst / udnpath.lst into a CMake list.
function(_ngspice_read_lst path out_var)
    set(_result "")
    if(EXISTS "${path}")
        file(STRINGS "${path}" _lines)
        foreach(_l IN LISTS _lines)
            string(STRIP "${_l}" _l)
            if(_l AND NOT _l MATCHES "^[#*]")
                list(APPEND _result "${_l}")
            endif()
        endforeach()
    endif()
    set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()


# Shared helper objects that every .cm needs.
function(_ngspice_add_cm_common)
    if(TARGET ngspice_cm_common)
        return()
    endif()
    add_library(ngspice_cm_common OBJECT
        "${CMAKE_SOURCE_DIR}/src/misc/dstring.c"
        "${CMAKE_SOURCE_DIR}/src/xspice/tlines/tline_common.c"
        "${CMAKE_SOURCE_DIR}/src/xspice/tlines/msline_common.c")
    target_include_directories(ngspice_cm_common PRIVATE
        "${CMAKE_SOURCE_DIR}/src/xspice/tlines")
    target_link_libraries(ngspice_cm_common PRIVATE ngspice_settings)
    set_target_properties(ngspice_cm_common PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
endfunction()


function(ngspice_add_code_model cm)
    set(_src "${NGSPICE_ICM_SRC_DIR}/${cm}")
    set(_bin "${NGSPICE_ICM_BIN_DIR}/${cm}")

    if(NOT EXISTS "${_src}/modpath.lst")
        message(WARNING "code model '${cm}' has no modpath.lst -- skipped")
        return()
    endif()

    _ngspice_read_lst("${_src}/modpath.lst" _models)
    _ngspice_read_lst("${_src}/udnpath.lst" _udns)

    file(MAKE_DIRECTORY "${_bin}")

    # -- step 1: the per-directory descriptor headers ------------------------
    set(_descr
        "${_bin}/cmextrn.h"  "${_bin}/cminfo.h"  "${_bin}/cminfo2.h"
        "${_bin}/udnextrn.h" "${_bin}/udninfo.h" "${_bin}/udninfo2.h"
        "${_bin}/objects.inc")

    add_custom_command(
        OUTPUT ${_descr}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_bin}"
        COMMAND ${CMAKE_COMMAND} -E env
                "CMPP_IDIR=${_src}" "CMPP_ODIR=${_bin}"
                ${NGSPICE_CMPP_EXECUTABLE} -lst
        DEPENDS "${_src}/modpath.lst" "${_src}/udnpath.lst"
                ${NGSPICE_CMPP_TARGET}
        COMMENT "cmpp -lst ${cm}"
        VERBATIM)

    set(_generated "")

    # -- step 2: ifspec.c / cfunc.c per model --------------------------------
    foreach(_m IN LISTS _models)
        file(MAKE_DIRECTORY "${_bin}/${_m}")

        add_custom_command(
            OUTPUT "${_bin}/${_m}/ifspec.c"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_bin}/${_m}"
            COMMAND ${CMAKE_COMMAND} -E env
                    "CMPP_IDIR=${_src}/${_m}" "CMPP_ODIR=${_bin}/${_m}"
                    ${NGSPICE_CMPP_EXECUTABLE} -ifs
            DEPENDS "${_src}/${_m}/ifspec.ifs" ${NGSPICE_CMPP_TARGET}
            COMMENT "cmpp -ifs ${cm}/${_m}"
            VERBATIM)

        add_custom_command(
            OUTPUT "${_bin}/${_m}/cfunc.c"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_bin}/${_m}"
            COMMAND ${CMAKE_COMMAND} -E env
                    "CMPP_IDIR=${_src}/${_m}" "CMPP_ODIR=${_bin}/${_m}"
                    ${NGSPICE_CMPP_EXECUTABLE} -mod
            DEPENDS "${_src}/${_m}/cfunc.mod" ${NGSPICE_CMPP_TARGET}
            COMMENT "cmpp -mod ${cm}/${_m}"
            VERBATIM)

        list(APPEND _generated
             "${_bin}/${_m}/ifspec.c" "${_bin}/${_m}/cfunc.c")
    endforeach()

    # -- step 3: user defined node types -------------------------------------
    set(_udn_sources "")
    foreach(_u IN LISTS _udns)
        list(APPEND _udn_sources "${_src}/${_u}/udnfunc.c")
    endforeach()

    # -- step 4: the module --------------------------------------------------
    _ngspice_add_cm_common()

    add_library(cm_${cm} MODULE
        "${NGSPICE_ICM_SRC_DIR}/dlmain.c"
        ${_generated}
        ${_udn_sources}
        $<TARGET_OBJECTS:ngspice_cm_common>)

    # dlmain.c includes "cmextrn.h" etc. from the *build* directory.
    # dlmain.c is shared by every code model, so the dependency cannot be
    # attached to the source file (that property is per directory, not per
    # target) -- use an intermediate target instead.
    add_custom_target(cm_${cm}_descr DEPENDS ${_descr})
    add_dependencies(cm_${cm} cm_${cm}_descr)

    target_include_directories(cm_${cm} PRIVATE
        "${_bin}"
        "${_src}"
        "${CMAKE_SOURCE_DIR}/src/xspice/tlines")

    # every model's cfunc.c may include headers next to its own sources
    foreach(_m IN LISTS _models)
        target_include_directories(cm_${cm} PRIVATE
            "${_src}/${_m}" "${_bin}/${_m}")
    endforeach()

    target_link_libraries(cm_${cm} PRIVATE ngspice_settings)

    set_target_properties(cm_${cm} PROPERTIES
        OUTPUT_NAME "${cm}"
        PREFIX ""
        SUFFIX ".cm"
        LIBRARY_OUTPUT_DIRECTORY "${NGSPICE_ICM_BIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${NGSPICE_ICM_BIN_DIR}"
        POSITION_INDEPENDENT_CODE ON)

    # multi-config generators (Visual Studio, Xcode) append the config name
    foreach(_c IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(cm_${cm} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${_c} "${NGSPICE_ICM_BIN_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${_c} "${NGSPICE_ICM_BIN_DIR}")
    endforeach()

    if(APPLE)
        # code models are dlopen()ed; unresolved references are looked up in
        # the host binary
        target_link_options(cm_${cm} PRIVATE
            -Wl,-undefined,dynamic_lookup)
    endif()

    if(UNIX)
        target_link_libraries(cm_${cm} PRIVATE m)
    endif()

    install(TARGETS cm_${cm}
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/ngspice"
            RUNTIME DESTINATION "${CMAKE_INSTALL_LIBDIR}/ngspice")
endfunction()


function(ngspice_add_all_code_models)
    foreach(_cm IN LISTS NGSPICE_CODE_MODELS)
        ngspice_add_code_model("${_cm}")
    endforeach()

    add_custom_target(codemodels ALL)
    foreach(_cm IN LISTS NGSPICE_CODE_MODELS)
        if(TARGET cm_${_cm})
            add_dependencies(codemodels cm_${_cm})
        endif()
    endforeach()
endfunction()
