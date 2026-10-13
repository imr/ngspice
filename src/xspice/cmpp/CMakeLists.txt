# cmpp -- the XSPICE code model preprocessor.
#
# cmpp is a build-time tool: it turns ifspec.ifs / cfunc.mod into C sources.
# When cross compiling it therefore has to be built for the *host*; see
# NGSPICE_CMPP_EXECUTABLE below.

set(_cmpp_gen "${CMAKE_BINARY_DIR}/generated/src/xspice/cmpp")
file(MAKE_DIRECTORY "${_cmpp_gen}")

bison_target(CmppIfsParser  ifs_yacc.y "${_cmpp_gen}/ifs_yacc.c"
             DEFINES_FILE "${_cmpp_gen}/ifs_yacc.h")
bison_target(CmppModParser  mod_yacc.y "${_cmpp_gen}/mod_yacc.c"
             DEFINES_FILE "${_cmpp_gen}/mod_yacc.h")
flex_target(CmppIfsLexer    ifs_lex.l  "${_cmpp_gen}/ifs_lex.c")
flex_target(CmppModLexer    mod_lex.l  "${_cmpp_gen}/mod_lex.c")

add_flex_bison_dependency(CmppIfsLexer CmppIfsParser)
add_flex_bison_dependency(CmppModLexer CmppModParser)

add_executable(cmpp
    main.c
    file_buffer.c
    pp_ifs.c
    pp_lst.c
    pp_mod.c
    read_ifs.c
    writ_ifs.c
    util.c
    ${BISON_CmppIfsParser_OUTPUTS}
    ${BISON_CmppModParser_OUTPUTS}
    ${FLEX_CmppIfsLexer_OUTPUTS}
    ${FLEX_CmppModLexer_OUTPUTS})

target_include_directories(cmpp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${_cmpp_gen}"
    "${NGSPICE_GENERATED_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/include")

if(MSVC)
    target_compile_definitions(cmpp PRIVATE
        _CRT_SECURE_NO_WARNINGS YY_NO_UNISTD_H)
    target_compile_options(cmpp PRIVATE /wd4996)
endif()
if(WIN32)
    target_link_libraries(cmpp PRIVATE shlwapi)
endif()

# NGSPICE_CMPP_EXECUTABLE lets a cross build point at a natively built cmpp:
#   -DNGSPICE_CMPP_EXECUTABLE=/path/to/host/cmpp
if(NOT NGSPICE_CMPP_EXECUTABLE)
    set(NGSPICE_CMPP_EXECUTABLE "$<TARGET_FILE:cmpp>" PARENT_SCOPE)
    set(NGSPICE_CMPP_TARGET cmpp PARENT_SCOPE)
else()
    set(NGSPICE_CMPP_TARGET "" PARENT_SCOPE)
endif()
