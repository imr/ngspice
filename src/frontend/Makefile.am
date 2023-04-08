## Process this file with automake to produce Makefile.in

SUBDIRS = plotting help parser wdisp numparam trannoise
DIST_SUBDIRS = plotting help parser wdisp numparam trannoise
EXTRA_DIST = testcommands.c parse-bison-y.h

BUILT_SOURCES = parse-bison.h

noinst_LTLIBRARIES = libfte.la

libfte_la_SOURCES = \
    com_measure2.c  \
    com_measure2.h  \
    commands.c  \
    commands.h  \
    com_ahelp.c \
    com_ahelp.h \
    com_alias.c \
    com_alias.h \
    com_asciiplot.c \
    com_asciiplot.h \
    com_cdump.c \
    com_cdump.h \
    com_chdir.c \
    com_chdir.h \
    com_compose.c   \
    com_compose.h   \
    com_commands.h  \
    com_dl.c    \
    com_dl.h    \
    com_display.c   \
    com_display.h   \
    com_dump.c  \
    com_dump.h  \
    com_echo.c  \
    com_echo.h  \
    com_ghelp.c \
    com_ghelp.h \
    com_gnuplot.h   \
    com_gnuplot.c   \
    com_hardcopy.c  \
    com_hardcopy.h  \
    com_help.c  \
    com_help.h  \
    com_history.c   \
    com_history.h   \
    com_let.c   \
    com_let.h   \
    com_option.c    \
    com_option.h    \
    com_plot.c  \
    com_plot.h  \
    com_rehash.c    \
    com_rehash.h    \
    com_set.c   \
    com_set.h   \
    com_setscale.c  \
    com_setscale.h  \
    com_shell.c \
    com_shell.h \
    com_shift.c \
    com_shift.h \
    com_fft.c   \
    com_fft.h   \
    com_state.c \
    com_state.h \
    com_strcmp.c    \
    com_strcmp.h    \
    com_sysinfo.c   \
    com_unset.c \
    com_unset.h \
    com_wr_ic.c  \
    com_wr_ic.h  \
    completion.h    \
    control.h   \
    control.c   \
    ftehelp.h   \
    ftesopt.c   \
    hcomp.c     \
    hcomp.h     \
    init.c      \
    init.h      \
    quote.c     \
    quote.h     \
    streams.h   \
    streams.c   \
    terminal.c  \
    terminal.h  \
    variable.c  \
    variable.h  \
            \
    arg.c       \
    arg.h       \
    aspice.c    \
    aspice.h    \
    breakp.c    \
    breakp.h    \
    breakp2.c   \
    breakp2.h   \
    circuits.c  \
    circuits.h  \
    cpitf.c     \
    cpitf.h     \
    define.c    \
    define.h    \
    device.c    \
    device.h    \
    diff.c      \
    diff.h      \
    dimens.c    \
    dimens.h    \
    display.c   \
    display.h   \
    dotcards.c  \
    dotcards.h  \
    dvec.c      \
    error.c     \
    evaluate.c  \
    evaluate.h  \
    fourier.c   \
    fourier.h   \
    gens.c      \
    gens.h      \
    get_avail_mem_size.c     \
    get_resident_set_size.c  \
    get_phys_mem_size.c      \
    hpgl.c      \
    hpgl.h      \
    inp.c       \
    inp.h       \
    inpcom.c    \
    inpcom.h    \
    inpcompat.c \
    inpcompat.h \
    inpc_probe.c    \
    interp.c    \
    interp.h    \
    inventory.c     \
    linear.c    \
    linear.h    \
    logicexp.c  \
    measure.c   \
    misccoms.c  \
    misccoms.h  \
    miscvars.c  \
    miscvars.h  \
    mw_coms.c   \
    mw_coms.h   \
    newcoms.c   \
    newcoms.h   \
    nutinp.c    \
    nutinp.h    \
    nutmegif.c  \
    nutmegif.h  \
    options.c   \
    options.h   \
    outitf.c    \
    outitf.h    \
    parse.c     \
    parse.h     \
    parse-bison.y   \
    points.c    \
    points.h    \
    postcoms.c  \
    postcoms.h  \
    postsc.c    \
    postsc.h    \
    rawfile.c   \
    rawfile.h   \
    resource.c  \
    resource.h  \
    runcoms.c   \
    runcoms.h   \
    runcoms2.c  \
    runcoms2.h  \
    shyu.c      \
    shyu.h      \
    signal_handler.c \
    signal_handler.h \
    spec.c      \
    spec.h      \
    spiceif.c   \
    spiceif.h   \
    subckt.c    \
    subckt.h    \
    svg.c       \
    svg.h       \
    typesdef.c  \
    typesdef.h  \
    udevices.c  \
    vectors.c   \
    vectors.h   \
    where.c     \
    where.h

# TESTS = testcommands
#
# bin_PROGRAMS = testcommands
#
# testcommands_SOURCES = \
#   testcommands.c \
#   testcommands.h
#
# testcommands_LDADD = libfte.a plotting/libplotting.a ../misc/libmisc.a


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include @X_CFLAGS@
AM_CFLAGS = $(STATIC)
AM_YFLAGS = -d

MAINTAINERCLEANFILES = Makefile.in
