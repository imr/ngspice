## Process this file with automake to produce Makefile.in
##
## Bundled copy of SuiteSparse (KLU, AMD, BTF, COLAMD, SuiteSparse_config),
## used when no installed SuiteSparse is found by configure.  The sources
## are unchanged, see import.sh and README.ngspice.

EXTRA_DIST = \
	import.sh \
	README.ngspice \
	License_AMD.txt \
	License_BTF.txt \
	License_COLAMD.txt \
	License_KLU.txt

noinst_LTLIBRARIES = libSuiteSparse_real.la libSuiteSparse_complex.la libSuiteSparse.la

noinst_HEADERS = \
	SuiteSparse_config.h \
	amd.h \
	amd_internal.h \
	btf.h \
	btf_internal.h \
	colamd.h \
	klu.h \
	klu_internal.h \
	klu_version.h

# numeric part of KLU, compiled for real and (with -DCOMPLEX) complex entries
libSuiteSparse_real_la_SOURCES = \
	klu.c \
	klu_diagnostics.c \
	klu_dump.c \
	klu_extract.c \
	klu_factor.c \
	klu_free_numeric.c \
	klu_kernel.c \
	klu_refactor.c \
	klu_scale.c \
	klu_solve.c \
	klu_sort.c \
	klu_tsolve.c

libSuiteSparse_complex_la_SOURCES = \
	klu.c \
	klu_diagnostics.c \
	klu_dump.c \
	klu_extract.c \
	klu_factor.c \
	klu_free_numeric.c \
	klu_kernel.c \
	klu_refactor.c \
	klu_scale.c \
	klu_solve.c \
	klu_sort.c \
	klu_tsolve.c

# AMD, BTF, COLAMD, SuiteSparse_config and the symbolic part of KLU
libSuiteSparse_la_SOURCES = \
	amd_1.c \
	amd_2.c \
	amd_aat.c \
	amd_control.c \
	amd_defaults.c \
	amd_dump.c \
	amd_info.c \
	amd_order.c \
	amd_post_tree.c \
	amd_postorder.c \
	amd_preprocess.c \
	amd_valid.c \
	amd_version.c \
	btf_maxtrans.c \
	btf_order.c \
	btf_strongcomp.c \
	btf_version.c \
	colamd.c \
	colamd_version.c \
	klu_analyze.c \
	klu_analyze_given.c \
	klu_defaults.c \
	klu_free_symbolic.c \
	klu_memory.c \
	klu_version.c \
	SuiteSparse_config.c

libSuiteSparse_real_la_CPPFLAGS = $(AM_CPPFLAGS)
libSuiteSparse_complex_la_CPPFLAGS = $(AM_CPPFLAGS) -DCOMPLEX

libSuiteSparse_la_LIBADD = \
	libSuiteSparse_real.la \
	libSuiteSparse_complex.la

AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
