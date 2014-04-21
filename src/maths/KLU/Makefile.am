## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libKLU_real.la libKLU_complex.la libKLU.la

libKLU_real_la_SOURCES = 	\
	amd_1.c			\
	amd_2.c			\
	amd_aat.c		\
	amd_control.c		\
	amd_defaults.c		\
	amd_dump.c		\
	amd_global.c		\
	amd_info.c		\
	amd_order.c		\
	amd_postorder.c		\
	amd_post_tree.c		\
	amd_preprocess.c	\
	amd_valid.c		\
	btf_maxtrans.c		\
	btf_order.c		\
	btf_strongcomp.c	\
	colamd.c		\
	colamd_global.c		\
	klu.c			\
	klu_analyze.c		\
	klu_analyze_given.c	\
	klu_defaults.c		\
	klu_diagnostics.c	\
	klu_dump.c		\
	klu_extract.c		\
	klu_factor.c		\
	klu_free_numeric.c 	\
	klu_free_symbolic.c	\
	klu_kernel.c		\
	klu_memory.c		\
	klu_refactor.c		\
	klu_scale.c		\
	klu_solve.c		\
	klu_sort.c		\
	klu_tsolve.c


libKLU_real_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include


libKLU_complex_la_SOURCES = 	\
	amd_1.c			\
	amd_2.c			\
	amd_aat.c		\
	amd_control.c		\
	amd_defaults.c		\
	amd_dump.c		\
	amd_global.c		\
	amd_info.c		\
	amd_order.c		\
	amd_postorder.c		\
	amd_post_tree.c		\
	amd_preprocess.c	\
	amd_valid.c		\
	btf_maxtrans.c		\
	btf_order.c		\
	btf_strongcomp.c	\
	colamd.c		\
	colamd_global.c		\
	klu.c			\
	klu_analyze.c		\
	klu_analyze_given.c	\
	klu_defaults.c		\
	klu_diagnostics.c	\
	klu_dump.c		\
	klu_extract.c		\
	klu_factor.c		\
	klu_free_numeric.c 	\
	klu_free_symbolic.c	\
	klu_kernel.c		\
	klu_memory.c		\
	klu_refactor.c		\
	klu_scale.c		\
	klu_solve.c		\
	klu_sort.c		\
	klu_tsolve.c


libKLU_complex_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -DCOMPLEX


libKLU_la_SOURCES =		\
	klusmp.c

libKLU_la_LIBADD =		\
	libKLU_real.la		\
	libKLU_complex.la


libKLU_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include

MAINTAINERCLEANFILES = Makefile.in
