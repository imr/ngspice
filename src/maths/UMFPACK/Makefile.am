## Process this file with automake to produce Makefile.in

#THIS LIBRARY IS VERY COMPLICATED TO BUILD

noinst_LTLIBRARIES =				\
	libUMFPACK_AMD.la			\
	libUMFPACK_global.la			\
	libUMFPACK_CONJ_DINT.la			\
	libUMFPACK_DOMAP_DINT.la		\
	libUMFPACK_DOVALUES_DINT.la		\
	libUMFPACK_DOMAP_DOVALUES_DINT.la	\
	libUMFPACK_DFIXQ_DINT.la		\
	libUMFPACK_DDROP_DINT.la		\
	libUMFPACK_DWSOLVE_DINT.la		\
	libUMFPACK_DINT.la			\
	libUMFPACK_ZINT.la			\
	libUMFPACK_CONJ_ZINT.la			\
	libUMFPACK_DOMAP_ZINT.la		\
	libUMFPACK_DOVALUES_ZINT.la		\
	libUMFPACK_DOMAP_DOVALUES_ZINT.la	\
	libUMFPACK_DFIXQ_ZINT.la		\
	libUMFPACK_DDROP_ZINT.la		\
	libUMFPACK_DWSOLVE_ZINT.la		\
	libUMFPACK.la




libUMFPACK_AMD_la_SOURCES =			\
	amd_1.c					\
	amd_2.c					\
	amd_aat.c				\
	amd_control.c				\
	amd_defaults.c				\
	amd_dump.c				\
	amd_global.c				\
	amd_info.c				\
	amd_order.c				\
	amd_postorder.c				\
	amd_post_tree.c				\
	amd_preprocess.c			\
	amd_valid.c

libUMFPACK_AMD_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include


libUMFPACK_global_la_SOURCES =			\
	umfpack_global.c			\
	umfpack_tictoc.c			\
	umfpack_timer.c				\
	umf_triplet.c

libUMFPACK_global_la_CPPFLAGS = @AM_CPPFLAGS@ -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_global_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_CONJ_DINT_la_SOURCES =		\
	umf_ltsolve.c				\
	umf_utsolve.c

libUMFPACK_CONJ_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DCONJUGATE_SOLVE -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_CONJ_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DOMAP_DINT_la_SOURCES =		\
	umf_triplet.c

libUMFPACK_DOMAP_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DDO_MAP -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DOMAP_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DOVALUES_DINT_la_SOURCES =		\
	umf_triplet.c

libUMFPACK_DOVALUES_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DDO_VALUES -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DOVALUES_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DOMAP_DOVALUES_DINT_la_SOURCES =	\
	umf_triplet.c

libUMFPACK_DOMAP_DOVALUES_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DDO_MAP -DDO_VALUES -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DOMAP_DOVALUES_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DFIXQ_DINT_la_SOURCES =		\
	umf_assemble.c

libUMFPACK_DFIXQ_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DFIXQ -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DFIXQ_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DDROP_DINT_la_SOURCES =		\
	umf_store_lu.c

libUMFPACK_DDROP_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DDROP -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DDROP_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DWSOLVE_DINT_la_SOURCES =		\
	umfpack_solve.c

libUMFPACK_DWSOLVE_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DWSOLVE -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DWSOLVE_DINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DINT_la_SOURCES = 			\
	umf_analyze.c				\
	umf_apply_order.c			\
	umf_assemble.c				\
	umf_blas3_update.c			\
	umf_build_tuples.c			\
	umf_cholmod.c				\
	umf_colamd.c				\
	umf_create_element.c			\
	umf_dump.c				\
	umf_extend_front.c			\
	umf_free.c				\
	umf_fsize.c				\
	umf_garbage_collection.c		\
	umf_get_memory.c			\
	umf_grow_front.c			\
	umf_init_front.c			\
	umf_is_permutation.c			\
	umf_kernel.c				\
	umf_kernel_init.c			\
	umf_kernel_wrapup.c			\
	umf_local_search.c			\
	umf_lsolve.c				\
	umf_ltsolve.c				\
	umf_malloc.c				\
	umf_mem_alloc_element.c			\
	umf_mem_alloc_head_block.c		\
	umf_mem_alloc_tail_block.c		\
	umf_mem_free_tail_block.c		\
	umf_mem_init_memoryspace.c		\
	umfpack_col_to_triplet.c		\
	umfpack_defaults.c			\
	umfpack_free_numeric.c			\
	umfpack_free_symbolic.c			\
	umfpack_get_determinant.c		\
	umfpack_get_lunz.c			\
	umfpack_get_numeric.c			\
	umfpack_get_symbolic.c			\
	umfpack_load_numeric.c			\
	umfpack_load_symbolic.c			\
	umfpack_numeric.c			\
	umfpack_qsymbolic.c			\
	umfpack_report_control.c		\
	umfpack_report_info.c			\
	umfpack_report_matrix.c			\
	umfpack_report_numeric.c		\
	umfpack_report_perm.c			\
	umfpack_report_status.c			\
	umfpack_report_symbolic.c		\
	umfpack_report_triplet.c		\
	umfpack_report_vector.c			\
	umfpack_save_numeric.c			\
	umfpack_save_symbolic.c			\
	umfpack_scale.c				\
	umfpack_solve.c				\
	umfpack_symbolic.c			\
	umfpack_transpose.c			\
	umfpack_triplet_to_col.c		\
	umf_realloc.c				\
	umf_report_perm.c			\
	umf_report_vector.c			\
	umf_row_search.c			\
	umf_scale.c				\
	umf_scale_column.c			\
	umf_set_stats.c				\
	umf_singletons.c			\
	umf_solve.c				\
	umf_start_front.c			\
	umf_store_lu.c				\
	umf_symbolic_usage.c			\
	umf_transpose.c				\
	umf_tuple_lengths.c			\
	umf_usolve.c				\
	umf_utsolve.c				\
	umf_valid_numeric.c			\
	umf_valid_symbolic.c

libUMFPACK_DINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DDINT -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DINT_la_LDFLAGS = -lf77blas -latlas




libUMFPACK_ZINT_la_SOURCES =	 		\
	umf_assemble.c				\
	umf_blas3_update.c			\
	umf_build_tuples.c			\
	umf_create_element.c			\
	umf_dump.c				\
	umf_extend_front.c			\
	umf_garbage_collection.c		\
	umf_get_memory.c			\
	umf_init_front.c			\
	umf_is_permutation.c			\
	umf_kernel.c				\
	umf_kernel_init.c			\
	umf_kernel_wrapup.c			\
	umf_local_search.c			\
	umf_lsolve.c				\
	umf_ltsolve.c				\
	umf_mem_alloc_element.c			\
	umf_mem_alloc_head_block.c		\
	umf_mem_alloc_tail_block.c		\
	umf_mem_free_tail_block.c		\
	umf_mem_init_memoryspace.c		\
	umfpack_col_to_triplet.c		\
	umfpack_defaults.c			\
	umfpack_free_numeric.c			\
	umfpack_free_symbolic.c			\
	umfpack_get_determinant.c		\
	umfpack_get_lunz.c			\
	umfpack_get_numeric.c			\
	umfpack_get_symbolic.c			\
	umfpack_load_numeric.c			\
	umfpack_load_symbolic.c			\
	umfpack_numeric.c			\
	umfpack_qsymbolic.c			\
	umfpack_report_control.c		\
	umfpack_report_info.c			\
	umfpack_report_matrix.c			\
	umfpack_report_numeric.c		\
	umfpack_report_perm.c			\
	umfpack_report_status.c			\
	umfpack_report_symbolic.c		\
	umfpack_report_triplet.c		\
	umfpack_report_vector.c			\
	umfpack_save_numeric.c			\
	umfpack_save_symbolic.c			\
	umfpack_scale.c				\
	umfpack_solve.c				\
	umfpack_symbolic.c			\
	umfpack_transpose.c			\
	umfpack_triplet_to_col.c		\
	umf_report_vector.c			\
	umf_row_search.c			\
	umf_scale.c				\
	umf_scale_column.c			\
	umf_set_stats.c				\
	umf_solve.c				\
	umf_start_front.c			\
	umf_store_lu.c				\
	umf_symbolic_usage.c			\
	umf_transpose.c				\
	umf_tuple_lengths.c			\
	umf_usolve.c				\
	umf_utsolve.c				\
	umf_valid_numeric.c			\
	umf_valid_symbolic.c

libUMFPACK_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_CONJ_ZINT_la_SOURCES =		\
	umf_ltsolve.c				\
	umf_utsolve.c

libUMFPACK_CONJ_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DCONJUGATE_SOLVE -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_CONJ_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DOMAP_ZINT_la_SOURCES =		\
	umf_triplet.c

libUMFPACK_DOMAP_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DDO_MAP -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DOMAP_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DOVALUES_ZINT_la_SOURCES =		\
	umf_triplet.c

libUMFPACK_DOVALUES_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DDO_VALUES -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DOVALUES_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DOMAP_DOVALUES_ZINT_la_SOURCES =	\
	umf_triplet.c

libUMFPACK_DOMAP_DOVALUES_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DDO_MAP -DDO_VALUES -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DOMAP_DOVALUES_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DFIXQ_ZINT_la_SOURCES =		\
	umf_assemble.c

libUMFPACK_DFIXQ_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DFIXQ -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DFIXQ_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DDROP_ZINT_la_SOURCES =		\
	umf_store_lu.c

libUMFPACK_DDROP_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DDROP -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DDROP_ZINT_la_LDFLAGS = -lf77blas -latlas


libUMFPACK_DWSOLVE_ZINT_la_SOURCES =		\
	umfpack_solve.c

libUMFPACK_DWSOLVE_ZINT_la_CPPFLAGS = @AM_CPPFLAGS@ -DZINT -DWSOLVE -DNCHOLMOD -I$(top_srcdir)/src/include

libUMFPACK_DWSOLVE_ZINT_la_LDFLAGS = -lf77blas -latlas




libUMFPACK_la_SOURCES =				\
	umfpacksmp.c

libUMFPACK_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include

libUMFPACK_la_LIBADD =				\
	libUMFPACK_AMD.la			\
	libUMFPACK_global.la			\
	libUMFPACK_CONJ_DINT.la			\
	libUMFPACK_DOMAP_DINT.la		\
	libUMFPACK_DOVALUES_DINT.la		\
	libUMFPACK_DOMAP_DOVALUES_DINT.la	\
	libUMFPACK_DFIXQ_DINT.la		\
	libUMFPACK_DDROP_DINT.la		\
	libUMFPACK_DWSOLVE_DINT.la		\
	libUMFPACK_DINT.la			\
	libUMFPACK_ZINT.la			\
	libUMFPACK_CONJ_ZINT.la			\
	libUMFPACK_DOMAP_ZINT.la		\
	libUMFPACK_DOVALUES_ZINT.la		\
	libUMFPACK_DOMAP_DOVALUES_ZINT.la	\
	libUMFPACK_DFIXQ_ZINT.la		\
	libUMFPACK_DDROP_ZINT.la		\
	libUMFPACK_DWSOLVE_ZINT.la

MAINTAINERCLEANFILES = Makefile.in
