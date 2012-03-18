## Process this file with automake to produce Makefile.in

#TRY TO FIND A WAY NOT TO OPTIMIZE superlu_timer AND dlamch COMPILATION

noinst_LTLIBRARIES = libSuperLU.la libSuperLU_real.la libSuperLU_complex.la

libSuperLU_real_la_SOURCES =	\
	colamd.c		\
	dcolumn_bmod.c		\
	dcolumn_dfs.c		\
	dcopy_to_ucol.c		\
	ddiagonal.c		\
	dgscon.c		\
	dgsequ.c		\
	dgsisx.c		\
	dgsitrf.c		\
	dgsrfs.c		\
	dgssv.c			\
	dgssvx.c		\
	dgstrf.c		\
	dgstrs.c		\
	dlacon.c		\
	dlamch.c		\
	dlangs.c		\
	dlaqgs.c		\
	dldperm.c		\
	dmemory.c		\
	dmyblas2.c		\
	dpanel_bmod.c		\
	dpanel_dfs.c		\
	dpivotgrowth.c		\
	dpivotL.c		\
	dpruneL.c		\
	dreadhb.c		\
	dreadrb.c		\
	dreadtriple.c		\
	dsnode_bmod.c		\
	dsnode_dfs.c		\
	dsp_blas2.c		\
	dsp_blas3.c		\
	dutil.c			\
	get_perm_c.c		\
	heap_relax_snode.c	\
	ilu_dcolumn_dfs.c	\
	ilu_dcopy_to_ucol.c	\
	ilu_ddrop_row.c		\
	ilu_dpanel_dfs.c	\
	ilu_dpivotL.c		\
	ilu_dsnode_dfs.c	\
	ilu_heap_relax_snode.c	\
	ilu_relax_snode.c	\
	lsame.c			\
	mark_relax.c		\
	mc64ad.c		\
	memory.c		\
	mmd.c			\
	qselect.c		\
	relax_snode.c		\
	sp_coletree.c		\
	sp_ienv.c		\
	sp_preorder.c		\
	superlu_timer.c		\
	util.c			\
	xerbla.c

libSuperLU_real_la_CPPFLAGS = @AM_CPPFLAGS@ -DAdd_ -DUSE_VENDOR_BLAS -I$(top_srcdir)/src/include

libSuperLU_real_la_LDFLAGS = -lf77blas -latlas


libSuperLU_complex_la_SOURCES =	\
	colamd.c		\
	dcomplex.c		\
	dlamch.c		\
	dzsum1.c		\
	get_perm_c.c		\
	heap_relax_snode.c	\
	ilu_heap_relax_snode.c	\
	ilu_relax_snode.c	\
	ilu_zcolumn_dfs.c	\
	ilu_zcopy_to_ucol.c	\
	ilu_zdrop_row.c		\
	ilu_zpanel_dfs.c	\
	ilu_zpivotL.c 		\
	ilu_zsnode_dfs.c	\
	izmax1.c		\
	lsame.c			\
	mark_relax.c		\
	mc64ad.c		\
	memory.c		\
	mmd.c			\
	qselect.c		\
	relax_snode.c		\
	sp_coletree.c		\
	sp_ienv.c		\
	sp_preorder.c		\
	superlu_timer.c		\
	util.c			\
	xerbla.c		\
	zgssv.c			\
	zgssvx.c		\
	zsp_blas2.c		\
	zsp_blas3.c		\
	zgscon.c		\
	zlangs.c		\
	zgsequ.c		\
	zlaqgs.c		\
	zpivotgrowth.c		\
	zgsrfs.c 		\
	zgstrf.c		\
	zgstrs.c		\
	zcopy_to_ucol.c		\
	zsnode_dfs.c		\
	zsnode_bmod.c		\
	zpanel_dfs.c		\
	zpanel_bmod.c		\
	zreadhb.c		\
	zreadrb.c		\
	zreadtriple.c		\
	zcolumn_dfs.c		\
	zcolumn_bmod.c		\
	zpivotL.c		\
	zpruneL.c		\
	zmemory.c		\
	zutil.c			\
	zmyblas2.c		\
	zgsisx.c		\
	zgsitrf.c		\
	zldperm.c		\
	zdiagonal.c		\
	zlacon.c

libSuperLU_complex_la_CPPFLAGS = @AM_CPPFLAGS@ -DAdd_ -DUSE_VENDOR_BLAS -I$(top_srcdir)/src/include

libSuperLU_complex_la_LDFLAGS = -lf77blas -latlas


libSuperLU_la_SOURCES =		\
	superlusmp.c

libSuperLU_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include

libSuperLU_la_LIBADD =		\
	libSuperLU_real.la	\
	libSuperLU_complex.la


MAINTAINERCLEANFILES = Makefile.in
