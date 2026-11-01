#!/bin/sh
#
# Copy the parts of SuiteSparse needed by ngspice (KLU, AMD, BTF, COLAMD,
# SuiteSparse_config) into this directory, unchanged.
#
#   usage:  ./import.sh /path/to/SuiteSparse
#
# /path/to/SuiteSparse is a checkout or an unpacked release of
#   https://github.com/DrTimothyAldenDavis/SuiteSparse
#
# Nothing in this directory is modified for ngspice.  All ngspice specific
# code lives one level up in src/maths/KLU (klusmp.c, klu_utils.c,
# klu_multiply.c, klu_compat.c, klu_ngspice.h) and in src/include/ngspice
# (klu.h, klu_ext.h).
#
# Only the 32 bit integer, real and complex routines are used.  The complex
# variants are built by compiling the real sources a second time with
# -DCOMPLEX (see ../Makefile.am), so the klu_z_*.c, klu_l_*.c, klu_zl_*.c,
# amd_l*.c, btf_l_*.c and colamd_l.c wrapper files are not imported.
#
# After an update: adjust the version in README.ngspice, then rebuild and
# run the KLU tests (see README.ngspice).

set -e

if [ $# -ne 1 ] || [ ! -f "$1/KLU/Include/klu.h" ]; then
    echo "usage: $0 /path/to/SuiteSparse" >&2
    exit 1
fi

src=$1
dst=$(cd "$(dirname "$0")" && pwd)

cp_files () {
    dir=$1; shift
    for f in "$@"; do
        cp "$src/$dir/$f" "$dst/$f"
    done
}

cp_files KLU/Include      klu.h klu_internal.h klu_version.h
cp_files KLU/Source       klu.c klu_analyze.c klu_analyze_given.c klu_defaults.c \
                          klu_diagnostics.c klu_dump.c klu_extract.c klu_factor.c \
                          klu_free_numeric.c klu_free_symbolic.c klu_kernel.c \
                          klu_memory.c klu_refactor.c klu_scale.c klu_solve.c \
                          klu_sort.c klu_tsolve.c klu_version.c
cp_files AMD/Include      amd.h amd_internal.h
cp_files AMD/Source       amd_1.c amd_2.c amd_aat.c amd_control.c amd_defaults.c \
                          amd_dump.c amd_info.c amd_order.c amd_post_tree.c \
                          amd_postorder.c amd_preprocess.c amd_valid.c amd_version.c
cp_files BTF/Include      btf.h btf_internal.h
cp_files BTF/Source       btf_maxtrans.c btf_order.c btf_strongcomp.c btf_version.c
cp_files COLAMD/Include   colamd.h
cp_files COLAMD/Source    colamd.c colamd_version.c
cp_files SuiteSparse_config SuiteSparse_config.h SuiteSparse_config.c

cp "$src/KLU/Doc/License.txt"    "$dst/License_KLU.txt"
cp "$src/AMD/Doc/License.txt"    "$dst/License_AMD.txt"
cp "$src/BTF/Doc/License.txt"    "$dst/License_BTF.txt"
cp "$src/COLAMD/Doc/License.txt" "$dst/License_COLAMD.txt"

grep -h -E '#define (KLU|AMD|BTF|COLAMD)_(DATE|MAIN_VERSION|SUB_VERSION|SUBSUB_VERSION) ' \
    "$dst/klu.h" "$dst/amd.h" "$dst/btf.h" "$dst/colamd.h"
grep -h -E '#define SUITESPARSE_(DATE|MAIN_VERSION|SUB_VERSION|SUBSUB_VERSION) ' \
    "$dst/SuiteSparse_config.h"
