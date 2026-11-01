/*
 * ngspice/klu.h
 *
 * Single entry point for the KLU sparse solver.
 *
 *   NGSPICE_SYSTEM_KLU defined : use the KLU of an installed SuiteSparse
 *                                (found by configure or CMake)
 *   otherwise                  : use the copy of SuiteSparse bundled in
 *                                src/maths/KLU/SuiteSparse
 *
 * In both cases the ngspice extensions from klu_ext.h are added.
 */

#ifndef ngspice_KLU_H
#define ngspice_KLU_H

#ifdef NGSPICE_SYSTEM_KLU
#  if defined(HAVE_SUITESPARSE_KLU_H)
#    include <suitesparse/klu.h>
#  else
#    include <klu.h>
#  endif
#else
#  include "../../maths/KLU/SuiteSparse/klu.h"
#endif

#include "ngspice/klu_ext.h"

#endif /* ngspice_KLU_H */
