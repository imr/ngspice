/* 
 * This file is part of the OSDI component of NGSPICE.
 * Copyright© 2022 SemiMod GmbH.
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. 
 *
 * Author: Pascal Kuthe <pascal.kuthe@semimod.de>
 */

#pragma once

#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/fteext.h"
#include "ngspice/gendefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/ngspice.h"
#include "ngspice/noisedef.h"
#include "ngspice/typedefs.h"

#include "osdi.h"
#include "osdiext.h"

#include <stddef.h>
#include <stdint.h>
#ifndef _MSC_VER
#include <stdalign.h>
#endif

#ifdef _MSC_VER
typedef struct {
    long long __max_align_ll ;
    long double __max_align_ld;
    /* _Float128 is defined as a basic type, so max_align_t must be
       sufficiently aligned for it.  This code must work in C++, so we
       use __float128 here; that is only available on some
       architectures, but only on i386 is extra alignment needed for
       __float128.  */
#ifdef __i386__
    __float128 __max_align_f128 __attribute__((__aligned__(__alignof(__float128))));
#endif
} max_align_t;
#endif

#ifdef _MSC_VER
#define MAX_ALIGN 8
#define alignof sizeof
#else
#define MAX_ALIGN alignof(max_align_t)
#endif


#ifndef _MSC_VER
#define ALIGN(pow) __attribute__((aligned(pow)))
#else
#define ALIGN(pow) __declspec(align(pow))
#endif

/* Per-absdelay slot descriptor read from the .osdi binary at load time. */
typedef struct OsdiAbsDelayInfo {
    uint32_t y_node;      /* OSDI node index for the synthetic input y_synth  */
    uint32_t z_node;      /* OSDI node index for the output node z             */
    uint32_t td_offset;   /* byte offset into OSDI instance data for td value  */
} OsdiAbsDelayInfo;

typedef struct OsdiExtraInstData {
  double dt;
  double temp;
  bool temp_given;
  bool dt_given;
  uint32_t eval_flags;

  /* Waveform history for absdelay — one row per slot, indexed by timepoint. */
  double **delay_hist;        /* [absdelay_count][capacity]  */
  uint32_t delay_hist_cap;    /* allocated timepoints in each row  */

  /* Pre-allocated KLU/sparse matrix pointers for the delay equation rows.
   * delay_jac_y[k] points to the (z_row, y_synth_col) entry,
   * delay_jac_z[k] points to the (z_row, z_col) entry.
   * These are the ACTIVE pointers used by the stamping code; under KLU they
   * are re-pointed between the real (CSC) and complex (CSC_Complex) arrays as
   * the analysis switches between DC/tran and AC.                */
  double **delay_jac_y;
  double **delay_jac_z;

  /* KLU only: saved real and complex CSC pointers for the delay rows, so the
   * active pointers above can be switched on each DC<->AC transition (mirrors
   * the regular Jacobian's inst_matrix_ptrs handling).  NULL under SPARSE,
   * where delay_jac_y/z already address an interleaved [real,imag] entry. */
  double **delay_jac_y_csc;
  double **delay_jac_z_csc;
  double **delay_jac_y_cx;
  double **delay_jac_z_cx;

} ALIGN(MAX_ALIGN) OsdiExtraInstData;

typedef struct OsdiModelData {
  GENmodel gen;
  max_align_t data;
} OsdiModelData;

extern size_t osdi_instance_data_off(const OsdiRegistryEntry *entry);
extern void *osdi_instance_data(const OsdiRegistryEntry *entry,
                                GENinstance *inst);
extern double *osdi_noise_data(const OsdiRegistryEntry *entry,
                                GENinstance *inst);
#ifdef KLU
extern size_t osdi_instance_matrix_ptr_off(const OsdiRegistryEntry *entry);
extern double **osdi_instance_matrix_ptr(const OsdiRegistryEntry *entry,
                                         GENinstance *inst);
#endif
extern OsdiExtraInstData *
osdi_extra_instance_data(const OsdiRegistryEntry *entry, GENinstance *inst);
extern size_t osdi_model_data_off(void);
extern void *osdi_model_data(GENmodel *model);
extern void *osdi_model_data_from_inst(GENinstance *inst);
extern OsdiRegistryEntry *osdi_reg_entry_model(const GENmodel *model);
extern OsdiRegistryEntry *osdi_reg_entry_inst(const GENinstance *inst);

typedef struct OsdiNgspiceHandle {
  uint32_t kind;
  char *name;
} OsdiNgspiceHandle;

/* values returned by $simparam*/
OsdiSimParas get_simparams(const CKTcircuit *ckt);

typedef void (*osdi_log_ptr)(void *handle, char *msg, uint32_t lvl);
void osdi_log(void *handle_, char *msg, uint32_t lvl);

typedef void (*osdi_log_ptr)(void *handle, char *msg, uint32_t lvl);

double osdi_pnjlim(bool init, bool *icheck, double vnew, double vold, double vt,
                   double vcrit);

double osdi_limvds(bool init, bool *icheck, double vnew, double vold);
double osdi_limitlog(bool init, bool *icheck, double vnew, double vold,
                     double LIM_TOL);
double osdi_fetlim(bool init, bool *icheck, double vnew, double vold,
                   double vto);
