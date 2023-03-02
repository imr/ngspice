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
#else
#define MAX_ALIGN alignof(max_align_t)
#endif


#ifndef _MSC_VER
#define ALIGN(pow) __attribute__((aligned(pow)))
#else
#define ALIGN(pow) __declspec(align(pow))
#endif

typedef struct OsdiExtraInstData {
  double dt;
  double temp;
  bool temp_given;
  bool dt_given;
  uint32_t eval_flags;

} ALIGN(MAX_ALIGN) OsdiExtraInstData;

typedef struct OsdiModelData {
  GENmodel gen;
  max_align_t data;
} OsdiModelData;

extern size_t osdi_instance_data_off(const OsdiRegistryEntry *entry);
extern void *osdi_instance_data(const OsdiRegistryEntry *entry, GENinstance *inst);
extern OsdiExtraInstData *osdi_extra_instance_data(const OsdiRegistryEntry *entry,
                                            GENinstance *inst);
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
