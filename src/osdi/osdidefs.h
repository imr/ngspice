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

typedef struct OsdiExtraInstData {
  double dt;
  double temp;
  bool temp_given;
  bool dt_given;
  uint32_t eval_flags;

  /* OSDI v0.5 — event-driven analog operators (S3b).
   *
   * Per-instance scratch buffers for the SimInfo.cross_expr and
   * .pending_events pointers.  Lazily allocated on first eval if
   * the descriptor advertises any cross expressions or event slots.
   * cross_expr_arr is sized to descr->num_cross_exprs; the model
   * writes current values here.  prev_cross_expr_arr stores the
   * last accepted value (one per slot) so the eval wrapper can
   * detect sign flips between consecutive accepted steps.
   *
   * pending_events_arr is sized to descr->num_event_slots; the
   * model writes one OsdiEventRequest per slot when it wants to
   * schedule a future event (timer expiration, etc.).  The eval
   * wrapper drains non-zero entries into the scheduled_events
   * queue after each accepted eval and zeros at_time to indicate
   * consumed.
   *
   * scheduled_events is the per-instance pending-event queue,
   * sorted by at_time ascending.  scheduled_count is the live
   * length; capacity is descr->num_event_slots (we never have
   * more pending than the model could request in one eval).
   *
   * cross_init flips true after the first eval has populated
   * prev_cross_expr_arr — suppresses a spurious "sign flip" at
   * t=0 when prev is still zero. */
  double *cross_expr_arr;
  double *prev_cross_expr_arr;
  OsdiEventRequest *pending_events_arr;
  OsdiEventRequest *scheduled_events;       /* sorted queue */
  uint32_t scheduled_count;
  bool cross_init;

  /* OSDI v0.5 (S3c) — abstime of the last accepted eval that
   * latched prev_cross_expr_arr.  Used as the lower bound for
   * interpolating the crossing time when the next eval detects a
   * sign flip.  Default -1.0 (no prior eval); the first latched
   * eval sets it to the eval's abstime. */
  double prev_eval_time;

  /* OSDI v0.5 — the SECOND prior accepted (time, cross_expr) sample,
   * kept so the crossing time can be QUADRATICALLY interpolated from
   * three points (t_prev2, t_prev, t_now) -> O(dt^3) error, instead of
   * the 2-point linear fit -> O(dt^2).  The cross_expr VALUE already
   * encodes both the cross-expr nonlinearity and the node-trajectory
   * curvature, so fitting a higher-order polynomial to it (no model
   * re-eval) sharpens the crossing time directly.  Matters for fast
   * carriers (PWM / RF, 100 MHz+) where the crossing TIME itself is the
   * measured quantity and a coarse local step would otherwise leave a
   * O(dt^2) ~hundreds-of-ps error.  Default -1.0 / NULL until two
   * accepted evals have latched; the detector falls back to the linear
   * fit until then. */
  double *prev2_cross_expr_arr;
  double prev2_eval_time;

  /* OSDI v0.5 (2C) — absdelay exact-transport delay rings.  Lazily
   * allocated on first eval when descr->num_delay_sites > 0.
   *   delay_input_arr[site] : current input written by the model each eval
   *       (exposed as SimInfo.delay_input); pushed into the ring at accept.
   *   delay_t[site] / delay_v[site] : grow-on-demand (time, value) ring for
   *       each site, delay_count[site] live samples in oldest..newest order,
   *       delay_cap[site] allocated capacity.  Pushed at each ACCEPTED step,
   *       pruned of samples older than abstime - max_td.  SimInfo.delay_state
   *       points back at THIS struct so SimInfo.delay_read can reach them. */
  double *delay_input_arr;
  double *delay_maxtd_arr;
  /* OSDI v0.5 (2C AC delay) — last td the model passed to delay_read per site,
   * stashed by osdi_delay_read.  At the AC operating point this holds td(OP),
   * which OSDIacLoad uses for e^{-jw*td} on that site's delay jacobian. */
  double *delay_td_arr;
  double **delay_t;
  double **delay_v;
  uint32_t *delay_count;
  uint32_t *delay_cap;

  /* OSDI 0.5 — DEFERRED-COMMIT snapshot of the model's persistent_state array
   * (transition()/slew()/event toolkit).  Sized to
   * entry->persistent_state_count, lazily allocated on first eval.  Holds the
   * value as of the last ACCEPTED step.  Each eval restores the live
   * persistent_state from this snapshot BEFORE running the model, so predictor
   * and Newton iterates always read the previous-accepted history and their
   * own writes are transient; OSDIaccept copies the converged live array back
   * into the snapshot.  persist_init flips true once seeded. */
  double *persist_snapshot;
  bool persist_init;
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

/* OSDI v0.5 (2C) — absdelay exact-transport reader (bound to
 * SimInfo.delay_read). */
double osdi_delay_read(const OsdiSimInfo *info, uint32_t site, double td,
                       double max_td);
