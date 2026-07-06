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

#include "ngspice/iferrmsg.h"
#include "ngspice/memory.h"
#include "ngspice/ngspice.h"
#include "ngspice/typedefs.h"

#include "osdi.h"
#include "osdidefs.h"

#include <math.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define NUM_SIM_PARAMS 15
char *sim_params[NUM_SIM_PARAMS + 1] = {
    "iniLim", "gmin", "gdev", "tnom",
    "simulatorVersion", "sourceScaleFactor",
    "epsmin", "reltol", "vntol", "abstol",
    "osdi_vlim",
    "osdi_vlim_vds", "osdi_vlim_vgs", "osdi_vlim_vbs", "osdi_vlim_nqs",
    NULL};
char *sim_params_str[1] = {NULL};

double sim_param_vals[NUM_SIM_PARAMS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* values returned by $simparam*/
OsdiSimParas get_simparams(const CKTcircuit *ckt) {
  double simulatorVersion = strtod(PACKAGE_VERSION, NULL);
  double gdev = ckt->CKTgmin;
  double sourceScaleFactor = ckt->CKTsrcFact;
  double gmin = ((ckt->CKTgmin) > (ckt->CKTdiagGmin)) ? (ckt->CKTgmin)
                                                      : (ckt->CKTdiagGmin);
  double initializeLimiting = (ckt->CKTmode & MODEINITJCT) ? 1 : 0;

  /* osdi_vlim*: per-iteration Δv bounds consumed by OpenVA's compiler-side
   * $limit synthesis.  Each V() probe inside a branch contribution gets
   * wrapped with a clamp |V_new - V_prev| ≤ osdi_vlim_<shape>, where
   * <shape> is the MOSFET-style classification of the probe (vds/vgs/vbs),
   * with osdi_vlim as the generic fallback.  Cascade: per-shape value if
   * user set it > 0, else generic osdi_vlim, else 0.1 V default.  All
   * voltage-range-agnostic — bounds Newton step magnitude, not supply
   * rails. */
  double osdi_vlim_v = (ckt->CKTosdiVlim > 0.0) ? ckt->CKTosdiVlim : 0.1;
  double osdi_vlim_vds_v = (ckt->CKTosdiVlimVds > 0.0) ? ckt->CKTosdiVlimVds : osdi_vlim_v;
  double osdi_vlim_vgs_v = (ckt->CKTosdiVlimVgs > 0.0) ? ckt->CKTosdiVlimVgs : osdi_vlim_v;
  double osdi_vlim_vbs_v = (ckt->CKTosdiVlimVbs > 0.0) ? ckt->CKTosdiVlimVbs : osdi_vlim_v;
  double osdi_vlim_nqs_v = (ckt->CKTosdiVlimNqs > 0.0) ? ckt->CKTosdiVlimNqs : osdi_vlim_v;
  double sim_param_vals_[NUM_SIM_PARAMS] = {
      // Verilog-A tnom is in degrees Celsius
      initializeLimiting, gmin, gdev, ckt->CKTnomTemp-CONSTCtoK, simulatorVersion, sourceScaleFactor,
      ckt->CKTepsmin, ckt->CKTreltol, ckt->CKTvoltTol, ckt->CKTabstol,
      osdi_vlim_v,
      osdi_vlim_vds_v, osdi_vlim_vgs_v, osdi_vlim_vbs_v, osdi_vlim_nqs_v };
  memcpy(&sim_param_vals, &sim_param_vals_, sizeof(double) * NUM_SIM_PARAMS);
  OsdiSimParas sim_params_ = {.names = sim_params,
                              .vals = (double *)&sim_param_vals,
                              .names_str = sim_params_str,
                              .vals_str = NULL};
  return sim_params_;
}

/* OSDI v0.5 (S3b) — per-instance event-state preparation BEFORE
 * descr->eval is called.  Lazy-allocate cross_expr / pending_events
 * buffers (sized from the descriptor), wire them onto the per-eval
 * SimInfo, and decide whether this eval is at a previously
 * scheduled event time.
 *
 * `sim_info_local` is a per-eval mutable copy of the caller's
 * shared SimInfo — the cross/event pointers are per-instance and
 * must not share across threads under OMP. */
static void osdi_event_prepare(OsdiExtraInstData *extra,
                               const OsdiDescriptor *descr,
                               uint32_t num_delay_sites,
                               OsdiSimInfo *sim_info_local) {
  if (descr->num_cross_exprs > 0 && extra->cross_expr_arr == NULL) {
    extra->cross_expr_arr = TMALLOC(double, descr->num_cross_exprs);
    extra->prev_cross_expr_arr = TMALLOC(double, descr->num_cross_exprs);
    extra->prev2_cross_expr_arr = TMALLOC(double, descr->num_cross_exprs);
    for (uint32_t i = 0; i < descr->num_cross_exprs; i++) {
      extra->cross_expr_arr[i] = 0.0;
      extra->prev_cross_expr_arr[i] = 0.0;
      extra->prev2_cross_expr_arr[i] = 0.0;
    }
    extra->prev_eval_time = -1.0;
    extra->prev2_eval_time = -1.0;
  }
  if (descr->num_event_slots > 0 && extra->pending_events_arr == NULL) {
    extra->pending_events_arr =
        TMALLOC(OsdiEventRequest, descr->num_event_slots);
    extra->scheduled_events =
        TMALLOC(OsdiEventRequest, descr->num_event_slots);
    for (uint32_t i = 0; i < descr->num_event_slots; i++) {
      extra->pending_events_arr[i].at_time = 0.0;
      extra->pending_events_arr[i].event_id = 0;
      extra->pending_events_arr[i].kind = 0;
      extra->scheduled_events[i].at_time = 0.0;
      extra->scheduled_events[i].event_id = 0;
      extra->scheduled_events[i].kind = 0;
    }
    extra->scheduled_count = 0;
  }
  if (num_delay_sites > 0 && extra->delay_input_arr == NULL) {
    extra->delay_input_arr = TMALLOC(double, num_delay_sites);
    extra->delay_maxtd_arr = TMALLOC(double, num_delay_sites);
    extra->delay_td_arr = TMALLOC(double, num_delay_sites);
    extra->delay_t = TMALLOC(double *, num_delay_sites);
    extra->delay_v = TMALLOC(double *, num_delay_sites);
    extra->delay_count = TMALLOC(uint32_t, num_delay_sites);
    extra->delay_cap = TMALLOC(uint32_t, num_delay_sites);
    for (uint32_t i = 0; i < num_delay_sites; i++) {
      extra->delay_input_arr[i] = 0.0;
      extra->delay_maxtd_arr[i] = -1.0;
      extra->delay_td_arr[i] = 0.0;
      extra->delay_t[i] = NULL;
      extra->delay_v[i] = NULL;
      extra->delay_count[i] = 0;
      extra->delay_cap[i] = 0;
    }
  }

  sim_info_local->cross_expr = extra->cross_expr_arr;
  sim_info_local->pending_events = extra->pending_events_arr;
  sim_info_local->delay_input = extra->delay_input_arr;
  sim_info_local->delay_maxtd = extra->delay_maxtd_arr;
  sim_info_local->delay_state = (void *)extra;
  sim_info_local->delay_read = osdi_delay_read;

  /* Is this eval at or PAST the front-of-queue scheduled event
   * time?  S3c plan-B firing semantics: cross events are scheduled
   * at a linear-interpolated t_event that almost always lands
   * BETWEEN two simulator steps — there's no way to make the
   * stepper land exactly on t_event without back-stepping (which
   * dctran's /8 rejection path makes impractical).  Instead we
   * fire on the first eval at or past t_event and report the
   * true t_event to the model via scheduled_event_time so the
   * lowering can latch the sample-exact value, not the post-fire
   * abstime. */
  sim_info_local->at_scheduled_event = 0;
  sim_info_local->fired_event_id = 0;
  sim_info_local->scheduled_event_time = 0.0;
  if (extra->scheduled_count > 0) {
    double t_event = extra->scheduled_events[0].at_time;
    if (sim_info_local->abstime >= t_event - 1.0e-12) {
      sim_info_local->at_scheduled_event = 1;
      sim_info_local->fired_event_id = extra->scheduled_events[0].event_id;
      sim_info_local->scheduled_event_time = t_event;
      /* Pop the consumed event. */
      for (uint32_t i = 1; i < extra->scheduled_count; i++) {
        extra->scheduled_events[i - 1] = extra->scheduled_events[i];
      }
      extra->scheduled_count--;
    }
  }
}

/* Crossing time of a cross_expr within the sign-flip bracket [t1, t2]
 * (v1, v2 have opposite signs).  When a second prior sample (t0, v0) is
 * available, fit a quadratic through the three (time, value) points and
 * take the in-bracket root -> O(dt^3) error; otherwise fall back to the
 * 2-point linear fit -> O(dt^2).  The cross_expr VALUE already folds in
 * both the cross-expr nonlinearity and the node-trajectory curvature, so
 * a higher-order fit of the value sharpens the crossing time with no
 * model re-evaluation.  Degenerate quadratics (near-zero leading coeff,
 * negative discriminant, no in-bracket root) fall back to linear. */
static double osdi_cross_time(double t0, double v0,
                              double t1, double v1,
                              double t2, double v2) {
  double span = t2 - t1;
  if (t0 >= 0.0 && t1 > t0 && span > 0.0) {
    /* Newton divided differences centered on [t1, t2]:
     *   v(t) = v1 + s*f12 + s*(s + (t1 - t2))*f012,  s = t - t1
     *        = f012*s^2 + (f12 + (t1 - t2)*f012)*s + v1 */
    double f12 = (v2 - v1) / span;
    double f01 = (v1 - v0) / (t1 - t0);
    double f012 = (f12 - f01) / (t2 - t0);
    double A = f012;
    double B = f12 + (t1 - t2) * f012;
    double C = v1;
    if (A != 0.0) {
      double disc = B * B - 4.0 * A * C;
      if (disc >= 0.0) {
        double sq = sqrt(disc);
        double s1 = (-B + sq) / (2.0 * A);
        double s2 = (-B - sq) / (2.0 * A);
        bool in1 = (s1 > 0.0 && s1 <= span);
        bool in2 = (s2 > 0.0 && s2 <= span);
        double s = -1.0;
        if (in1 && in2) {
          s = (s1 < s2) ? s1 : s2; /* earliest crossing in the bracket */
        } else if (in1) {
          s = s1;
        } else if (in2) {
          s = s2;
        }
        if (s >= 0.0) {
          return t1 + s;
        }
      }
    }
    /* fall through to linear for a degenerate quadratic */
  }
  double denom = fabs(v1) + fabs(v2);
  return (denom > 0.0) ? t1 + span * (fabs(v1) / denom) : t2;
}

/* OSDI v0.5 — post-eval processing.
 *
 * S3b: drain model-written pending_events into the sorted
 *      scheduled_events queue.
 * S3c (plan B): detect cross_expr sign flips and schedule the
 *      cross event at a linear-interpolated crossing time.  The
 *      original idea — back-step the simulator and bisect — runs
 *      into ngspice's reject path: dctran cuts CKTdelta by 8 per
 *      rejection cumulatively and skips CKTtrunc between retries,
 *      so the bisection dt-clamp never gets to drive the next
 *      step toward the bracket midpoint.  Linear interpolation
 *      gives O(dt^2) accuracy for smooth signals — typically a
 *      ~1000× improvement over S3b's "post-flip step abstime"
 *      without touching the transient stepper.
 *
 * The interpolated t_event almost always falls between two
 * simulator steps.  Prepare() handles that mismatch by firing the
 * event on the first eval at or past t_event and exposing the
 * true t_event via sim_info_local->scheduled_event_time. */
static void osdi_event_postprocess(OsdiExtraInstData *extra,
                                   const OsdiDescriptor *descr,
                                   const OsdiSimInfo *sim_info_local) {
  /* 1. Drain pending events. */
  if (descr->num_event_slots > 0 &&
      (extra->eval_flags & EVAL_RET_FLAG_EVENT)) {
    for (uint32_t i = 0; i < descr->num_event_slots; i++) {
      OsdiEventRequest *req = &extra->pending_events_arr[i];
      if (req->at_time > sim_info_local->abstime &&
          extra->scheduled_count < descr->num_event_slots) {
        /* Insert into scheduled_events keeping sorted order. */
        uint32_t j = 0;
        while (j < extra->scheduled_count &&
               extra->scheduled_events[j].at_time < req->at_time) {
          j++;
        }
        for (uint32_t k = extra->scheduled_count; k > j; k--) {
          extra->scheduled_events[k] = extra->scheduled_events[k - 1];
        }
        extra->scheduled_events[j] = *req;
        extra->scheduled_count++;
      }
      /* Zero the pending slot to indicate consumed. */
      req->at_time = 0.0;
      req->event_id = 0;
      req->kind = 0;
    }
  }

  /* 2. Detect cross sign flips + schedule at linear-interp time. */
  if (descr->num_cross_exprs > 0 &&
      (extra->eval_flags & EVAL_RET_FLAG_CROSS) &&
      extra->cross_init && extra->prev_eval_time >= 0.0) {
    double t_now = sim_info_local->abstime;
    for (uint32_t i = 0; i < descr->num_cross_exprs; i++) {
      const OsdiCrossExprMeta *meta = &descr->cross_expr_metadata[i];
      double prev_val = extra->prev_cross_expr_arr[i];
      double curr_val = extra->cross_expr_arr[i];
      bool flipped = false;
      if (meta->direction > 0 && prev_val < 0.0 && curr_val >= 0.0) {
        flipped = true;
      } else if (meta->direction < 0 &&
                 prev_val > 0.0 && curr_val <= 0.0) {
        flipped = true;
      } else if (meta->direction == 0 && prev_val * curr_val < 0.0) {
        flipped = true;
      }
      if (flipped && extra->scheduled_count < descr->num_event_slots) {
        /* Crossing time within [t_prev, t_now].  A 3-point quadratic fit
         * through the second-prior sample (O(dt^3)) when available, else
         * the 2-point linear fit (O(dt^2)).  See osdi_cross_time(). */
        double t_cross = osdi_cross_time(
            extra->prev2_eval_time, extra->prev2_cross_expr_arr[i],
            extra->prev_eval_time, prev_val,
            t_now, curr_val);
        extra->scheduled_events[extra->scheduled_count].at_time = t_cross;
        extra->scheduled_events[extra->scheduled_count].event_id =
            meta->event_id;
        extra->scheduled_events[extra->scheduled_count].kind =
            OSDI_EVENT_KIND_CROSS;
        extra->scheduled_count++;
      }
    }
  }

  /* 3. Latch current cross values + times, shifting the 2-deep history
   *    (prev2 <- prev <- curr) so the next flip can fit a quadratic. */
  if (descr->num_cross_exprs > 0) {
    for (uint32_t i = 0; i < descr->num_cross_exprs; i++) {
      extra->prev2_cross_expr_arr[i] = extra->prev_cross_expr_arr[i];
      extra->prev_cross_expr_arr[i] = extra->cross_expr_arr[i];
    }
    extra->cross_init = true;
    extra->prev2_eval_time = extra->prev_eval_time;
    extra->prev_eval_time = sim_info_local->abstime;
  }
}

static void eval(const OsdiDescriptor *descr, const GENinstance *gen_inst,
                 void *inst, OsdiExtraInstData *extra_inst_data,
                 const void *model, const OsdiSimInfo *sim_info) {

  OsdiNgspiceHandle handle =
      (OsdiNgspiceHandle){.kind = 3, .name = gen_inst->GENname};

  /* OSDI v0.5 — per-instance scratch SimInfo with event-state
   * pointers wired up.  Originals stay shared/read-only across the
   * OMP parallel region. */
  OsdiSimInfo sim_info_local = *sim_info;
  OsdiRegistryEntry *entry = osdi_reg_entry_inst(gen_inst);
  osdi_event_prepare(extra_inst_data, descr, entry->num_delay_sites,
                     &sim_info_local);

  /* TODO initial conditions? */
  extra_inst_data->eval_flags =
      descr->eval(&handle, inst, model, &sim_info_local);

  osdi_event_postprocess(extra_inst_data, descr, &sim_info_local);
}

static void sanitize_residuals(CKTcircuit *ckt, void *inst,
                               const OsdiDescriptor *descr, bool is_tran,
                               const char *name, double t) {
  /* NaN/Inf in a residual means the model's eval produced a non-finite
   * value — always a real anomaly.  Zero it out and signal non-
   * convergence so Newton retries with a smaller step.
   *
   * Finite-magnitude clamping was removed: it had no physical basis.
   * For body-diode-style "huge" residuals (V/Vt deep enough that
   * exp(V/Vt) blows up to 1e83+ A), the model's Jacobian scales with
   * the current itself, so Newton's update Δx = -F/J is naturally
   * bounded by the model.  And in power designs (PMIC, LDMOS arrays,
   * battery switches) a single OSDI instance with m × nf in the
   * thousands can legitimately stamp hundreds-to-thousands of amps —
   * clipping those was breaking convergence.  HSPICE and Spectre do
   * not magnitude-clip residuals either. */
  for (uint32_t i = 0; i < descr->num_nodes; i++) {
    if (descr->nodes[i].resist_residual_off != UINT32_MAX) {
      double *r = (double *)(((char *)inst) + descr->nodes[i].resist_residual_off);
      if (!isfinite(*r)) {
        *r = 0.0; ckt->CKTnoncon++;
      }
    }
    if (is_tran && descr->nodes[i].react_residual_off != UINT32_MAX) {
      double *r = (double *)(((char *)inst) + descr->nodes[i].react_residual_off);
      if (!isfinite(*r)) {
        *r = 0.0; ckt->CKTnoncon++;
      }
    }
  }
}

static void sanitize_jacobian(CKTcircuit *ckt, void *inst,
                              const OsdiDescriptor *descr,
                              const char *name, double t) {
  double **jptr =
      (double **)(((char *)inst) + descr->jacobian_ptr_resist_offset);
  bool is_tran = (ckt->CKTmode & MODETRAN) != 0;
  for (uint32_t i = 0; i < descr->num_jacobian_entries; i++) {
    bool is_diag = (descr->jacobian_entries[i].nodes.node_1 ==
                    descr->jacobian_entries[i].nodes.node_2);
    /* During transient, load_jacobian_tran folds the reactive Jacobian into
       this resistive slot as ag0*dQ/dV, where ag0 ~ 1/dt.  At a small step
       (e.g. the 1e-14 s initial step) a perfectly physical reactive coupling
       can therefore be huge -- a 1 pF cap gives 1e2, a 1 F differentiator
       (bare ddt(V)) gives 1e14.  That is NOT a bad linearization, so it must
       not be hit by the huge-finite cap below (which exists to bound bad
       *resistive* linearizations).  The underlying dQ/dV is still guarded by
       the reactive-pointer cap further down.  PDK transcaps (~1e-15 F) never
       approach the threshold even at the initial step, so they are unaffected. */
    bool has_react = descr->jacobian_entries[i].react_ptr_off != UINT32_MAX;
    if (jptr[i] && !isfinite(*jptr[i])) {
      if (is_diag) {
        /* Use |I_resist|/VDD so N-R step stays bounded to ~VDD instead of
           the 2.4MV update that gmin alone would allow. */
        double g_replacement = ckt->CKTgmin;
        uint32_t node_idx = descr->jacobian_entries[i].nodes.node_1;
        if (node_idx < descr->num_nodes &&
            descr->nodes[node_idx].resist_residual_off != UINT32_MAX) {
          double resid = fabs(*((double *)(((char *)inst) +
                              descr->nodes[node_idx].resist_residual_off)));
          double g_safe = resid / 0.9;
          if (g_safe > g_replacement) g_replacement = g_safe;
        }
        *jptr[i] = g_replacement;
      } else {
        *jptr[i] = 0.0;
      }
      ckt->CKTnoncon++;
      /* Axis-3: NaN/Inf in the Jacobian means the model's evaluation
         diverged at the predicted operating point.  No amount of
         clipping or iteration can recover a falsified linearization;
         signal step rejection so the transient driver retries with
         smaller delta.  Gated by .option noosdistepreject. */
      if (!ckt->CKTosdiStepRejectOff)
        ckt->CKTosdiStepReject = 1;
    } else if (jptr[i] && fabs(*jptr[i]) > 1.0e6 && !(is_tran && has_react)) {
      /* Huge-finite Jacobian: cap so the N-R solve cannot produce absurd
         node updates.  CKTnoncon is incremented so the iteration loop
         knows the iterate hasn't truly settled.  CKThugeJThisIter is
         incremented so NIiter knows to apply per-node Δv limiting on
         this iteration.  Skipped for reactive-bearing entries in transient
         (see has_react note above) -- their magnitude is the physical
         ag0*dQ/dV companion, not a falsified linearization. */
      *jptr[i] = is_diag ? ((*jptr[i] > 0) ? 1.0e6 : -1.0e6) : 0.0;
      ckt->CKTnoncon++;
      ckt->CKThugeJThisIter++;
    }
    /* Sanitize the per-entry reactive Jacobian pointer (separate sparse-matrix
       slot from the resist stamp; may be NaN independently). */
    uint32_t rpt_off = descr->jacobian_entries[i].react_ptr_off;
    if (rpt_off != UINT32_MAX) {
      double *rp = *(double **)(((char *)inst) + rpt_off);
      if (rp && !isfinite(*rp)) {
        *rp = is_diag ? ckt->CKTgmin : 0.0;
        ckt->CKTnoncon++;
        if (!ckt->CKTosdiStepRejectOff)
          ckt->CKTosdiStepReject = 1;
      } else if (rp && fabs(*rp) > 1.0e6) {
        *rp = is_diag ? 1.0e6 : 0.0;
        ckt->CKTnoncon++;
        ckt->CKThugeJThisIter++;
      }
    }
  }
}

static void load(CKTcircuit *ckt, const GENinstance *gen_inst, void *model,
                 void *inst, OsdiExtraInstData *extra_inst_data, bool is_tran,
                 bool is_init_tran, const OsdiDescriptor *descr) {

  NG_IGNORE(extra_inst_data);

  const char *name = gen_inst->GENname;
  double t = ckt->CKTtime;
  double dump;

  if (is_tran) {
    /* zero NaN residuals before they propagate into the RHS */
    sanitize_residuals(ckt, inst, descr, true, name, t);

    /* load dc matrix and capacitances (charge derivative multiplied with
     * CKTag[0]) */
    descr->load_jacobian_tran(inst, model, ckt->CKTag[0]);

    /* zero any NaN entries written into the sparse matrix */
    sanitize_jacobian(ckt, inst, descr, name, t);

    /* load static rhs and dynamic linearized rhs (SUM Vb * dIa/dVb)*/
    descr->load_spice_rhs_tran(inst, model, ckt->CKTrhs, ckt->CKTrhsOld,
                               ckt->CKTag[0]);

    uint32_t *node_mapping =
        (uint32_t *)(((char *)inst) + descr->node_mapping_offset);

    /* use numeric integration to obtain the remainer of the RHS*/
    int state = gen_inst->GENstate + (int)descr->num_states;
    for (uint32_t i = 0; i < descr->num_nodes; i++) {
      if (descr->nodes[i].react_residual_off != UINT32_MAX) {

        double residual_react =
            *((double *)(((char *)inst) + descr->nodes[i].react_residual_off));

        /* save previous accepted charge for oscillation detection */
        double q_old_charge = ckt->CKTstate1[state];

        /* store charges in state vector*/
        ckt->CKTstate0[state] = residual_react;
        if (is_init_tran) {
          ckt->CKTstate1[state] = residual_react;
        }

        /* we only care about the numeric integration itself not ceq/geq
        because those are already calculated by load_jacobian_tran and
        load_spice_rhs_tran*/
        NIintegrate(ckt, &dump, &dump, 0, state);

        /* add the numeric derivative to the rhs */
        double integrated = ckt->CKTstate0[state + 1];
        double stamp;
        if (!isfinite(integrated)) {
          /* NaN/inf: reset history to prevent propagation */
          ckt->CKTstate0[state + 1] = 0.0;
          stamp = 0.0;
          ckt->CKTnoncon++;
        } else if (residual_react == q_old_charge) {
          /* Charge is identical to the previous accepted value (zeroth-order
             predictor).  TRAP gives q_dot_new = −q_dot_old, so the sign
             flips every timestep — an artificial oscillation that reverses
             the sign of the RHS contribution and derails the first N-R step.
             Zero both the stamp and the stored derivative to kill it. */
          ckt->CKTstate0[state + 1] = 0.0;
          stamp = 0.0;
        } else {
          /* Finite q_dot of any magnitude is stamped as-is.  No
             magnitude clamp — see sanitize_residuals comment for why
             clipping was actively wrong for power devices.  Genuine
             overflow surfaces as NaN/Inf and is caught above. */
          stamp = integrated;
        }
        ckt->CKTrhs[node_mapping[i]] -= stamp;

        if (is_init_tran) {
          ckt->CKTstate1[state + 1] = ckt->CKTstate0[state + 1];
        }

        state += 2;
      }
    }
  } else {
    /* zero NaN residuals before they propagate into the RHS */
    sanitize_residuals(ckt, inst, descr, false, name, t);

    /* copy internal derivatives into global matrix */
    descr->load_jacobian_resist(inst, model);

    /* zero any NaN entries written into the sparse matrix */
    sanitize_jacobian(ckt, inst, descr, name, t);

    /* calculate spice RHS from internal currents and store into global RHS
     */
    descr->load_spice_rhs_dc(inst, model, ckt->CKTrhs, ckt->CKTrhsOld);

    /* Seed the reactive (charge) state from the transient operating point.
     *
     * The OP solve runs with is_tran == 0, so the reactive integration block
     * above is skipped and the per-node charge slots CKTstate0[state] are
     * never written -- they stay 0.  dctran then memcpy's CKTstate0 ->
     * CKTstate1 right before MODEINITTRAN, propagating that 0.  The first
     * transient step therefore sees a charge that jumps from 0 to the real
     * OP charge in one step, injecting a phantom dq/dt (an OSDI device that
     * holds a nonzero charge at the bias point -- e.g. a DC-biased cap --
     * starts the transient as if discharged and relaxes back over its RC).
     *
     * Native ngspice devices avoid this by seeding CKTstate1[qcap] =
     * CKTstate0[qcap] = C*vcap at the operating point (see capload.c).  Mirror
     * that here: CALC_REACT_RESIDUAL was requested for the transient OP
     * (MODETRANOP) above, so the model's react residual now holds the OP
     * charge; store it into both state slots so the first step's dq/dt = 0. */
    if (ckt->CKTmode & MODETRANOP) {
      int state = gen_inst->GENstate + (int)descr->num_states;
      for (uint32_t i = 0; i < descr->num_nodes; i++) {
        if (descr->nodes[i].react_residual_off != UINT32_MAX) {
          double residual_react =
              *((double *)(((char *)inst) + descr->nodes[i].react_residual_off));
          if (isfinite(residual_react)) {
            ckt->CKTstate0[state] = residual_react;
            ckt->CKTstate1[state] = residual_react;
          }
          state += 2;
        }
      }
    }
  }
}

/* OSDI v0.5 (2C) — push one accepted (time, value) sample into delay ring
 * `s`, growing capacity geometrically, then prune leading samples that can
 * no longer bracket any delay <= maxtd (keeping >= 2 for interpolation).
 * maxtd < 0 means unbounded — capped at a defensive ceiling so a deck that
 * never supplies max_delay can't grow the ring without bound. */
static void osdi_delay_push(OsdiExtraInstData *extra, uint32_t s, double t,
                            double v, double maxtd) {
  uint32_t n = extra->delay_count[s];
  if (n == extra->delay_cap[s]) {
    uint32_t newcap = extra->delay_cap[s] ? extra->delay_cap[s] * 2 : 64;
    extra->delay_t[s] = TREALLOC(double, extra->delay_t[s], newcap);
    extra->delay_v[s] = TREALLOC(double, extra->delay_v[s], newcap);
    extra->delay_cap[s] = newcap;
  }
  extra->delay_t[s][n] = t;
  extra->delay_v[s][n] = v;
  extra->delay_count[s] = n + 1;

  double *tt = extra->delay_t[s];
  double *vv = extra->delay_v[s];
  uint32_t cnt = extra->delay_count[s];
  uint32_t drop = 0;
  if (maxtd >= 0.0) {
    double horizon = t - maxtd;
    /* advance past samples strictly older than the one bracketing the
     * horizon; keep that bracketing sample and everything newer. */
    while (drop + 2 < cnt && tt[drop + 1] <= horizon) {
      drop++;
    }
  } else {
    const uint32_t CEIL = 1u << 22; /* ~4M samples */
    if (cnt > CEIL) {
      drop = cnt - CEIL;
    }
  }
  if (drop > 0) {
    memmove(tt, tt + drop, (cnt - drop) * sizeof(double));
    memmove(vv, vv + drop, (cnt - drop) * sizeof(double));
    extra->delay_count[s] = cnt - drop;
  }
}

/* OSDI 0.5 — DEFERRED-COMMIT persistent state.
 *
 * The model's persistent_state array (transition()/slew()/event toolkit) is
 * read-modify-written in place during eval, so a predictor eval and every
 * Newton iterate within a timestep would overwrite the previous-ACCEPTED value
 * the model needs to read back -- in a loose-feedback wrapper that corrupts the
 * state and the builtin computes the wrong value.  Fix: keep a per-instance
 * snapshot of the array as of the last accepted step.  Restore the live array
 * from the snapshot BEFORE every eval (so reads always see the committed value
 * and in-step writes are transient); commit the snapshot from the live array
 * only at an accepted step (OSDIaccept).
 *
 * osdi_persist_restore: called before each eval.  Lazily allocates the snapshot
 * (capturing the initial zero state) on first use, then copies snapshot -> live.
 */
static void osdi_persist_restore(const OsdiRegistryEntry *entry,
                                 GENinstance *gen_inst, void *inst,
                                 OsdiExtraInstData *extra) {
  uint32_t n = entry->persistent_state_count;
  if (n == 0) {
    return;
  }
  NG_IGNORE(gen_inst);
  double *live = (double *)(((char *)inst) + entry->persistent_state_offset);
  if (!extra->persist_snapshot) {
    /* First eval: capture the (zero-initialised) live array as the committed
     * baseline.  Do NOT overwrite live, so this eval still sees is_first. */
    extra->persist_snapshot = TMALLOC(double, n);
    memcpy(extra->persist_snapshot, live, (size_t)n * sizeof(double));
    extra->persist_init = true;
  } else {
    /* Restore the previous-accepted value before the model reads it. */
    memcpy(live, extra->persist_snapshot, (size_t)n * sizeof(double));
  }
}

/* osdi_persist_commit: called once per ACCEPTED step (OSDIaccept).  Copies the
 * converged live array into the snapshot so the next step reads it back. */
static void osdi_persist_commit(const OsdiRegistryEntry *entry, void *inst,
                                OsdiExtraInstData *extra) {
  uint32_t n = entry->persistent_state_count;
  if (n == 0 || !extra->persist_snapshot) {
    return;
  }
  double *live = (double *)(((char *)inst) + entry->persistent_state_offset);
  memcpy(extra->persist_snapshot, live, (size_t)n * sizeof(double));
}

/* OSDI v0.5 (2C) — DEVaccept hook.  Fires once per ACCEPTED transient step
 * (from CKTaccept); records each instance's current delay input (written by
 * the model during the converged eval) into its per-site transport-delay
 * ring.  Only the accepted trajectory is recorded, which the model itself
 * could not do (it can't tell an accepted step from a Newton iterate). */
int OSDIaccept(CKTcircuit *ckt, GENmodel *in_model) {
  OsdiRegistryEntry *entry = osdi_reg_entry_model(in_model);
  uint32_t num_delay_sites = entry->num_delay_sites;
  uint32_t persistent_count = entry->persistent_state_count;
  /* Nothing per-instance to do if this model has neither delay rings nor
   * deferred-commit persistent state. */
  if (num_delay_sites == 0 && persistent_count == 0) {
    return OK;
  }
  double t = ckt->CKTtime;
  for (GENmodel *model = in_model; model; model = model->GENnextModel) {
    for (GENinstance *inst = model->GENinstances; inst;
         inst = inst->GENnextInstance) {
      OsdiExtraInstData *extra = osdi_extra_instance_data(entry, inst);
      if (!extra) {
        continue;
      }
      /* Commit the converged persistent state (the model's history) for this
       * accepted step so the next step reads it back. */
      if (persistent_count) {
        void *idata = osdi_instance_data(entry, inst);
        osdi_persist_commit(entry, idata, extra);
      }
      if (num_delay_sites && extra->delay_input_arr != NULL) {
        for (uint32_t s = 0; s < num_delay_sites; s++) {
          osdi_delay_push(extra, s, t, extra->delay_input_arr[s],
                          extra->delay_maxtd_arr[s]);
        }
      }
    }
  }
  return OK;
}

extern int OSDIload(GENmodel *inModel, CKTcircuit *ckt) {
  GENmodel *gen_model;
  GENinstance *gen_inst;

  bool is_init_smsig = ckt->CKTmode & MODEINITSMSIG;
  bool is_dc = ckt->CKTmode & (MODEDCOP | MODEDCTRANCURVE);
  bool is_ac = ckt->CKTmode & (MODEAC | MODEINITSMSIG);
  bool is_tran = ckt->CKTmode & (MODETRAN);
  bool is_tran_op = ckt->CKTmode & (MODETRANOP);
  bool is_init_tran = ckt->CKTmode & MODEINITTRAN;
  /* INIT_LIM gating: fire whenever the simulator is in any DC-OP
   * initialization mode where PrevState may be stale (== 0).  Includes
   * the literal first iter (MODEINITJCT), the floating-restart mode used
   * by gmin and source stepping when normal Newton fails (MODEINITFLOAT),
   * and the fixed-IC mode (MODEINITFIX).  All three lead the simulator
   * to re-enter CKTload with PrevState still uninitialized, so the
   * compiler-side $limit synthesis must bypass its per-iter Δv clamp
   * (which would otherwise pin every V() to ±vlim of zero, derailing
   * the OP-finding machinery).  Transient init modes (MODEINITTRAN /
   * MODEINITPRED) are intentionally EXCLUDED — by the time transient
   * starts the OP has converged and PrevState carries a valid value
   * from there; the clamp is desired throughout transient. */
  bool is_init_junc = ckt->CKTmode &
                      (MODEINITJCT | MODEINITFLOAT | MODEINITFIX);

  OsdiSimInfo sim_info = {
      .paras = get_simparams(ckt),
      .abstime = is_tran ? ckt->CKTtime : 0.0,
      .prev_solve = ckt->CKTrhsOld,
      .prev_state = ckt->CKTstates[0],
      .next_state = ckt->CKTstates[0],
      .flags = CALC_RESIST_JACOBIAN,
      /* Axis 2 — current Newton iteration index for iteration-aware
       * limiting (published by NIiter before this CKTload). */
      .newton_iter = (uint32_t)ckt->CKTosdiNewtonIter,
  };

  /* OSDI limiting at the DC->transient boundary.
   *
   * The compiler-synthesized $limit on V() probes is a Newton step limiter:
   * it clamps |V_probe - PrevState| to a knee (~10*kT/q).  Throughout
   * transient PrevState is read from CKTstates[0] (== next_state), which works
   * because each N-R iteration writes its NewState there for the next one to
   * read.  But at the very first transient step (MODEINITTRAN) CKTstates[0]
   * does not hold the operating point: dctran rotates the state pointers
   * (cktdefs: CKTstates[i+1] = CKTstates[i]) just before this solve, which
   * moves the converged OP limiter state into CKTstates[1] and leaves
   * CKTstates[0] pointing at a recycled (stale, == 0) buffer.  PrevState would
   * therefore be 0, and the limiter would clamp the steady-state OP voltage as
   * if stepping up from zero -- e.g. a device biased at 1 V starts the
   * transient with its probe log-compressed to ~0.61 V, corrupting the first
   * step (a linear cap then "converges" at the clamped value; stiff models
   * iterate out of it, which is why this stayed hidden).
   *
   * Read PrevState from CKTstates[1] (the rotated-away OP state) for this one
   * step so the limiter sees delta = V_guess - V_op ~ 0 and stays transparent.
   * NewState is still written to CKTstates[0].  Limiting remains fully active
   * for every subsequent step. */
  if ((ckt->CKTmode & MODEINITTRAN) && ckt->CKTstates[1]) {
    sim_info.prev_state = ckt->CKTstates[1];
  }

  sim_info.flags |= CALC_OP;

  if (is_dc) {
    sim_info.flags |= ANALYSIS_DC | ANALYSIS_STATIC;
  }

  if (!is_init_smsig) {
    sim_info.flags |= CALC_RESIST_RESIDUAL | ENABLE_LIM | CALC_RESIST_LIM_RHS;
  }

  if (is_tran) {
    sim_info.flags |= CALC_REACT_JACOBIAN | CALC_REACT_RESIDUAL |
                      CALC_REACT_LIM_RHS | ANALYSIS_TRAN;

    /* OSDI 0.5 — flag the final transient timepoint so a model can fire
     * an @(final_step) body.  dctran puts a hard breakpoint at
     * CKTfinalTime, so the last accepted step lands exactly on it. */
    if (AlmostEqualUlps(ckt->CKTtime, ckt->CKTfinalTime, 100)) {
      sim_info.at_final_step = 1;
    }
  }

  if (is_tran_op) {
    sim_info.flags |= ANALYSIS_TRAN;
    /* Compute the reactive residual (charge) at the transient operating
     * point so it can be seeded into the state vector.  The OP solve itself
     * does not stamp any reactive term (the DC load path below only loads the
     * resistive Jacobian/RHS), so requesting the charge here is side-effect
     * free for the OP and gives the first transient step a correct dq/dt = 0
     * starting history (see the seeding block in load()). */
    sim_info.flags |= CALC_REACT_RESIDUAL;
  }

  if (is_ac) {
    sim_info.flags |= CALC_REACT_JACOBIAN | ANALYSIS_AC;
  }

  if (is_init_tran) {
    sim_info.flags |= ANALYSIS_IC | ANALYSIS_STATIC;
  }

  if (is_init_junc) {
    sim_info.flags |= INIT_LIM;
  }

  if (ckt->CKTmode & MODEACNOISE) {
    sim_info.flags |= ANALYSIS_NOISE;
  }
  sim_info.flags |= CALC_NOISE;

  OsdiRegistryEntry *entry = osdi_reg_entry_model(inModel);
  const OsdiDescriptor *descr = entry->descriptor;
  uint32_t eval_flags = 0;

#ifdef USE_OMP
  int ret = OK;

  /* use openmp 3.0 tasks to parallelize linked list transveral */
#pragma omp parallel
#pragma omp single
  {
    for (gen_model = inModel; gen_model; gen_model = gen_model->GENnextModel) {
      void *model = osdi_model_data(gen_model);

      for (gen_inst = gen_model->GENinstances; gen_inst;
           gen_inst = gen_inst->GENnextInstance) {

        void *inst = osdi_instance_data(entry, gen_inst);

        OsdiExtraInstData *extra_inst_data =
            osdi_extra_instance_data(entry, gen_inst);

        /* Deferred-commit: feed this eval the previous-accepted persistent
         * state (done synchronously before the async task). */
        osdi_persist_restore(entry, gen_inst, inst, extra_inst_data);

#pragma omp task firstprivate(gen_inst, inst, extra_inst_data, model)
        eval(descr, gen_inst, inst, extra_inst_data, model, &sim_info);
      }
    }
  }

  /* init small signal analysis does not require loading values into
   * matrix/rhs*/
  if (is_init_smsig) {
    return ret;
  }

  for (gen_model = inModel; gen_model; gen_model = gen_model->GENnextModel) {
    void *model = osdi_model_data(gen_model);

    for (gen_inst = gen_model->GENinstances; gen_inst;
         gen_inst = gen_inst->GENnextInstance) {
      void *inst = osdi_instance_data(entry, gen_inst);
      OsdiExtraInstData *extra_inst_data =
          osdi_extra_instance_data(entry, gen_inst);

      load(ckt, gen_inst, model, inst, extra_inst_data, is_tran, is_init_tran,
           descr);
      eval_flags |= extra_inst_data->eval_flags;
    }
  }
#else
  for (gen_model = inModel; gen_model; gen_model = gen_model->GENnextModel) {
    void *model = osdi_model_data(gen_model);

    for (gen_inst = gen_model->GENinstances; gen_inst;
         gen_inst = gen_inst->GENnextInstance) {
      void *inst = osdi_instance_data(entry, gen_inst);

      OsdiExtraInstData *extra_inst_data =
          osdi_extra_instance_data(entry, gen_inst);

      /* Deferred-commit: feed this eval the previous-accepted persistent
       * state so predictor/Newton iterates can't corrupt the model's history. */
      osdi_persist_restore(entry, gen_inst, inst, extra_inst_data);

      eval(descr, gen_inst, inst, extra_inst_data, model, &sim_info);

      /* init small signal analysis does not require loading values into
       * matrix/rhs*/
      if (!is_init_smsig) {
        load(ckt, gen_inst, model, inst, extra_inst_data, is_tran, is_init_tran,
             descr);
        eval_flags |= extra_inst_data->eval_flags;
      }
    }
  }
#endif

  /* call to $fatal in Verilog-A abort simulation!*/
  if (eval_flags & EVAL_RET_FLAG_FATAL) {
    return E_PANIC;
  }

  /* EVAL_RET_FLAG_LIM: model applied Verilog-A $limit to damp a node voltage
   * (e.g. NQS internal nodes) during this N-R step.  $limit is the designed
   * mechanism for NQS convergence — the step is internally clamped to a safe
   * range, but the solution has not yet converged.  Signal non-convergence so
   * the N-R keeps iterating rather than accepting the under-damped iterate. */
  if (eval_flags & EVAL_RET_FLAG_LIM) {
    ckt->CKTtroubleElt = gen_inst;
    ckt->CKTnoncon++;
  }

  /* Axis-3 (OSDI v0.5): a model has raised REJECT_STEP, signalling that
   * its linearization is invalid at the current operating point (predicted
   * voltage outside model validity, internal-node blow-up, ...).  NIiter
   * will observe CKTosdiStepReject after CKTload returns and bail with
   * E_ITERLIM so dctran cuts CKTdelta. */
  if ((eval_flags & EVAL_RET_FLAG_REJECT_STEP) && !ckt->CKTosdiStepRejectOff) {
    ckt->CKTtroubleElt = gen_inst;
    ckt->CKTosdiStepReject = 1;
    ckt->CKTnoncon++;
  }

  if (eval_flags & EVAL_RET_FLAG_STOP) {
    return E_PAUSE;
  }

  return OK;
}
