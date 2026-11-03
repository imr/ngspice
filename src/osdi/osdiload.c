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

/* -----------------------------------------------------------------------
 * absdelay transient stamping helpers
 * -----------------------------------------------------------------------
 *
 * History layout: delay_hist[k][i] = V(y_synth) at CKTtimePoints[i].
 * During Newton iterations CKTtimeIndex = ti is fixed; we keep updating
 * hist[k][ti] with the latest CKTrhsOld value so that at convergence
 * hist[k][ti] holds the true accepted value for the next step.
 */

/* Grow delay_hist rows to hold at least new_cap entries. */
static void absdelay_grow_hist(OsdiExtraInstData *extra, uint32_t n_delays,
                               uint32_t new_cap) {
  for (uint32_t k = 0; k < n_delays; k++) {
    extra->delay_hist[k] =
        TREALLOC(double, extra->delay_hist[k], new_cap);
  }
  extra->delay_hist_cap = new_cap;
}

/* Ensure CKTtimePoints is allocated (if no LTRA device is in the circuit
 * optran.c leaves it NULL).  We allocate it ourselves on the first transient
 * call and let optran.c's nextTime: grow it thereafter.                    */
static void absdelay_ensure_timepoints(CKTcircuit *ckt) {
  if (ckt->CKTtimePoints == NULL) {
    uint32_t cap = (ckt->CKTtimeListSize > 0) ? (uint32_t)ckt->CKTtimeListSize : 256;
    ckt->CKTtimePoints = TMALLOC(double, cap);
    ckt->CKTtimeListSize = (int)cap;
    ckt->CKTtimeIndex = 0;
    ckt->CKTtimePoints[0] = 0.0;
  } else if (ckt->CKTtimeIndex < 0) {
    ckt->CKTtimeIndex = 0;
    ckt->CKTtimePoints[0] = 0.0;
  }
}

/*
 * Lookup the delayed value for slot k, and return the Jacobian alpha
 * (sensitivity of delayed_value w.r.t. V_y_current) via *alpha_out.
 *
 * Uses linear interpolation over accepted timepoints.
 * When t_lookup falls between the last accepted time and CKTtime, the
 * interpolation crosses into the current Newton iteration, giving alpha > 0.
 */
static double absdelay_lookup(const OsdiExtraInstData *extra, uint32_t k,
                              double td, const CKTcircuit *ckt,
                              double V_y_old, double *alpha_out) {
  *alpha_out = 0.0;

  int ti = ckt->CKTtimeIndex;
  if (ti < 0 || ckt->CKTtimePoints == NULL) {
    /* No history yet — pass through */
    return V_y_old;
  }

  double t_lookup = ckt->CKTtime - td;

  /* Clamp to the beginning of history */
  if (t_lookup <= ckt->CKTtimePoints[0]) {
    return extra->delay_hist[k][0];
  }

  double t_last_accepted = ckt->CKTtimePoints[ti];

  if (t_lookup >= ckt->CKTtime && ti >= 0) {
    /* delay <= 0: return current value with full Jacobian sensitivity */
    *alpha_out = 1.0;
    return V_y_old;
  }

  if (t_lookup >= t_last_accepted) {
    /* delay is smaller than current timestep: interpolate between last
     * accepted point and CKTtime (current candidate).               */
    double dt_step = ckt->CKTtime - t_last_accepted;
    double alpha = (dt_step > 0.0)
                       ? (t_lookup - t_last_accepted) / dt_step
                       : 1.0;
    *alpha_out = alpha;
    double hist_last = extra->delay_hist[k][ti];
    return hist_last + alpha * (V_y_old - hist_last);
  }

  /* General case: binary search through accepted timepoints [0 .. ti] */
  int lo = 0, hi = ti;
  while (lo + 1 < hi) {
    int mid = (lo + hi) / 2;
    if (ckt->CKTtimePoints[mid] <= t_lookup)
      lo = mid;
    else
      hi = mid;
  }
  double t0 = ckt->CKTtimePoints[lo];
  double t1 = ckt->CKTtimePoints[hi];
  double v0 = extra->delay_hist[k][lo];
  double v1 = extra->delay_hist[k][hi];
  double dt = t1 - t0;
  if (dt <= 0.0)
    return v0;
  double frac = (t_lookup - t0) / dt;
  return v0 + frac * (v1 - v0);
}

/*
 * DC / TRAN-OP pass-through stamp for absdelay slots.
 * In steady-state absdelay reduces to an ideal wire: V(z) = V(y_synth).
 * Without this the z-row has no matrix entries and the solver reports a
 * singular matrix.
 */
static void absdelay_stamp_dc(void *inst, OsdiExtraInstData *extra,
                               const OsdiRegistryEntry *entry,
                               const OsdiDescriptor *descr) {
  NG_IGNORE(entry);

  uint32_t n = descr->absdelay_count;
  uint32_t *node_mapping =
      (uint32_t *)(((char *)inst) + descr->node_mapping_offset);

  for (uint32_t k = 0; k < n; k++) {
    /* V(z) - V(y_synth) = 0  →  jac[z,y]+=1, jac[z,z]+=-1, rhs[z]+=0 */
    *(extra->delay_jac_y[k]) += 1.0;
    *(extra->delay_jac_z[k]) += -1.0;
    NG_IGNORE(node_mapping);
  }
}

/*
 * Stamp residual and Jacobian for all absdelay slots of one instance.
 * Called after the standard OSDI load() for each transient step.
 */
static void absdelay_stamp_tran(CKTcircuit *ckt, GENinstance *gen_inst,
                                void *inst, OsdiExtraInstData *extra,
                                const OsdiRegistryEntry *entry,
                                const OsdiDescriptor *descr,
                                bool is_init_tran) {
  NG_IGNORE(gen_inst);
  NG_IGNORE(entry);

  uint32_t n = descr->absdelay_count;
  if (n == 0)
    return;
  
  uint32_t *node_mapping =
      (uint32_t *)(((char *)inst) + descr->node_mapping_offset);

  /* On the first transient call: allocate CKTtimePoints if needed and
   * initialize the history arrays.                                        */
  if (is_init_tran) {
    absdelay_ensure_timepoints(ckt);
    uint32_t cap = (uint32_t)(ckt->CKTtimeListSize > 0
                                  ? (uint32_t)ckt->CKTtimeListSize
                                  : 256) + 64;
    if (extra->delay_hist_cap < cap) {
      absdelay_grow_hist(extra, n, cap);
    }
    /* Initialize hist[k][0] to 0.0 (V_y at t=0 before the transient begins).
     * CKTtimePoints[0] was set to 0.0 by absdelay_ensure_timepoints.
     * OSDIaccept() will update hist[k][ti] for ti >= 1 as timepoints are
     * accepted.  For the very first timestep, the output is forced to track
     * the input (pass-through IC) so the matrix is non-singular.          */
    for (uint32_t k = 0; k < n; k++) {
      extra->delay_hist[k][0] = 0.0;
      *(extra->delay_jac_y[k]) += 1.0;
      *(extra->delay_jac_z[k]) += -1.0;
    }
    return;
  }

  /* Ensure history capacity matches CKTtimeListSize growth */
  uint32_t needed = (uint32_t)(ckt->CKTtimeListSize) + 64;
  if (extra->delay_hist_cap < needed) {
    absdelay_grow_hist(extra, n, needed);
  }

  int ti = ckt->CKTtimeIndex;
  if (ti < 0 || ckt->CKTtimePoints == NULL)
    return;

  for (uint32_t k = 0; k < n; k++) {
    uint32_t y_mapped = node_mapping[descr->absdelays[k].y_node];
    uint32_t z_mapped = node_mapping[descr->absdelays[k].z_node];

    /* Read td from OSDI instance data */
    double td = *((double *)(((char *)inst) + descr->absdelays[k].td_offset));
    uint32_t maxdelay_offset = descr->absdelays[k].maxdelay_offset;
    if (maxdelay_offset!=UINT32_MAX) {
      double tdmax = *((double *)(((char *)inst) + maxdelay_offset));
      if (td>tdmax) {
        td = tdmax;
      }
    }
    if (td < 0.0)
      td = 0.0;

    double V_y_old = ckt->CKTrhsOld[y_mapped];
    double V_z_old = ckt->CKTrhsOld[z_mapped];

    /* Treat sub-femtosecond delays as zero: stamp as DC pass-through to avoid
     * forcing the timestep below the delay value (which would cause timestep-
     * too-small failures).  Real photonic delays are >> 1 fs. */
    if (td < 1e-15) {
      *(extra->delay_jac_y[k]) += 1.0;
      *(extra->delay_jac_z[k]) += -1.0;
      /* RHS: zero — for V(y_synth) - V(z) = 0 the constant term is 0 */
      NG_IGNORE(V_z_old);
      continue;
    }

    double alpha = 0.0;
    double delayed_val = absdelay_lookup(extra, k, td, ckt, V_y_old, &alpha);

    /* Stamp into pre-allocated matrix entries and RHS.
     * z-row equation: delayed_val - V_z = 0
     *   d/dV_y: alpha   (nonzero only when delay < current timestep)
     *   d/dV_z: -1.0
     * RHS contribution: alpha * V_y_old - delayed_val               */
    *(extra->delay_jac_y[k]) += alpha;
    *(extra->delay_jac_z[k]) += -1.0;
    ckt->CKTrhs[z_mapped] += alpha * V_y_old - delayed_val;

    NG_IGNORE(V_z_old);
  }
}

#define NUM_SIM_PARAMS 10
char *sim_params[NUM_SIM_PARAMS + 1] = {
    "iniLim", "gmin", "gdev", "tnom", 
    "simulatorVersion", "sourceScaleFactor", 
    "epsmin", "reltol", "vntol", "abstol", 
    NULL};
char *sim_params_str[1] = {NULL};

double sim_param_vals[NUM_SIM_PARAMS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* values returned by $simparam*/
OsdiSimParas get_simparams(const CKTcircuit *ckt) {
  double simulatorVersion = strtod(PACKAGE_VERSION, NULL);
  double gdev = ckt->CKTgmin;
  double sourceScaleFactor = ckt->CKTsrcFact;
  double gmin = ((ckt->CKTgmin) > (ckt->CKTdiagGmin)) ? (ckt->CKTgmin)
                                                      : (ckt->CKTdiagGmin);
  double initializeLimiting = (ckt->CKTmode & MODEINITJCT) ? 1 : 0;

  double sim_param_vals_[NUM_SIM_PARAMS] = {
      // Verilog-A tnom is in degrees Celsius
      initializeLimiting, gmin, gdev, ckt->CKTnomTemp-CONSTCtoK, simulatorVersion, sourceScaleFactor, 
      ckt->CKTepsmin, ckt->CKTreltol, ckt->CKTvoltTol, ckt->CKTabstol };
  memcpy(&sim_param_vals, &sim_param_vals_, sizeof(double) * NUM_SIM_PARAMS);
  OsdiSimParas sim_params_ = {.names = sim_params,
                              .vals = (double *)&sim_param_vals,
                              .names_str = sim_params_str,
                              .vals_str = NULL};
  return sim_params_;
}

static void eval(const OsdiDescriptor *descr, const GENinstance *gen_inst,
                 void *inst, OsdiExtraInstData *extra_inst_data,
                 const void *model, const OsdiSimInfo *sim_info) {

  OsdiNgspiceHandle handle =
      (OsdiNgspiceHandle){.kind = 3, .name = gen_inst->GENname};
  /* TODO initial conditions? */
  extra_inst_data->eval_flags = descr->eval(&handle, inst, model, sim_info);
}

static void load(CKTcircuit *ckt, const GENinstance *gen_inst, void *model,
                 void *inst, OsdiExtraInstData *extra_inst_data, bool is_tran,
                 bool is_init_tran, const OsdiDescriptor *descr) {

  NG_IGNORE(extra_inst_data);

  double dump;
  if (is_tran) {
    /* load dc matrix and capacitances (charge derivative multiplied with
     * CKTag[0]) */
    descr->load_jacobian_tran(inst, model, ckt->CKTag[0]);

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
        ckt->CKTrhs[node_mapping[i]] -= ckt->CKTstate0[state + 1];

        if (is_init_tran) {
          ckt->CKTstate1[state + 1] = ckt->CKTstate0[state + 1];
        }

        state += 2;
      }
    }
  } else {
    /* copy internal derivatives into global matrix */
    descr->load_jacobian_resist(inst, model);

    /* calculate spice RHS from internal currents and store into global RHS
     */
    descr->load_spice_rhs_dc(inst, model, ckt->CKTrhs, ckt->CKTrhsOld);
  }
}

extern int OSDIload(GENmodel *inModel, CKTcircuit *ckt) {
  GENmodel *gen_model;
  GENinstance *gen_inst;
  /* first instance that asked for limiting, for the non-convergence report */
  GENinstance *trouble_inst = NULL;

  bool is_init_smsig = ckt->CKTmode & MODEINITSMSIG;
  bool is_dc = ckt->CKTmode & (MODEDCOP | MODEDCTRANCURVE);
  bool is_ac = ckt->CKTmode & (MODEAC | MODEINITSMSIG);
  bool is_tran = ckt->CKTmode & (MODETRAN);
  bool is_tran_op = ckt->CKTmode & (MODETRANOP);
  bool is_init_tran = ckt->CKTmode & MODEINITTRAN;
  bool is_init_junc = ckt->CKTmode & MODEINITJCT;

  OsdiSimInfo sim_info = {
      .paras = get_simparams(ckt),
      .abstime = is_tran ? ckt->CKTtime : 0.0,
      .prev_solve = ckt->CKTrhsOld,
      .prev_state = ckt->CKTstates[0],
      .next_state = ckt->CKTstates[0],
      .flags = CALC_RESIST_JACOBIAN,
  };

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
  }

  if (is_tran_op) {
    sim_info.flags |= ANALYSIS_TRAN;
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
      if (is_tran) {
        if (entry->experimental)
          absdelay_stamp_tran(ckt, gen_inst, inst, extra_inst_data, entry,
                            descr, is_init_tran);
      } else if (entry->experimental && descr->absdelay_count > 0) {
        absdelay_stamp_dc(inst, extra_inst_data, entry, descr);
      }
      if (!trouble_inst &&
          (extra_inst_data->eval_flags & EVAL_RET_FLAG_LIM)) {
        trouble_inst = gen_inst;
      }
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
      eval(descr, gen_inst, inst, extra_inst_data, model, &sim_info);

      /* init small signal analysis does not require loading values into
       * matrix/rhs*/
      if (!is_init_smsig) {
        load(ckt, gen_inst, model, inst, extra_inst_data, is_tran, is_init_tran,
             descr);
        if (is_tran) {
          if (entry->experimental)
            absdelay_stamp_tran(ckt, gen_inst, inst, extra_inst_data, entry,
                                descr, is_init_tran);
        } else if (entry->experimental && descr->absdelay_count > 0) {
          absdelay_stamp_dc(inst, extra_inst_data, entry, descr);
        }
        if (!trouble_inst &&
            (extra_inst_data->eval_flags & EVAL_RET_FLAG_LIM)) {
          trouble_inst = gen_inst;
        }
        eval_flags |= extra_inst_data->eval_flags;
      }
    }
  }
#endif

  /* call to $fatal in Verilog-A abort simulation!*/
  if (eval_flags & EVAL_RET_FLAG_FATAL) {
    return E_PANIC;
  }

  if (eval_flags & EVAL_RET_FLAG_LIM) {
    ckt->CKTnoncon++;
    /* gen_inst is NULL here, both loops have run to the end */
    ckt->CKTtroubleElt = trouble_inst;
  }

  if (eval_flags & EVAL_RET_FLAG_STOP) {
    return E_PAUSE;
  }

  return OK;
}
