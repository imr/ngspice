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

#include "ngspice/cktdefs.h"
#include "osdidefs.h"
#include <stdio.h>

int OSDItrunc(GENmodel *in_model, CKTcircuit *ckt, double *timestep) {
  OsdiRegistryEntry *entry = osdi_reg_entry_model(in_model);
  const OsdiDescriptor *descr = entry->descriptor;
  uint32_t offset = descr->bound_step_offset;
  bool has_boundstep = offset != UINT32_MAX;
  offset += entry->inst_offset;

  /* OSDI v0.5 (S3b) — clamp the proposed timestep against any
   * pending scheduled events on instances of this model.  Each
   * instance has a sorted per-instance queue of scheduled events;
   * the earliest one bounds dt so the simulator lands exactly on
   * the event time and the model's `@(timer)` / `@(cross)` body
   * fires with at_scheduled_event = true. */
  double t_now = ckt->CKTtime;
  bool model_has_events = descr->num_event_slots > 0;

  for (GENmodel *model = in_model; model; model = model->GENnextModel) {
    for (GENinstance *inst = model->GENinstances; inst;
         inst = inst->GENnextInstance) {

      if (has_boundstep) {
        double *del = (double *)(((char *)inst) + offset);
        if (*del < *timestep) {
          *timestep = *del;
        }
      }

      if (model_has_events) {
        OsdiExtraInstData *extra =
            osdi_extra_instance_data(entry, inst);
        if (extra && extra->scheduled_count > 0) {
          double t_event = extra->scheduled_events[0].at_time;
          double dt_to_event = t_event - t_now;
          if (dt_to_event > 0.0 && dt_to_event < *timestep) {
            *timestep = dt_to_event;
          }
        }
      }

      int state = inst->GENstate + (int)descr->num_states;
      for (uint32_t i = 0; i < descr->num_nodes; i++) {
        if (descr->nodes[i].react_residual_off != UINT32_MAX) {
          CKTterr(state, ckt, timestep);
          state += 2;
        }
      }
    }
  }

  /* CKTterr is backward-looking: it sees low curvature in the charges just
   * before a switching transition and allows the timestep to grow, producing
   * a large step that spans the entire transition.  Limit growth to 1.2×
   * the most recently accepted step so that fast edges are resolved
   * gradually.
   *
   * Was 2.0× originally, tightened to 1.5× for pinb/net_7 oscillation
   * (commit 143a0805f) and now to 1.2× for TSMC22 ULP driver_lv_2v5_tb
   * to address residual pinb failure at t ≈ 1.369 µs.  At 1.5×, dt
   * could grow from ~10 ps post-breakpoint to ~5 ns (the user's max
   * step) in ~15 accepted steps; once Newton had to land a 0.5 V swing
   * at the next switching edge from a 5 ns extrapolation, the
   * linearization was too coarse to track within itl4.  1.2× still
   * grows by ~6× over 10 accepted steps (1.2^10), so a quiet 6 ns
   * interval gets ~12 steps to grow from 10 ps to ~300 ps — small
   * enough that the next sharp edge is easily resolved. */
  if (ckt->CKTdeltaOld[0] > 0.0) {
    double max_step = ckt->CKTdeltaOld[0] * 1.2;
    if (*timestep > max_step) {
      *timestep = max_step;
    }
  }

  return 0;
}
