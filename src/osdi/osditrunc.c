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

int OSDItrunc(GENmodel *in_model, CKTcircuit *ckt, double *timestep) {
  OsdiRegistryEntry *entry = osdi_reg_entry_model(in_model);
  const OsdiDescriptor *descr = entry->descriptor;
  uint32_t offset = descr->bound_step_offset;
  bool has_boundstep = offset != UINT32_MAX;
  offset += entry->inst_offset;

  for (GENmodel *model = in_model; model; model = model->GENnextModel) {
    for (GENinstance *inst = model->GENinstances; inst;
         inst = inst->GENnextInstance) {

      if (has_boundstep) {
        double *del = (double *)(((char *)inst) + offset);
        if (*del < *timestep) {
          *timestep = *del;
        }
      }

      int state = inst->GENstate;
      for (uint32_t i = 0; i < descr->num_nodes; i++) {
        if (descr->nodes[i].react_residual_off != UINT32_MAX) {
          CKTterr(state, ckt, timestep);
          state += 2;
        }
      }
    }
  }
  return 0;
}
