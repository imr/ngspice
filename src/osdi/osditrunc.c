/* 
 * This file is part of the OSDI component of NGSPICE.
 * Copyright© 2022 SemiMod GmbH.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
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
