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

int OSDIacLoad(GENmodel *inModel, CKTcircuit *ckt) {

  GENmodel *gen_model;
  GENinstance *gen_inst;

  OsdiRegistryEntry *entry = osdi_reg_entry_model(inModel);
  const OsdiDescriptor *descr = entry->descriptor;
  for (gen_model = inModel; gen_model; gen_model = gen_model->GENnextModel) {
    void *model = osdi_model_data(gen_model);

    for (gen_inst = gen_model->GENinstances; gen_inst;
         gen_inst = gen_inst->GENnextInstance) {
      void *inst = osdi_instance_data(entry, gen_inst);
      // nothing to calculate just load the matrix entries calculated during
      // operating point iterations
      descr->load_jacobian_resist(inst, model);
      descr->load_jacobian_react(inst, model, ckt->CKTomega);

      /* OSDI v0.5 (2C AC delay — 1B).  absdelay's exact AC phase: each delay
       * jacobian entry's value V (= ddx(-input), computed at the OP) is stamped
       * as V * e^{-jw*td(site)} into the complex matrix -- V*cos(w*td) into the
       * real part and V*(-sin(w*td)) into the imag part.  This is the
       * transcendental delay term that G+jwC cannot represent; the model's
       * autodiff supplies V (real), the simulator supplies the frequency
       * factor.  Guarded by the ABI-size-checked entry->num_delay_sites so a
       * pre-2C-AC .osdi never reaches the new descriptor tail fields. */
      if (entry->num_delay_sites > 0 && descr->num_delay_jacobian_entries > 0) {
        OsdiExtraInstData *extra = osdi_extra_instance_data(entry, gen_inst);
        if (extra && extra->delay_td_arr) {
          double **jac_resist =
              (double **)(((char *)inst) + descr->jacobian_ptr_resist_offset);
          uint32_t ndel = descr->num_delay_jacobian_entries;
          double *vals = TMALLOC(double, ndel);
          descr->write_jacobian_array_delay(inst, model, vals);
          double w = ckt->CKTomega;
          uint32_t di = 0;
          for (uint32_t i = 0; i < descr->num_jacobian_entries; i++) {
            if (descr->jacobian_entries[i].flags & JACOBIAN_ENTRY_DELAY) {
              uint32_t site = descr->delay_jacobian_sites[di];
              double td = extra->delay_td_arr[site];
              double c = cos(w * td);
              double s = sin(w * td);
              double *elt = jac_resist[i]; /* real part; elt[1] is imag */
              elt[0] += vals[di] * c;
              elt[1] += vals[di] * (-s);
              di++;
            }
          }
          FREE(vals);
        }
      }
    }
  }
  return (OK);
}
