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
      /* Nothing to calculate: just load the matrix entries calculated during
       * operating-point iterations.                                         */
      descr->load_jacobian_resist(inst, model);
      descr->load_jacobian_react(inst, model, ckt->CKTomega);

      /* AC stamping for absdelay slots.
       *
       * For each slot k, the z-equation row has two contributions:
       *   (z_row, y_synth_col): e^{-j*omega*td}   (delayed input)
       *   (z_row, z_col):      -1                  (output)
       *
       * ngspice uses SPICE complex matrix format:
       *   SMPcADD(matrix, row, col, real_part, imag_part)
       * The matrix entries were pre-allocated in OSDIsetup as real pointers;
       * for AC we use the complex variants instead.
       */
      if (entry->experimental && descr->absdelay_count > 0) {
        OsdiExtraInstData *extra = osdi_extra_instance_data(entry, gen_inst);
        double omega = ckt->CKTomega;

        for (uint32_t k = 0; k < descr->absdelay_count; k++) {
          double td = *((double *)(((char *)inst) + descr->absdelays[k].td_offset));
          uint32_t maxdelay_offset = descr->absdelays[k].maxdelay_offset;
          if (maxdelay_offset!=UINT32_MAX) {
            double tdmax = *((double *)(((char *)inst) + maxdelay_offset));
            if (td>tdmax) {
              td = tdmax;
            }
          }
          if (td < 0.0) td = 0.0;

          /* e^{-j*omega*td} = cos(omega*td) - j*sin(omega*td)
           *
           * Complex matrix layout: each entry is two consecutive doubles
           * [real, imag], so ptr+0 = real, ptr+1 = imag.  delay_jac_y/z point
           * to the real slot of that entry: the SMPmakeElt element under the
           * SPARSE solver, or the live CSC_Complex slot under KLU (switched to
           * the complex array by OSDIbindCSCComplex before AC load).         */
          double c =  cos(omega * td);
          double s = -sin(omega * td);

          /* (z_row, y_synth_col): e^{-j*omega*td} */
          *(extra->delay_jac_y[k])     += c;
          *(extra->delay_jac_y[k] + 1) += s;

          /* (z_row, z_col): -1 */
          *(extra->delay_jac_z[k]) += -1.0;
          /* imaginary part stays zero */
        }
      }
    }
  }
  return (OK);
}
