/*
 * This file is part of the OSDI component of NGSPICE.
 * Copyright© 2022 SemiMod GmbH.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/*
 * OSDIaccept — called by CKTaccept() after each accepted transient timepoint.
 *
 * For absdelay slots we commit the converged V(y_synth) value into the
 * waveform history at the current CKTtimeIndex.  CKTtimePoints[ti] has
 * already been set to the accepted time by optran.c before CKTaccept() is
 * called, so the pairing (CKTtimePoints[ti], delay_hist[k][ti]) is correct.
 */

#include "ngspice/iferrmsg.h"
#include "ngspice/memory.h"
#include "ngspice/ngspice.h"
#include "ngspice/typedefs.h"

#include "osdi.h"
#include "osdidefs.h"

#include <stdint.h>
#include <string.h>

int OSDIaccept(CKTcircuit *ckt, GENmodel *inModel) {
  OsdiRegistryEntry *entry = osdi_reg_entry_model(inModel);
  const OsdiDescriptor *descr = entry->descriptor;

  if (!entry->experimental || descr->absdelay_count == 0)
    return OK;

  bool is_tran = (bool)(ckt->CKTmode & MODETRAN);
  if (!is_tran)
    return OK;

  /* CKTtimePoints and CKTtimeIndex are populated by absdelay_stamp_tran
   * during the MODEINITTRAN Newton call.  If still NULL the transient hasn't
   * started yet (e.g., MODETRANOP DC OP call).                             */
  if (ckt->CKTtimePoints == NULL || ckt->CKTtimeIndex < 0)
    return OK;

  uint32_t n = descr->absdelay_count;
  int ti = ckt->CKTtimeIndex;

  for (GENmodel *gen_model = inModel; gen_model;
       gen_model = gen_model->GENnextModel) {
    for (GENinstance *gen_inst = gen_model->GENinstances; gen_inst;
         gen_inst = gen_inst->GENnextInstance) {
      void *inst = osdi_instance_data(entry, gen_inst);
      OsdiExtraInstData *extra = osdi_extra_instance_data(entry, gen_inst);

      if (!extra->delay_hist)
        continue;

      /* Grow history arrays if needed (optran.c may have grown CKTtimePoints) */
      uint32_t needed = (uint32_t)(ckt->CKTtimeListSize) + 64;
      if (extra->delay_hist_cap < needed) {
        for (uint32_t k = 0; k < n; k++) {
          extra->delay_hist[k] =
              TREALLOC(double, extra->delay_hist[k], needed);
        }
        extra->delay_hist_cap = needed;
      }

      if ((uint32_t)ti >= extra->delay_hist_cap)
        continue;

      uint32_t *node_mapping =
          (uint32_t *)(((char *)inst) + descr->node_mapping_offset);

      for (uint32_t k = 0; k < n; k++) {
        uint32_t y_mapped = node_mapping[descr->absdelays[k].y_node];
        /* Store the CONVERGED V(y_synth) at the just-accepted timepoint. */
        extra->delay_hist[k][ti] = ckt->CKTrhsOld[y_mapped];
      }
    }
  }

  return OK;
}
