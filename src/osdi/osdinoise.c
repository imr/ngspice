/*
 * This file is part of the OSDI component of NGSPICE.
 * Copyright© 2023 Pascal Kuthe.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Author: Pascal Kuthe <pascal.kuthe@semimod.de>
 */

#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/ngspice.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"

#include "osdi.h"
#include "osdidefs.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#ifdef RFSPICE
extern CMat *eyem;
extern CMat *zref;
extern CMat *gn;
extern CMat *gninv;
extern CMat *vNoise;
extern CMat *iNoise;
#include "../maths/dense/denseinlines.h"
#endif

static double *noise_dens = NULL;
static double *noise_dens_ln = NULL;
static uint32_t noise_dense_len = 0;

#define nVar(i, j) noise_vals[i * descr->num_noise_src + j]
/*
 * HICUMnoise (mode, operation, firstModel, ckt, data, OnDens)
 *
 *    This routine names and evaluates all of the noise sources
 *    associated with HICUM's.  It starts with the model *firstModel and
 *    traverses all of its insts.  It then proceeds to any other models
 *    on the linked list.  The total output noise density generated by
 *    all of the HICUM's is summed with the variable "OnDens".
 */

int OSDInoise(int mode, int operation, GENmodel *inModel, CKTcircuit *ckt,
              Ndata *data, double *OnDens) {
  GENmodel *gen_model;
  GENinstance *gen_inst;
  uint32_t i, node1, node2;
  double realVal, imagVal, gain, tempOnoise, tempInoise, totalNoise,
      totalNoiseLn, inoise;
  OsdiNoiseSource src;
  uint32_t *node_mapping;
  double *noise_vals;
  NOISEAN *job = (NOISEAN *)ckt->CKTcurJob;

  OsdiRegistryEntry *entry = osdi_reg_entry_model(inModel);
  const OsdiDescriptor *descr = entry->descriptor;

  if (descr->num_noise_src == 0) {
    return OK;
  }

  if (noise_dense_len < descr->num_noise_src) {
    noise_dens = realloc(noise_dens, descr->num_noise_src * sizeof(double));
    noise_dens_ln =
        realloc(noise_dens_ln, descr->num_noise_src * sizeof(double));
  }

  for (gen_model = inModel; gen_model; gen_model = gen_model->GENnextModel) {
    void *model = osdi_model_data(gen_model);

    for (gen_inst = gen_model->GENinstances; gen_inst;
         gen_inst = gen_inst->GENnextInstance) {
      void *inst = osdi_instance_data(entry, gen_inst);
      totalNoise = 0.0;
      noise_vals = osdi_noise_data(entry, gen_inst);

      switch (operation) {

      case N_OPEN:
        /* see if we have to to produce a summary report */
        /* if so, name all the noise generators */
        if (job->NStpsSm != 0) {
          switch (mode) {

          case N_DENS:
            for (i = 0; i < descr->num_noise_src; i++) {
              NOISE_ADD_OUTVAR(ckt, data, "onoise_%s_%s", gen_inst->GENname,
                               descr->noise_sources[i].name);
            }
            // TOTAL noise
            NOISE_ADD_OUTVAR(ckt, data, "onoise_%s%s", gen_inst->GENname, "");
            break;

          case INT_NOIZ:
            for (i = 0; i < descr->num_noise_src; i++) {
              NOISE_ADD_OUTVAR(ckt, data, "onoise_total_%s_%s",
                               gen_inst->GENname, descr->noise_sources[i].name);
              NOISE_ADD_OUTVAR(ckt, data, "inoise_total_%s_%s",
                               gen_inst->GENname, descr->noise_sources[i].name);
            }
            // TOTAL noise
            NOISE_ADD_OUTVAR(ckt, data, "onoise_total_%s%s", gen_inst->GENname,
                             " ");
            NOISE_ADD_OUTVAR(ckt, data, "inoise_total_%s%s", gen_inst->GENname,
                             " ");
            break;
          }
        }
        break;
      case N_CALC:
        switch (mode) {

        case N_DENS:
          descr->load_noise(inst, model, data->freq, noise_dens);
          node_mapping =
              (uint32_t *)(((char *)inst) + descr->node_mapping_offset);
          for (i = 0; i < descr->num_noise_src; i++) {
            src = descr->noise_sources[i];
            node1 = node_mapping[src.nodes.node_1];
            if (src.nodes.node_2 == UINT32_MAX) {
              node2 = 0;
            } else {
              node2 = node_mapping[src.nodes.node_2];
            };
#ifdef RFSPICE
            if (ckt->CKTcurrentAnalysis & DOING_SP) {
              inoise = sqrt(noise_dens[i]);
              // Calculate input equivalent noise current source (we have port
              // impedance attached)
              for (int s = 0; s < ckt->CKTportCount; s++)
                vNoise->d[0][s] =
                    cmultdo(csubco(ckt->CKTadjointRHS->d[s][node1],
                                   ckt->CKTadjointRHS->d[s][node2]),
                            inoise);

              for (int d = 0; d < ckt->CKTportCount; d++) {
                cplx in;
                double yport = 1.0 / zref->d[d][d].re;

                in.re = vNoise->d[0][d].re * yport;
                in.im = vNoise->d[0][d].im * yport;

                for (int s = 0; s < ckt->CKTportCount; s++)
                  caddc(&in, in,
                        cmultco(ckt->CKTYmat->d[d][s], vNoise->d[0][s]));

                iNoise->d[0][d] = in;
              }

              for (int d = 0; d < ckt->CKTportCount; d++)
                for (int s = 0; s < ckt->CKTportCount; s++)
                  ckt->CKTNoiseCYmat->d[d][s] =
                      caddco(ckt->CKTNoiseCYmat->d[d][s],
                             cmultco(iNoise->d[0][d], conju(iNoise->d[0][s])));

              continue;
            }

#endif
            realVal = ckt->CKTrhs[node1] - ckt->CKTrhs[node2];
            imagVal = ckt->CKTirhs[node1] - ckt->CKTirhs[node2];
            gain = (realVal * realVal) + (imagVal * imagVal);
            noise_dens[i] *= gain;
            noise_dens_ln[i] = log(MAX(noise_dens[i], N_MINLOG));
            totalNoise += noise_dens[i];
          }
#ifdef RFSPICE
          if (ckt->CKTcurrentAnalysis & DOING_SP) {
            continue;
            ;
          }
#endif

          *OnDens += totalNoise;
          totalNoiseLn = log(MAX(totalNoise, N_MINLOG));
          if (data->delFreq == 0.0) {
            /* if we haven't done any previous integration, we need to */
            /* initialize our "history" variables                      */

            for (i = 0; i < descr->num_noise_src; i++) {
              nVar(LNLSTDENS, i) = noise_dens_ln[i];
            }

            /* clear out our integration variables if it's the first pass */

            if (data->freq == job->NstartFreq) {
              for (i = 0; i < descr->num_noise_src; i++) {
                nVar(OUTNOIZ, i) = 0.0;
                nVar(INNOIZ, i) = 0.0;
              }
              nVar(OUTNOIZ, descr->num_noise_src) = 0.0;
              nVar(INNOIZ, descr->num_noise_src) = 0.0;
            }
          } else { /* data->delFreq != 0.0 (we have to integrate) */

            /* In order to get the best curve fit, we have to integrate each
             * component separately */

            for (i = 0; i < descr->num_noise_src; i++) {
              tempOnoise = Nintegrate(noise_dens[i], noise_dens_ln[i],
                                      nVar(LNLSTDENS, i), data);
              tempInoise =
                  Nintegrate(noise_dens[i] * data->GainSqInv,
                             noise_dens_ln[i] + data->lnGainInv,
                             nVar(LNLSTDENS, i) + data->lnGainInv, data);
              nVar(LNLSTDENS, i) = noise_dens_ln[i];
              data->outNoiz += tempOnoise;
              data->inNoise += tempInoise;
              if (job->NStpsSm != 0) {
                nVar(OUTNOIZ, i) += tempOnoise;
                nVar(INNOIZ, i) += tempInoise;
                nVar(OUTNOIZ, descr->num_noise_src) += tempOnoise;
                nVar(INNOIZ, descr->num_noise_src) += tempInoise;
              }
            }
          }
          if (data->prtSummary) {
            for (i = 0; i < descr->num_noise_src;
                 i++) { /* print a summary report */
              data->outpVector[data->outNumber++] = noise_dens[i];
            }
            data->outpVector[data->outNumber++] = totalNoise;
          }
          break;
        case INT_NOIZ: /* already calculated, just output */
          if (job->NStpsSm != 0) {
            for (i = 0; i <= descr->num_noise_src; i++) {
              data->outpVector[data->outNumber++] = nVar(OUTNOIZ, i);
              data->outpVector[data->outNumber++] = nVar(INNOIZ, i);
            }
          } /* if */
          break;
        } /* switch (mode) */
        break;

      case N_CLOSE:
        return (OK); /* do nothing, the main calling routine will close */
        break;       /* the plots */
      }
    }
  }
  return (OK);
}
