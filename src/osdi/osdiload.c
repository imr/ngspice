/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#include "ngspice/iferrmsg.h"
#include "ngspice/memory.h"
#include "ngspice/ngspice.h"
#include "ngspice/typedefs.h"

#include "osdi.h"
#include "osdidefs.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_SIM_PARAMS 4
char *sim_params[NUM_SIM_PARAMS + 1] = {"gdev", "gmin", "tnom",
                                        "simulatorVersion", NULL};
char *sim_params_str[1] = {NULL};

extern int OSDIload(GENmodel *inModel, CKTcircuit *ckt) {
  OsdiNgspiceHandle handle;
  GENmodel *gen_model;
  GENinstance *gen_inst;
  double dump;

  bool is_init_smsig = ckt->CKTmode & MODEINITSMSIG;
  bool is_ac = ckt->CKTmode & (MODEAC | MODEINITSMSIG);
  bool is_tran_op = ((ckt->CKTmode & MODETRANOP) && (ckt->CKTmode & MODEUIC));
  bool is_tran = ckt->CKTmode & (MODEDCTRANCURVE | MODETRAN) || is_tran_op;
  bool is_init_tran = ckt->CKTmode & MODEINITTRAN;

  uint32_t flags = CALC_RESIST_JACOBIAN;

  if (is_init_smsig) {
    flags |= CALC_OP;
  } else {
    flags |= CALC_RESIST_RESIDUAL;
  }

  if (is_tran || is_ac || is_tran_op) {
    flags |= CALC_REACT_JACOBIAN;
  }

  if (is_tran || is_tran_op) {
    flags |= CALC_REACT_RESIDUAL;
  }

  if (ckt->CKTmode & MODEACNOISE) {
    flags |= CALC_NOISE;
  }

  int ret = OK;

  /* values returned by $simparam*/
  double simulatorVersion = strtod(PACKAGE_VERSION, NULL);
  double gdev = ckt->CKTgmin;
  double gmin = ((ckt->CKTgmin) > (ckt->CKTdiagGmin)) ? (ckt->CKTgmin)
                                                      : (ckt->CKTdiagGmin);
  double sim_param_vals[NUM_SIM_PARAMS] = {gdev, gmin, ckt->CKTnomTemp,
                                           simulatorVersion};
  OsdiSimParas sim_params_ = {.names = sim_params,
                              .vals = sim_param_vals,
                              .names_str = sim_params_str,
                              .vals_str = NULL};

  OsdiRegistryEntry *entry = osdi_reg_entry_model(inModel);
  const OsdiDescriptor *descr = entry->descriptor;

  for (gen_model = inModel; gen_model; gen_model = gen_model->GENnextModel) {
    void *model = osdi_model_data(gen_model);

    for (gen_inst = gen_model->GENinstances; gen_inst;
         gen_inst = gen_inst->GENnextInstance) {
      void *inst = osdi_instance_data(entry, gen_inst);

      /* hpyothetically this could run in parallel we do not write any shared
       data here*/
      handle = (OsdiNgspiceHandle){.kind = 3, .name = gen_inst->GENname};
      /* TODO initial conditions? */
      uint32_t ret_flags = descr->eval(&handle, inst, model, flags,
                                       ckt->CKTrhsOld, &sim_params_);

      /* call to $fatal in Verilog-A abort!*/
      if (ret_flags & EVAL_RET_FLAG_FATAL) {
        return E_PANIC;
      }

      /* init small signal analysis does not require loading values into
       * matrix/rhs*/
      if (is_init_smsig) {
        continue;
      }

      /* handle calls to $finish, $limit, $stop
       * TODO actually do something with extra_inst_data->finish  and
       * extra_inst_data->limt
       * */
      OsdiExtraInstData *extra_inst_data =
          osdi_extra_instance_data(entry, gen_inst);
      if (ret_flags & EVAL_RET_FLAG_FINISH) {
        extra_inst_data->finish = true;
      }
      if (ret_flags & EVAL_RET_FLAG_LIM) {
        extra_inst_data->lim = true;
      }
      if (ret_flags & EVAL_RET_FLAG_STOP) {
        ret = (E_PAUSE);
      }

      if (is_tran) {
        /* load dc matrix and capacitances (charge derivative multiplied with
         * CKTag[0]) */
        descr->load_jacobian_tran(inst, ckt->CKTag[0]);

        /* load static rhs and dynamic linearized rhs (SUM Vb * dIa/dVb)*/
        descr->load_spice_rhs_tran(inst, ckt->CKTrhs, ckt->CKTrhsOld,
                                   ckt->CKTag[0]);

        uint32_t *node_mapping =
            (uint32_t *)(((char *)inst) + descr->node_mapping_offset);

        double *residual_react =
            (double *)(((char *)inst) + descr->residual_react_offset);

        /* use numeric integration to obtain the remainer of the RHS*/
        int state = gen_inst->GENstate;
        for (uint32_t i = 0; i < descr->num_nodes; i++) {
          if (descr->nodes[i].is_reactive) {
            /* store charges in state vector*/
            ckt->CKTstate0[state] = residual_react[i];
            if (is_init_tran) {
              ckt->CKTstate1[state] = residual_react[i];
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
        descr->load_jacobian_resist(inst);
        /* calculate spice RHS from internal currents and store into global RHS
         */
        descr->load_spice_rhs_dc(inst, ckt->CKTrhs, ckt->CKTrhsOld);
      }
    }
  }
  return ret;
}
