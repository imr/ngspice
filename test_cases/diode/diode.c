/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 *
 * This is an exemplary implementation of the OSDI interface for the Verilog-A
 * model specified in diode.va. In the future, the OpenVAF compiler shall
 * generate an comparable object file. Primary purpose of this is example to
 * have a concrete example for the OSDI interface, OpenVAF will generate a more
 * optimized implementation.
 *
 */

#include "osdi.h"
#include "string.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// public interface
extern uint32_t OSDI_VERSION_MAJOR;
extern uint32_t OSDI_VERSION_MINOR;
extern uint32_t OSDI_NUM_DESCRIPTORS;
extern OsdiDescriptor OSDI_DESCRIPTORS[1];
extern OsdiLimFunction OSDI_LIM_TABLE[1];
extern uint32_t OSDI_LIM_TABLE_LEN;

#define sqrt2 1.4142135623730950488016887242097

#define IGNORE(x) (void)x

// number of nodes and definitions of node ids for nicer syntax in this file
// note: order should be same as "nodes" list defined later
#define NUM_NODES 4
#define A 0
#define C 1
#define TNODE 2
#define CI 3

#define NUM_COLLAPSIBLE 2

// number of matrix entries and definitions for Jacobian entries for nicer
// syntax in this file
#define NUM_MATRIX 14
#define CI_CI 0
#define CI_C 1
#define C_CI 2
#define C_C 3
#define A_A 4
#define A_CI 5
#define CI_A 6
#define A_TNODE 7
#define C_TNODE 8
#define CI_TNODE 9
#define TNODE_TNODE 10
#define TNODE_A 11
#define TNODE_C 12
#define TNODE_CI 13

// The model structure for the diode
typedef struct DiodeModel {
  double Rs;
  bool Rs_given;
  double Is;
  bool Is_given;
  double zetars;
  bool zetars_given;
  double N;
  bool N_given;
  double Cj0;
  bool Cj0_given;
  double Vj;
  bool Vj_given;
  double M;
  bool M_given;
  double Rth;
  bool Rth_given;
  double zetarth;
  bool zetarth_given;
  double zetais;
  bool zetais_given;
  double Tnom;
  bool Tnom_given;
  double mfactor; // multiplication factor for parallel devices
  bool mfactor_given;
  // InitError errors[MAX_ERROR_NUM],
} DiodeModel;

// The instace structure for the diode
typedef struct DiodeInstace {
  double mfactor; // multiplication factor for parallel devices
  bool mfactor_given;
  double temperature;
  double residual_resist[NUM_NODES];
  double lim_rhs_resist_A;
  double lim_rhs_resist_CI;
  double lim_rhs_react_A;
  double lim_rhs_react_CI;
  double residual_react_A;
  double residual_react_CI;
  double jacobian_resist[NUM_MATRIX];
  double jacobian_react[NUM_MATRIX];
  bool collapsed[NUM_COLLAPSIBLE];
  double *jacobian_ptr_resist[NUM_MATRIX];
  double *jacobian_ptr_react[NUM_MATRIX];
  uint32_t node_off[NUM_NODES];
  uint32_t state_idx;
} DiodeInstace;

#define EXP_LIM 80.0

static double limexp(double x) {
  if (x < EXP_LIM) {
    return exp(x);
  } else {
    return exp(EXP_LIM) * (x + 1 - EXP_LIM);
  }
}

static double dlimexp(double x) {
  if (x < EXP_LIM) {
    return exp(x);
  } else {
    return exp(EXP_LIM);
  }
}

// implementation of the access function as defined by the OSDI spec
static void *osdi_access(void *inst_, void *model_, uint32_t id,
                         uint32_t flags) {
  DiodeModel *model = (DiodeModel *)model_;
  DiodeInstace *inst = (DiodeInstace *)inst_;

  bool *given;
  void *value;

  switch (id) // id of params defined in param_opvar array
  {
  case 0:
    if (flags & ACCESS_FLAG_INSTANCE) {
      value = (void *)&inst->mfactor;
      given = &inst->mfactor_given;
    } else {
      value = (void *)&model->mfactor;
      given = &model->mfactor_given;
    }
    break;
  case 1:
    value = (void *)&model->Rs;
    given = &model->Rs_given;
    break;
  case 2:
    value = (void *)&model->Is;
    given = &model->Is_given;
    break;
  case 3:
    value = (void *)&model->zetars;
    given = &model->zetars_given;
    break;
  case 4:
    value = (void *)&model->N;
    given = &model->N_given;
    break;
  case 5:
    value = (void *)&model->Cj0;
    given = &model->Cj0_given;
    break;
  case 6:
    value = (void *)&model->Vj;
    given = &model->Vj_given;
    break;
  case 7:
    value = (void *)&model->M;
    given = &model->M_given;
    break;
  case 8:
    value = &model->Rth;
    given = &model->Rth_given;
    break;
  case 9:
    value = (void *)&model->zetarth;
    given = &model->zetarth_given;
    break;
  case 10:
    value = (void *)&model->zetais;
    given = &model->zetais_given;
    break;
  case 11:
    value = (void *)&model->Tnom;
    given = &model->Tnom_given;
    break;
  default:
    return NULL;
  }

  if (flags & ACCESS_FLAG_SET) {
    *given = true;
  }

  return value;
}

// implementation of the setup_model function as defined in the OSDI spec
static void setup_model(void *handle, void *model_, OsdiSimParas *sim_params,
                        OsdiInitInfo *res) {
  DiodeModel *model = (DiodeModel *)model_;

  IGNORE(handle);
  IGNORE(sim_params);

  // set parameters and check bounds
  if (!model->mfactor_given) {
    model->mfactor = 1.0;
  }
  if (!model->Rs_given) {
    model->Rs = 1e-9;
  }
  if (!model->Is_given) {
    model->Is = 1e-14;
  }
  if (!model->zetars_given) {
    model->zetars = 0;
  }
  if (!model->N_given) {
    model->N = 1;
  }
  if (!model->Cj0_given) {
    model->Cj0 = 0;
  }
  if (!model->Vj_given) {
    model->Vj = 1.0;
  }
  if (!model->M_given) {
    model->M = 0.5;
  }
  if (!model->Rth_given) {
    model->Rth = 0;
  }
  if (!model->zetarth_given) {
    model->zetarth = 0;
  }
  if (!model->zetais_given) {
    model->zetais = 0;
  }
  if (!model->Tnom_given) {
    model->Tnom = 300;
  }

  *res = (OsdiInitInfo){.flags = 0, .num_errors = 0, .errors = NULL};
}

// implementation of the setup_instace function as defined in the OSDI spec
static void setup_instance(void *handle, void *inst_, void *model_,
                           double temperature, uint32_t num_terminals,
                           OsdiSimParas *sim_params, OsdiInitInfo *res) {

  IGNORE(handle);
  IGNORE(num_terminals);
  IGNORE(sim_params);

  DiodeInstace *inst = (DiodeInstace *)inst_;
  DiodeModel *model = (DiodeModel *)model_;

  // Here the logic for node collapsing ist implemented. The indices in this
  // list must adhere to the "collapsible" List of node pairs.
  if (model->Rs < 1e-9) { // Rs between Ci C
    inst->collapsed[0] = true;
  }
  if (model->Rth < 1e-9) { // Rs between Ci C
    inst->collapsed[1] = true;
  }

  if (!inst->mfactor_given) {
    if (model->mfactor_given) {
      inst->mfactor = model->mfactor;
    } else {
      inst->mfactor = 1;
    }
  }

  inst->temperature = temperature;
  *res = (OsdiInitInfo){.flags = 0, .num_errors = 0, .errors = NULL};
}

#define CONSTsqrt2 1.4142135623730950488016887242097
typedef double (*pnjlim_t)(bool, bool *, double, double, double, double);

// implementation of the eval function as defined in the OSDI spec
static uint32_t eval(void *handle, void *inst_, void *model_,
                     OsdiSimInfo *info) {
  IGNORE(handle);
  DiodeModel *model = (DiodeModel *)model_;
  DiodeInstace *inst = (DiodeInstace *)inst_;

  // get voltages
  double *prev_solve = info->prev_solve;
  double va = prev_solve[inst->node_off[A]];
  double vc = prev_solve[inst->node_off[C]];
  double vci = prev_solve[inst->node_off[CI]];
  double vdtj = prev_solve[inst->node_off[TNODE]];

  double vcic = vci - vc;
  double vaci = va - vci;

  double gmin = 1e-12;
  for (int i = 0; info->paras.names[i] != NULL; i++) {
    if (strcmp(info->paras.names[i], "gmin") == 0) {
      gmin = info->paras.vals[i];
    }
  }

  uint32_t ret_flags = 0;
  ////////////////////////////////
  // evaluate model equations
  ////////////////////////////////

  // temperature update
  double pk = 1.3806503e-23;
  double pq = 1.602176462e-19;
  double t_dev = inst->temperature + vdtj;
  double tdev_tnom = t_dev / model->Tnom;
  double rs_t = model->Rs * pow(tdev_tnom, model->zetars);
  double rth_t = model->Rth * pow(tdev_tnom, model->zetarth);
  double is_t = model->Is * pow(tdev_tnom, model->zetais);
  double vt = t_dev * pk / pq;

  double delvaci = 0.0;
  if (info->flags & ENABLE_LIM && OSDI_LIM_TABLE[0].func_ptr) {
    double vte = inst->temperature * pk / pq;
    bool icheck = false;
    double vaci_old = info->prev_state[inst->state_idx];
    pnjlim_t pnjlim = OSDI_LIM_TABLE[0].func_ptr;
    double vaci_new = pnjlim(info->flags & INIT_LIM, &icheck, vaci, vaci_old,
                             vte, vte * log(vte / (sqrt2 * model->Is)));
    printf("%g %g\n", vaci, vaci_new);
    delvaci = vaci_new - vaci;
    vaci = vaci_new;
    info->prev_state[inst->state_idx] = vaci;
  } else {
    printf("ok?");
  }

  // derivatives w.r.t. temperature
  double rs_dt = model->zetars * model->Rs *
                 pow(tdev_tnom, model->zetars - 1.0) / model->Tnom;
  double rth_dt = model->zetarth * model->Rth *
                  pow(tdev_tnom, model->zetarth - 1.0) / model->Tnom;
  double is_dt = model->zetais * model->Is *
                 pow(tdev_tnom, model->zetais - 1.0) / model->Tnom;
  double vt_tj = pk / pq;

  // evaluate model equations and calculate all derivatives
  // diode current
  double id = is_t * (limexp(vaci / (model->N * vt)) - 1.0);
  double gd = is_t / vt * dlimexp(vaci / (model->N * vt));
  double gdt = -is_t * dlimexp(vaci / (model->N * vt)) * vaci / model->N / vt /
                   vt * vt_tj +
               1.0 * exp((vaci / (model->N * vt)) - 1.0) * is_dt;

  // resistor
  double irs = 0;
  double g = 0;
  double grt = 0;
  if (!inst->collapsed[0]) {
    irs = vcic / rs_t;
    g = 1.0 / rs_t;
    grt = -irs / rs_t * rs_dt;
  }

  // thermal resistance
  double irth = 0;
  double gt = 0;
  if (!inst->collapsed[1]) {
    irth = vdtj / rth_t;
    gt = 1.0 / rth_t - irth / rth_t * rth_dt;
  }

  // charge
  double vf = model->Vj * (1.0 - pow(3.04, -1.0 / model->M));
  double x = (vf - vaci) / vt;
  double x_vt = -x / vt;
  double x_dtj = x_vt * vt_tj;
  double x_vaci = -1.0 / vt;
  double y = sqrt(x * x + 1.92);
  double y_x = 0.5 / y * 2.0 * x;
  double y_vaci = y_x * x_vaci;
  double y_dtj = y_x * x_dtj;
  double vd = vf - vt * (x + y) / (2.0);
  double vd_x = -vt / 2.0;
  double vd_y = -vt / 2.0;
  double vd_vt = -(x + y) / (2.0);
  double vd_dtj = vd_x * x_dtj + vd_y * y_dtj + vd_vt * vt_tj;
  double vd_vaci = vd_x * x_vaci + vd_y * y_vaci;
  double qd = model->Cj0 * vaci * model->Vj *
              (1.0 - pow(1.0 - vd / model->Vj, 1.0 - model->M)) /
              (1.0 - model->M);
  double qd_vd = model->Cj0 * model->Vj / (1.0 - model->M) * (1.0 - model->M) *
                 pow(1.0 - vd / model->Vj, 1.0 - model->M - 1.0) / model->Vj;
  double qd_dtj = qd_vd * vd_dtj;
  double qd_vaci = qd_vd * vd_vaci;

  // thermal power source = current source
  double ith = id * vaci;
  double ith_vtj = gdt * vaci;
  double ith_vcic = 0;
  double ith_vaci = gd * vaci + id;
  if (!inst->collapsed[0]) {
    ith_vcic = 2.0 * vcic / rs_t;
    ith += pow(vcic, 2.0) / rs_t;
    ith_vtj -= -pow(vcic, 2.0) / rs_t / rs_t * rs_dt;
  }

  id += gmin * vaci;
  gd += gmin;

  double mfactor = inst->mfactor;

  ////////////////
  // write rhs
  ////////////////

  if (info->flags & CALC_RESIST_RESIDUAL) {
    // write resist rhs
    inst->residual_resist[A] = id * mfactor;
    inst->residual_resist[CI] = -id * mfactor + irs * mfactor;
    inst->residual_resist[C] = -irs * mfactor;
    inst->residual_resist[TNODE] = -ith * mfactor + irth * mfactor;
  }

  if (info->flags & CALC_RESIST_LIM_RHS) {
    // write resist rhs
    inst->lim_rhs_resist_A = gd * mfactor * delvaci;
    inst->lim_rhs_resist_CI = -gd * mfactor * delvaci;
  }

  if (info->flags & CALC_REACT_RESIDUAL) {
    // write react rhs
    inst->residual_react_A = qd * mfactor;
    inst->residual_react_CI = -qd * mfactor;
  }

  if (info->flags & CALC_REACT_LIM_RHS) {
    // write resist rhs
    inst->lim_rhs_react_A = qd_vaci * mfactor * delvaci;
    inst->lim_rhs_react_CI = -qd_vaci * mfactor * delvaci;
  }

  //////////////////
  // write Jacobian
  //////////////////

  if (info->flags & CALC_RESIST_JACOBIAN) {
    // stamp diode (current flowing from Ci into A)
    inst->jacobian_resist[A_A] = gd * mfactor;
    inst->jacobian_resist[A_CI] = -gd * mfactor;
    inst->jacobian_resist[CI_A] = -gd * mfactor;
    inst->jacobian_resist[CI_CI] = gd * mfactor;
    // diode thermal
    inst->jacobian_resist[A_TNODE] = gdt * mfactor;
    inst->jacobian_resist[CI_TNODE] = -gdt * mfactor;

    // stamp resistor (current flowing from C into CI)
    inst->jacobian_resist[CI_CI] += g * mfactor;
    inst->jacobian_resist[CI_C] = -g * mfactor;
    inst->jacobian_resist[C_CI] = -g * mfactor;
    inst->jacobian_resist[C_C] = g * mfactor;
    // resistor thermal
    inst->jacobian_resist[CI_TNODE] = grt * mfactor;
    inst->jacobian_resist[C_TNODE] = -grt * mfactor;

    // stamp rth flowing into node dTj
    inst->jacobian_resist[TNODE_TNODE] = gt * mfactor;

    // stamp ith flowing out of T node
    inst->jacobian_resist[TNODE_TNODE] -= ith_vtj * mfactor;
    inst->jacobian_resist[TNODE_CI] = (ith_vcic - ith_vaci) * mfactor;
    inst->jacobian_resist[TNODE_C] = -ith_vcic * mfactor;
    inst->jacobian_resist[TNODE_A] = ith_vaci * mfactor;
  }

  if (info->flags & CALC_REACT_JACOBIAN) {
    // write react matrix
    // stamp Qd between nodes A and Ci depending also on dT
    inst->jacobian_react[A_A] = qd_vaci * mfactor;
    inst->jacobian_react[A_CI] = -qd_vaci * mfactor;
    inst->jacobian_react[CI_A] = -qd_vaci * mfactor;
    inst->jacobian_react[CI_CI] = qd_vaci * mfactor;

    inst->jacobian_react[A_TNODE] = qd_dtj * mfactor;
    inst->jacobian_react[CI_TNODE] = -qd_dtj * mfactor;
  }

  return ret_flags;
}

// TODO implementation of the load_noise function as defined in the OSDI spec
static void load_noise(void *inst, void *model, double freq, double *noise_dens,
                       double *ln_noise_dens) {
  IGNORE(inst);
  IGNORE(model);
  IGNORE(freq);
  IGNORE(noise_dens);
  IGNORE(ln_noise_dens);
  // TODO add noise to example
}

#define LOAD_RESIDUAL_RESIST(name)                                             \
  dst[inst->node_off[name]] += inst->residual_resist[name];

// implementation of the load_rhs_resist function as defined in the OSDI spec
static void load_residual_resist(void *inst_, void *model, double *dst) {
  DiodeInstace *inst = (DiodeInstace *)inst_;

  IGNORE(model);
  LOAD_RESIDUAL_RESIST(A)
  LOAD_RESIDUAL_RESIST(CI)
  LOAD_RESIDUAL_RESIST(C)
  LOAD_RESIDUAL_RESIST(TNODE)
}

// implementation of the load_rhs_react function as defined in the OSDI spec
static void load_residual_react(void *inst_, void *model, double *dst) {
  IGNORE(model);
  DiodeInstace *inst = (DiodeInstace *)inst_;

  dst[inst->node_off[A]] += inst->residual_react_A;
  dst[inst->node_off[CI]] += inst->residual_react_CI;
}

// implementation of the load_lim_rhs_resist function as defined in the OSDI
// spec
static void load_lim_rhs_resist(void *inst_, void *model, double *dst) {
  DiodeInstace *inst = (DiodeInstace *)inst_;

  IGNORE(model);
  dst[inst->node_off[A]] += inst->lim_rhs_resist_A;
  dst[inst->node_off[CI]] += inst->lim_rhs_resist_CI;
}

// implementation of the load_lim_rhs_react function as defined in the OSDI spec
static void load_lim_rhs_react(void *inst_, void *model, double *dst) {
  DiodeInstace *inst = (DiodeInstace *)inst_;

  IGNORE(model);
  dst[inst->node_off[A]] += inst->lim_rhs_react_A;
  dst[inst->node_off[CI]] += inst->lim_rhs_react_CI;
}

#define LOAD_MATRIX_RESIST(name)                                               \
  *inst->jacobian_ptr_resist[name] += inst->jacobian_resist[name];

// implementation of the load_matrix_resist function as defined in the OSDI spec
static void load_jacobian_resist(void *inst_, void *model) {
  IGNORE(model);
  DiodeInstace *inst = (DiodeInstace *)inst_;
  LOAD_MATRIX_RESIST(A_A)
  LOAD_MATRIX_RESIST(A_CI)
  LOAD_MATRIX_RESIST(A_TNODE)

  LOAD_MATRIX_RESIST(CI_A)
  LOAD_MATRIX_RESIST(CI_CI)
  LOAD_MATRIX_RESIST(CI_C)
  LOAD_MATRIX_RESIST(CI_TNODE)

  LOAD_MATRIX_RESIST(C_CI)
  LOAD_MATRIX_RESIST(C_C)
  LOAD_MATRIX_RESIST(C_TNODE)

  LOAD_MATRIX_RESIST(TNODE_TNODE)
  LOAD_MATRIX_RESIST(TNODE_A)
  LOAD_MATRIX_RESIST(TNODE_C)
  LOAD_MATRIX_RESIST(TNODE_CI)
}

#define LOAD_MATRIX_REACT(name)                                                \
  *inst->jacobian_ptr_react[name] += inst->jacobian_react[name] * alpha;

// implementation of the load_matrix_react function as defined in the OSDI spec
static void load_jacobian_react(void *inst_, void *model, double alpha) {
  IGNORE(model);
  DiodeInstace *inst = (DiodeInstace *)inst_;
  LOAD_MATRIX_REACT(A_A)
  LOAD_MATRIX_REACT(A_CI)
  LOAD_MATRIX_REACT(CI_A)
  LOAD_MATRIX_REACT(CI_CI)

  LOAD_MATRIX_REACT(A_TNODE)
  LOAD_MATRIX_REACT(CI_TNODE)
}

#define LOAD_MATRIX_TRAN(name)                                                 \
  *inst->jacobian_ptr_resist[name] += inst->jacobian_react[name] * alpha;

// implementation of the load_matrix_tran function as defined in the OSDI spec
static void load_jacobian_tran(void *inst_, void *model, double alpha) {
  DiodeInstace *inst = (DiodeInstace *)inst_;

  // set dc stamps
  load_jacobian_resist(inst_, model);

  // add reactive contributions
  LOAD_MATRIX_TRAN(A_A)
  LOAD_MATRIX_TRAN(A_CI)
  LOAD_MATRIX_TRAN(CI_A)
  LOAD_MATRIX_TRAN(CI_CI)

  LOAD_MATRIX_TRAN(A_TNODE)
  LOAD_MATRIX_TRAN(CI_TNODE)
}

// implementation of the load_spice_rhs_dc function as defined in the OSDI spec
static void load_spice_rhs_dc(void *inst_, void *model, double *dst,
                              double *prev_solve) {
  IGNORE(model);
  DiodeInstace *inst = (DiodeInstace *)inst_;
  double va = prev_solve[inst->node_off[A]];
  double vci = prev_solve[inst->node_off[CI]];
  double vc = prev_solve[inst->node_off[C]];
  double vdtj = prev_solve[inst->node_off[TNODE]];

  dst[inst->node_off[A]] += inst->jacobian_resist[A_A] * va +
                            inst->jacobian_resist[A_TNODE] * vdtj +
                            inst->jacobian_resist[A_CI] * vci +
                            inst->lim_rhs_resist_A - inst->residual_resist[A];

  dst[inst->node_off[CI]] += inst->jacobian_resist[CI_A] * va +
                             inst->jacobian_resist[CI_TNODE] * vdtj +
                             inst->jacobian_resist[CI_CI] * vci +
                             inst->lim_rhs_resist_CI -
                             inst->residual_resist[CI];

  dst[inst->node_off[C]] +=
      inst->jacobian_resist[C_C] * vc + inst->jacobian_resist[C_CI] * vci +
      inst->jacobian_resist[C_TNODE] * vdtj - inst->residual_resist[C];

  dst[inst->node_off[TNODE]] += inst->jacobian_resist[TNODE_A] * va +
                                inst->jacobian_resist[TNODE_C] * vc +
                                inst->jacobian_resist[TNODE_CI] * vci +
                                inst->jacobian_resist[TNODE_TNODE] * vdtj -
                                inst->residual_resist[TNODE];
}

// implementation of the load_spice_rhs_tran function as defined in the OSDI
// spec
static void load_spice_rhs_tran(void *inst_, void *model, double *dst,
                                double *prev_solve, double alpha) {

  DiodeInstace *inst = (DiodeInstace *)inst_;
  double va = prev_solve[inst->node_off[A]];
  double vci = prev_solve[inst->node_off[CI]];
  double vdtj = prev_solve[inst->node_off[TNODE]];

  // set DC rhs
  load_spice_rhs_dc(inst_, model, dst, prev_solve);

  // add contributions due to reactive elements
  dst[inst->node_off[A]] +=
      alpha *
      (inst->jacobian_react[A_A] * va + inst->jacobian_react[A_CI] * vci +
       inst->jacobian_react[A_TNODE] * vdtj + inst->lim_rhs_react_A);

  dst[inst->node_off[CI]] +=
      alpha *
      (inst->jacobian_react[CI_CI] * vci + inst->jacobian_react[CI_A] * va +
       inst->jacobian_react[CI_TNODE] * vdtj + inst->lim_rhs_react_CI);
}

#define RESIST_RESIDUAL_OFF(NODE)                                              \
  (offsetof(DiodeInstace, residual_resist) + sizeof(uint32_t) * NODE)

// structure that provides information of all nodes of the model
const OsdiNode nodes[NUM_NODES] = {
    {
        .name = "A",
        .units = "V",
        .residual_units = "A",
        .resist_residual_off = RESIST_RESIDUAL_OFF(A),
        .react_residual_off = offsetof(DiodeInstace, residual_react_A),
    },
    {
        .name = "C",
        .units = "V",
        .residual_units = "A",
        .resist_residual_off = RESIST_RESIDUAL_OFF(C),
        .react_residual_off = UINT32_MAX, // no reactive residual

    },
    {
        .name = "dT",
        .units = "K",
        .residual_units = "W",
        .resist_residual_off = RESIST_RESIDUAL_OFF(TNODE),
        .react_residual_off = UINT32_MAX, // no reactive residual
    },
    {
        .name = "CI",
        .units = "V",
        .residual_units = "A",
        .resist_residual_off = RESIST_RESIDUAL_OFF(TNODE),
        .react_residual_off = offsetof(DiodeInstace, residual_react_CI),

    },
};
#define JACOBI_ENTRY(N1, N2)                                                   \
  {                                                                            \
    .nodes = {N1, N2}, .flags = JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT,  \
    .react_ptr_off = offsetof(DiodeInstace, jacobian_ptr_react) +              \
                     sizeof(double *) * N1##_##N2                              \
  }

#define RESIST_JACOBI_ENTRY(N1, N2)                                            \
  {                                                                            \
    .nodes = {N1, N2}, .flags = JACOBIAN_ENTRY_RESIST,                         \
    .react_ptr_off = UINT32_MAX                                                \
  }

// these node pairs specify which entries in the Jacobian must be accounted for
OsdiJacobianEntry jacobian_entries[NUM_MATRIX] = {
    JACOBI_ENTRY(CI, CI),
    RESIST_JACOBI_ENTRY(CI, C),
    RESIST_JACOBI_ENTRY(C, CI),
    RESIST_JACOBI_ENTRY(C, C),
    JACOBI_ENTRY(A, A),
    JACOBI_ENTRY(A, CI),
    JACOBI_ENTRY(CI, A),
    JACOBI_ENTRY(A, TNODE),
    RESIST_JACOBI_ENTRY(C, TNODE),
    JACOBI_ENTRY(CI, TNODE),
    RESIST_JACOBI_ENTRY(TNODE, TNODE),
    RESIST_JACOBI_ENTRY(TNODE, A),
    RESIST_JACOBI_ENTRY(TNODE, C),
    RESIST_JACOBI_ENTRY(TNODE, CI),
};
OsdiNodePair collapsible[NUM_COLLAPSIBLE] = {
    {CI, C},
    {TNODE, NUM_NODES},
};

#define NUM_PARAMS 12
// the model parameters as defined in Verilog-A, bounds and default values are
// stored elsewhere as they may depend on model parameters etc.
OsdiParamOpvar params[NUM_PARAMS] = {
    {
        .name = (char *[]){"$mfactor"},
        .num_alias = 0,
        .description = "Verilog-A multiplication factor for parallel devices",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_INST,
        .len = 0,
    },
    {
        .name = (char *[]){"Rs"},
        .num_alias = 0,
        .description = "Ohmic res",
        .units = "Ohm",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"Is"},
        .num_alias = 0,
        .description = "Saturation current",
        .units = "A",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"zetars"},
        .num_alias = 0,
        .description = "Temperature coefficient of ohmic res",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"N"},
        .num_alias = 0,
        .description = "Emission coefficient",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"Cj0"},
        .num_alias = 0,
        .description = "Junction capacitance",
        .units = "F",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"Vj"},
        .num_alias = 0,
        .description = "Junction potential",
        .units = "V",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"M"},
        .num_alias = 0,
        .description = "Grading coefficient",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"Rth"},
        .num_alias = 0,
        .description = "Thermal resistance",
        .units = "K/W",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"zetarth"},
        .num_alias = 0,
        .description = "Temperature coefficient of thermal res",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"zetais"},
        .num_alias = 0,
        .description = "Temperature coefficient of Is",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
    {
        .name = (char *[]){"Tnom"},
        .num_alias = 0,
        .description = "Reference temperature",
        .units = "",
        .flags = PARA_TY_REAL | PARA_KIND_MODEL,
        .len = 0,
    },
};

// fill exported data
uint32_t OSDI_VERSION_MAJOR = OSDI_VERSION_MAJOR_CURR;
uint32_t OSDI_VERSION_MINOR = OSDI_VERSION_MINOR_CURR;
uint32_t OSDI_NUM_DESCRIPTORS = 1;
// this is the main structure used by simulators, it gives access to all
// information in a model
OsdiDescriptor OSDI_DESCRIPTORS[1] = {{
    // metadata
    .name = "diode_va",

    // nodes
    .num_nodes = NUM_NODES,
    .num_terminals = 3,
    .nodes = (OsdiNode *)&nodes,
    .node_mapping_offset = offsetof(DiodeInstace, node_off),

    // matrix entries
    .num_jacobian_entries = NUM_MATRIX,
    .jacobian_entries = (OsdiJacobianEntry *)&jacobian_entries,
    .jacobian_ptr_resist_offset = offsetof(DiodeInstace, jacobian_ptr_resist),

    // node collapsing
    .num_collapsible = NUM_COLLAPSIBLE,
    .collapsible = collapsible,
    .collapsed_offset = offsetof(DiodeInstace, collapsed),

    // noise
    .noise_sources = NULL,
    .num_noise_src = 0,

    // parameters and op vars
    .num_params = NUM_PARAMS,
    .num_instance_params = 1,
    .num_opvars = 0,
    .param_opvar = (OsdiParamOpvar *)&params,

    // step size bound
    .bound_step_offset = UINT32_MAX,

    .num_states = 1,
    .state_idx_off = offsetof(DiodeInstace, state_idx),

    // memory
    .instance_size = sizeof(DiodeInstace),
    .model_size = sizeof(DiodeModel),

    // setup
    .access = osdi_access,
    .setup_model = setup_model,
    .setup_instance = setup_instance,
    .eval = eval,
    .load_noise = load_noise,
    .load_residual_resist = load_residual_resist,
    .load_residual_react = load_residual_react,
    .load_spice_rhs_dc = load_spice_rhs_dc,
    .load_spice_rhs_tran = load_spice_rhs_tran,
    .load_jacobian_resist = load_jacobian_resist,
    .load_jacobian_react = load_jacobian_react,
    .load_jacobian_tran = load_jacobian_tran,
    .load_limit_rhs_react = load_lim_rhs_react,
    .load_limit_rhs_resist = load_lim_rhs_resist,
}};

OsdiLimFunction OSDI_LIM_TABLE[1] = {{.name = "pnjlim", .num_args = 2}};

uint32_t OSDI_LIM_TABLE_LEN = 1;
