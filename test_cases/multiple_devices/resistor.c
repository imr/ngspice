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

// number of nodes and definitions of node ids for nicer syntax in this file
// note: order should be same as "nodes" list defined later
#define NUM_NODES 3
#define P 0
#define M 1

// number of matrix entries and definitions for Jacobian entries for nicer
// syntax in this file
#define NUM_MATRIX 4
#define P_P 0
#define P_M 1
#define M_P 2
#define M_M 3

// The model structure for the diode
typedef struct ResistorModel
{
  double R;
  bool R_given;
} ResistorModel;

// The instace structure for the diode
typedef struct ResistorInstance
{
  double temperature;
  double rhs_resist[NUM_NODES];
  double rhs_react[NUM_NODES];
  double jacobian_resist[NUM_MATRIX];
  double jacobian_react[NUM_MATRIX];
  double *jacobian_ptr_resist[NUM_MATRIX];
  double *jacobian_ptr_react[NUM_MATRIX];
  uint32_t node_off[NUM_NODES];
} ResistorInstance;

// implementation of the access function as defined by the OSDI spec
void *osdi_access(void *inst_, void *model_, uint32_t id, uint32_t flags)
{
  ResistorModel *model = (ResistorModel *)model_;
  ResistorInstance *inst = (ResistorInstance *)inst_;

  bool *given;
  void *value;

  switch (id) // id of params defined in param_opvar array
  {
  case 0:
    value = (void *)&model->R;
    given = &model->R_given;
    break;
  default:
    return NULL;
  }

  if (flags & ACCESS_FLAG_SET)
  {
    *given = true;
  }

  return value;
}

// implementation of the setup_model function as defined in the OSDI spec
OsdiInitInfo setup_model(void *_handle, void *model_)
{
  ResistorModel *model = (ResistorModel *)model_;

  // set parameters and check bounds
  if (!model->R_given)
  {
    model->R = 1;
  }
  return (OsdiInitInfo){.flags = 0, .num_errors = 0, .errors = NULL};
}

// implementation of the setup_instace function as defined in the OSDI spec
OsdiInitInfo setup_instance(void *_handle, void *inst_, void *model_,
                            double temperature, uint32_t _num_terminals)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;
  ResistorModel *model = (ResistorModel *)model_;

  inst->temperature = temperature;
  return (OsdiInitInfo){.flags = 0, .num_errors = 0, .errors = NULL};
}

// implementation of the eval function as defined in the OSDI spec
uint32_t eval(void *handle, void *inst_, void *model_, uint32_t flags,
              double *prev_solve, OsdiSimParas *sim_params)
{
  ResistorModel *model = (ResistorModel *)model_;
  ResistorInstance *inst = (ResistorInstance *)inst_;

  // get voltages
  double vp = prev_solve[inst->node_off[P]];
  double vm = prev_solve[inst->node_off[M]];

  double vpm = vp - vm;

  double ir = vpm / model->R;
  double g = 1 / model->R;

  ////////////////
  // write rhs
  ////////////////

  if (flags & CALC_RESIST_RESIDUAL)
  {
    // write resist rhs
    inst->rhs_resist[P] = ir;
    inst->rhs_resist[M] = -ir;
  }

  //////////////////
  // write Jacobian
  //////////////////

  if (flags & CALC_RESIST_JACOBIAN)
  {
    // stamp resistor
    inst->jacobian_resist[P_P] = g;
    inst->jacobian_resist[P_M] = -g;
    inst->jacobian_resist[M_P] = -g;
    inst->jacobian_resist[M_M] = g;
  }

  return 0;
}

// TODO implementation of the load_noise function as defined in the OSDI spec
void load_noise(void *inst, void *model, double freq, double *noise_dens,
                double *ln_noise_dens)
{
  // TODO add noise to example
}

#define LOAD_RHS_RESIST(name) \
  dst[inst->node_off[name]] += inst->rhs_resist[name];

// implementation of the load_rhs_resist function as defined in the OSDI spec
void load_residual_resist(void *inst_, double *dst)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;

  LOAD_RHS_RESIST(P)
  LOAD_RHS_RESIST(M)
}

#define LOAD_RHS_REACT(name) dst[inst->node_off[name]] += inst->rhs_react[name];

// implementation of the load_rhs_react function as defined in the OSDI spec
void load_residual_react(void *inst_, double *dst)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;

  LOAD_RHS_REACT(P)
  LOAD_RHS_REACT(M)
}

#define LOAD_MATRIX_RESIST(name) \
  *inst->jacobian_ptr_resist[name] += inst->jacobian_resist[name];

// implementation of the load_matrix_resist function as defined in the OSDI spec
void load_jacobian_resist(void *inst_)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;
  LOAD_MATRIX_RESIST(P_P)
  LOAD_MATRIX_RESIST(P_M)
  LOAD_MATRIX_RESIST(M_P)
  LOAD_MATRIX_RESIST(M_M)
}

#define LOAD_MATRIX_REACT(name) \
  *inst->jacobian_ptr_react[name] += inst->jacobian_react[name] * alpha;

// implementation of the load_matrix_react function as defined in the OSDI spec
void load_jacobian_react(void *inst_, double alpha)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;
  LOAD_MATRIX_REACT(P_P)
  LOAD_MATRIX_REACT(M_M)
  LOAD_MATRIX_REACT(P_M)
  LOAD_MATRIX_REACT(M_P)
}

#define LOAD_MATRIX_TRAN(name) \
  *inst->jacobian_ptr_resist[name] += inst->jacobian_react[name] * alpha;

// implementation of the load_matrix_tran function as defined in the OSDI spec
void load_jacobian_tran(void *inst_, double alpha)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;

  // set dc stamps
  load_jacobian_resist(inst_);

  // add reactive contributions
  LOAD_MATRIX_TRAN(P_P)
  LOAD_MATRIX_TRAN(M_M)
  LOAD_MATRIX_TRAN(M_P)
  LOAD_MATRIX_TRAN(M_M)
}

// implementation of the load_spice_rhs_dc function as defined in the OSDI spec
void load_spice_rhs_dc(void *inst_, double *dst, double *prev_solve)
{
  ResistorInstance *inst = (ResistorInstance *)inst_;
  double vp = prev_solve[inst->node_off[P]];
  double vm = prev_solve[inst->node_off[M]];

  dst[inst->node_off[P]] += inst->jacobian_resist[P_M] * vm +
                            inst->jacobian_resist[P_P] * vp -
                            inst->rhs_resist[P];

  dst[inst->node_off[M]] += inst->jacobian_resist[M_P] * vp +
                            inst->jacobian_resist[M_M] * vm -
                            inst->rhs_resist[M];
}

// implementation of the load_spice_rhs_tran function as defined in the OSDI
// spec
void load_spice_rhs_tran(void *inst_, double *dst, double *prev_solve,
                         double alpha)
{

  ResistorInstance *inst = (ResistorInstance *)inst_;
  double vp = prev_solve[inst->node_off[P]];
  double vm = prev_solve[inst->node_off[M]];

  // set DC rhs
  load_spice_rhs_dc(inst_, dst, prev_solve);

  // add contributions due to reactive elements
  dst[inst->node_off[P]] +=
      alpha * (inst->jacobian_react[P_P] * vp +
               inst->jacobian_react[P_M] * vm);

  dst[inst->node_off[M]] += alpha * (inst->jacobian_react[M_M] * vm +
                                     inst->jacobian_react[M_P] * vp);
}

// structure that provides information of all nodes of the model
OsdiNode nodes[NUM_NODES] = {
    {.name = "P", .units = "V", .is_reactive = false},
    {.name = "M", .units = "V", .is_reactive = false},
};

// boolean array that tells which Jacobian entries are constant. Nothing is
// constant with selfheating, though.
bool const_jacobian_entries[NUM_MATRIX] = {};
// these node pairs specify which entries in the Jacobian must be accounted for
OsdiNodePair jacobian_entries[NUM_MATRIX] = {
    {P, P},
    {P, M},
    {M, P},
    {M, M},
};

#define NUM_PARAMS 1
// the model parameters as defined in Verilog-A, bounds and default values are
// stored elsewhere as they may depend on model parameters etc.
OsdiParamOpvar params[NUM_PARAMS] = {
    {
        .name = (char *[]){"R"},
        .num_alias = 0,
        .description = "Resistance",
        .units = "Ohm",
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
    .name = "resistor_va",

    // nodes
    .num_nodes = NUM_NODES,
    .num_terminals = 2,
    .nodes = (OsdiNode *)&nodes,

    // matrix entries
    .num_jacobian_entries = NUM_MATRIX,
    .jacobian_entries = (OsdiNodePair *)&jacobian_entries,
    .const_jacobian_entries = (bool *)&const_jacobian_entries,

    // memory
    .instance_size = sizeof(ResistorInstance),
    .model_size = sizeof(ResistorModel),
    .residual_resist_offset = offsetof(ResistorInstance, rhs_resist),
    .residual_react_offset = offsetof(ResistorInstance, rhs_react),
    .node_mapping_offset = offsetof(ResistorInstance, node_off),
    .jacobian_resist_offset = offsetof(ResistorInstance, jacobian_resist),
    .jacobian_react_offset = offsetof(ResistorInstance, jacobian_react),
    .jacobian_ptr_resist_offset = offsetof(ResistorInstance, jacobian_ptr_resist),
    .jacobian_ptr_react_offset = offsetof(ResistorInstance, jacobian_ptr_react),

    // TODO add node collapsing to example
    // node collapsing
    .num_collapsible = 0,
    .collapsible = NULL,
    .is_collapsible_offset = 0,

    // noise
    .noise_sources = NULL,
    .num_noise_src = 0,

    // parameters and op vars
    .num_params = NUM_PARAMS,
    .num_instance_params = 0,
    .num_opvars = 0,
    .param_opvar = (OsdiParamOpvar *)&params,

    // setup
    .access = &osdi_access,
    .setup_model = &setup_model,
    .setup_instance = &setup_instance,
    .eval = &eval,
    .load_noise = &load_noise,
    .load_residual_resist = &load_residual_resist,
    .load_residual_react = &load_residual_react,
    .load_spice_rhs_dc = &load_spice_rhs_dc,
    .load_spice_rhs_tran = &load_spice_rhs_tran,
    .load_jacobian_resist = &load_jacobian_resist,
    .load_jacobian_react = &load_jacobian_react,
    .load_jacobian_tran = &load_jacobian_tran,
}};
