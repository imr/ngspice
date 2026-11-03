#pragma once
#include "osdi.h"

typedef struct OsdiDescriptor03 {
  char *name;

  uint32_t num_nodes;
  uint32_t num_terminals;
  OsdiNode *nodes;

  uint32_t num_jacobian_entries;
  OsdiJacobianEntry *jacobian_entries;

  uint32_t num_collapsible;
  OsdiNodePair *collapsible;
  uint32_t collapsed_offset;

  OsdiNoiseSource *noise_sources;
  uint32_t num_noise_src;

  uint32_t num_params;
  uint32_t num_instance_params;
  uint32_t num_opvars;
  OsdiParamOpvar *param_opvar;

  uint32_t node_mapping_offset;
  uint32_t jacobian_ptr_resist_offset;

  uint32_t num_states;
  uint32_t state_idx_off;

  uint32_t bound_step_offset;

  uint32_t instance_size;
  uint32_t model_size;

  void *(*access)(void *inst, void *model, uint32_t id, uint32_t flags);

  void (*setup_model)(void *handle, void *model, OsdiSimParas *sim_params,
                                     OsdiInitInfo *res);
  void (*setup_instance)(void *handle, void *inst, void *model,
                                     double temperature, uint32_t num_terminals,
                                     OsdiSimParas *sim_params, OsdiInitInfo *res);

  uint32_t (*eval)(void *handle, void *inst, const void *model, const OsdiSimInfo *info);
  void (*load_noise)(void *inst, void *model, double freq, double *noise_dens);
  void (*load_residual_resist)(void *inst, void* model, double *dst);
  void (*load_residual_react)(void *inst, void* model, double *dst);
  void (*load_limit_rhs_resist)(void *inst, void* model, double *dst);
  void (*load_limit_rhs_react)(void *inst, void* model, double *dst);
  void (*load_spice_rhs_dc)(void *inst, void* model, double *dst,
                  double* prev_solve);
  void (*load_spice_rhs_tran)(void *inst, void* model, double *dst,
                  double* prev_solve, double alpha);
  void (*load_jacobian_resist)(void *inst, void* model);
  void (*load_jacobian_react)(void *inst, void* model, double alpha);
  void (*load_jacobian_tran)(void *inst, void* model, double alpha);
}OsdiDescriptor03;

typedef struct OsdiDescriptor04 {
  char *name;

  uint32_t num_nodes;
  uint32_t num_terminals;
  OsdiNode *nodes;

  uint32_t num_jacobian_entries;
  OsdiJacobianEntry *jacobian_entries;

  uint32_t num_collapsible;
  OsdiNodePair *collapsible;
  uint32_t collapsed_offset;

  OsdiNoiseSource *noise_sources;
  uint32_t num_noise_src;

  uint32_t num_params;
  uint32_t num_instance_params;
  uint32_t num_opvars;
  OsdiParamOpvar *param_opvar;

  uint32_t node_mapping_offset;
  uint32_t jacobian_ptr_resist_offset;

  uint32_t num_states;
  uint32_t state_idx_off;

  uint32_t bound_step_offset;

  uint32_t instance_size;
  uint32_t model_size;

  void *(*access)(void *inst, void *model, uint32_t id, uint32_t flags);

  void (*setup_model)(void *handle, void *model, OsdiSimParas *sim_params,
                                     OsdiInitInfo *res);
  void (*setup_instance)(void *handle, void *inst, void *model,
                                     double temperature, uint32_t num_terminals,
                                     OsdiSimParas *sim_params, OsdiInitInfo *res);

  uint32_t (*eval)(void *handle, void *inst, void *model, OsdiSimInfo *info);
  void (*load_noise)(void *inst, void *model, double freq, double *noise_dens);
  void (*load_residual_resist)(void *inst, void* model, double *dst);
  void (*load_residual_react)(void *inst, void* model, double *dst);
  void (*load_limit_rhs_resist)(void *inst, void* model, double *dst);
  void (*load_limit_rhs_react)(void *inst, void* model, double *dst);
  void (*load_spice_rhs_dc)(void *inst, void* model, double *dst,
                  double* prev_solve);
  void (*load_spice_rhs_tran)(void *inst, void* model, double *dst,
                  double* prev_solve, double alpha);
  void (*load_jacobian_resist)(void *inst, void* model);
  void (*load_jacobian_react)(void *inst, void* model, double alpha);
  void (*load_jacobian_tran)(void *inst, void* model, double alpha);
  uint32_t (*given_flag_model)(void *model, uint32_t id);
  uint32_t (*given_flag_instance)(void *inst, uint32_t id);
  uint32_t num_resistive_jacobian_entries;
  uint32_t num_reactive_jacobian_entries;
  void (*write_jacobian_array_resist)(void *inst, void* model, double* destination);
  void (*write_jacobian_array_react)(void *inst, void* model, double* destination);
  uint32_t num_inputs;
  OsdiNodePair* inputs;
  void (*load_jacobian_with_offset_resist)(void *inst, void* model, size_t offset);
  void (*load_jacobian_with_offset_react)(void *inst, void* model, size_t offset);
  OsdiNatureRef* unknown_nature;
  OsdiNatureRef* residual_nature;
  uint32_t *noise_source_type;
  void (*load_noise_params)(void *inst, void *model, double *power, double *exponent);
}OsdiDescriptor04;
