/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#pragma once

#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/gendefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/ngspice.h"
#include "ngspice/noisedef.h"
#include "ngspice/typedefs.h"

#include "osdi.h"
#include "osdiext.h"

#include "stddef.h"
#include <stddef.h>
#include <stdint.h>

typedef struct OsdiModelData {
  GENmodel gen;
  max_align_t data;
} OsdiModelData;

typedef struct OsdiExtraInstData {
  double dt;
  double temp;
  bool temp_given;
  bool dt_given;
  bool lim;
  bool finish;
  
} __attribute__((aligned(sizeof(max_align_t)))) OsdiExtraInstData;

typedef struct OsdiRegistryEntry {
  const OsdiDescriptor *descriptor;
  uint32_t inst_offset;
  uint32_t dt;
  uint32_t temp;
} OsdiRegistryEntry;

extern OsdiRegistryEntry *registry;
extern uint32_t registry_off;

inline size_t osdi_instance_data_off(OsdiRegistryEntry entry) {
  return entry.inst_offset;
}
inline void *osdi_instance_data(OsdiRegistryEntry entry, GENinstance *inst) {
  return (void *)(((char *)inst) + osdi_instance_data_off(entry));
}
inline OsdiExtraInstData *osdi_extra_instance_data(OsdiRegistryEntry entry,
                                                   GENinstance *inst) {
  return (OsdiExtraInstData *)(((char *)inst) + entry.inst_offset +
                               entry.descriptor->instance_size);
}

inline size_t osdi_model_data_off() { return offsetof(OsdiModelData, data); }

inline void *osdi_model_data(GENmodel *model) {
  return (void *)&((OsdiModelData *)model)->data;
}

inline void *osdi_model_data_from_inst(GENinstance *inst) {
  return osdi_model_data(inst->GENmodPtr);
}

inline OsdiRegistryEntry osdi_reg_entry_model(GENmodel *model) {
  return registry[(uint32_t)model->GENmodType - registry_off];
}

inline OsdiRegistryEntry osdi_reg_entry_inst(GENinstance *inst) {
  return osdi_reg_entry_model(inst->GENmodPtr);
}

typedef struct OsdiNgspiceHandle {
  uint32_t kind;
  char *name;
} OsdiNgspiceHandle;
