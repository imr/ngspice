/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#pragma once

#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/fteext.h"
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

size_t osdi_instance_data_off(const OsdiRegistryEntry *entry);
void *osdi_instance_data(const OsdiRegistryEntry *entry, GENinstance *inst);
OsdiExtraInstData *osdi_extra_instance_data(const OsdiRegistryEntry *entry,
                                            GENinstance *inst);
size_t osdi_model_data_off(void);
void *osdi_model_data(GENmodel *model);
void *osdi_model_data_from_inst(GENinstance *inst);
OsdiRegistryEntry *osdi_reg_entry_model(const GENmodel *model);
OsdiRegistryEntry *osdi_reg_entry_inst(const GENinstance *inst);

typedef struct OsdiNgspiceHandle {
  uint32_t kind;
  char *name;
} OsdiNgspiceHandle;
