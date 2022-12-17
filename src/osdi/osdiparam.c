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
#include "ngspice/ngspice.h"
#include "ngspice/typedefs.h"

#include "osdidefs.h"

#include <stdint.h>
#include <string.h>

static int osdi_param_access(OsdiParamOpvar *param_info, bool write_value,
                             IFvalue *value, void *ptr) {
  size_t len;
  void *val_ptr;
  switch (param_info->flags & PARA_TY_MASK) {
  case PARA_TY_REAL:
    len = sizeof(double);
    if (param_info->len) {
      len *= param_info->len;
      val_ptr = &value->v.vec.rVec;
    } else {
      val_ptr = &value->rValue;
    }
    break;
  case PARA_TY_INT:
    len = sizeof(int);
    if (param_info->len) {
      len *= param_info->len;
      val_ptr = &value->v.vec.iVec;
    } else {
      val_ptr = &value->iValue;
    }
    break;
  case PARA_TY_STR:
    len = sizeof(char *);
    if (param_info->len) {
      len *= param_info->len;
      val_ptr = &value->v.vec.cVec;
    } else {
      val_ptr = &value->cValue;
    }
    break;
  default:
    return (E_PARMVAL);
  }
  if (write_value) {
    memcpy(val_ptr, ptr, len);
  } else {
    memcpy(ptr, val_ptr, len);
  }

  return OK;
}

static int osdi_write_param(void *dst, IFvalue *value, int param,
                            const OsdiDescriptor *descr) {
  if (dst == NULL) {
    return (E_PANIC);
  }

  OsdiParamOpvar *param_info = &descr->param_opvar[param];

  if (param_info->len) {
    if ((uint32_t)value->v.numValue != param_info->len) {
      return (E_PARMVAL);
    }
  }

  return osdi_param_access(param_info, false, value, dst);
}

extern int OSDIparam(int param, IFvalue *value, GENinstance *instPtr,
                     IFvalue *select) {

  NG_IGNORE(select);
  OsdiRegistryEntry *entry = osdi_reg_entry_inst(instPtr);
  const OsdiDescriptor *descr = entry->descriptor;

  if (param >= (int)descr->num_instance_params) {
    // special handling for temperature parameters
    OsdiExtraInstData *inst = osdi_extra_instance_data(entry, instPtr);
    if (param == (int)entry->dt) {
      inst->dt = value->rValue;
      inst->dt_given = true;
      return (OK);
    }
    if (param == (int)entry->temp) {
      inst->temp = value->rValue;
      inst->temp_given = true;
      return (OK);
    }

    return (E_BADPARM);
  }

  void *inst = osdi_instance_data(entry, instPtr);
  void *dst = descr->access(inst, NULL, (uint32_t)param,
                            ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE);

  return osdi_write_param(dst, value, param, descr);
}

extern int OSDImParam(int param, IFvalue *value, GENmodel *modelPtr) {
  OsdiRegistryEntry *entry = osdi_reg_entry_model(modelPtr);
  const OsdiDescriptor *descr = entry->descriptor;

  if (param > (int)descr->num_params ||
      param < (int)descr->num_instance_params) {
    return (E_BADPARM);
  }

  void *model = osdi_model_data(modelPtr);
  void *dst = descr->access(NULL, model, (uint32_t)param, ACCESS_FLAG_SET);

  return osdi_write_param(dst, value, param, descr);
}

static int osdi_read_param(void *src, IFvalue *value, int id,
                           const OsdiDescriptor *descr) {
  if (src == NULL) {
    return (E_PANIC);
  }

  OsdiParamOpvar *param_info = &descr->param_opvar[id];

  if (param_info->len) {
    value->v.numValue = (int)param_info->len;
  }

  return osdi_param_access(param_info, true, value, src);
}

extern int OSDIask(CKTcircuit *ckt, GENinstance *instPtr, int id,
                   IFvalue *value, IFvalue *select) {
  NG_IGNORE(select);
  NG_IGNORE(ckt);

  OsdiRegistryEntry *entry = osdi_reg_entry_inst(instPtr);
  void *inst = osdi_instance_data(entry, instPtr);
  void *model = osdi_model_data_from_inst(instPtr);

  const OsdiDescriptor *descr = entry->descriptor;

  if (id >= (int)(descr->num_params + descr->num_opvars)) {
    return (E_BADPARM);
  }
  uint32_t flags = ACCESS_FLAG_READ;
  if (id < (int)descr->num_instance_params) {
    flags |= ACCESS_FLAG_INSTANCE;
  }

  void *src = descr->access(inst, model, (uint32_t)id, flags);
  return osdi_read_param(src, value, id, descr);
}
