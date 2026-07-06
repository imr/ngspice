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

#pragma once

#include "ngspice/config.h"
#include "ngspice/devdefs.h"
#include <stdint.h>

typedef struct OsdiRegistryEntry {
  const void *descriptor;
  uint32_t inst_offset;
  uint32_t noise_offset;
  uint32_t dt;
  uint32_t temp;

  bool has_m;

  /* OSDI v0.5 (2C) — number of absdelay transport-delay sites, guarded at
   * load time against the published OSDI_DESCRIPTOR_SIZE so a model compiled
   * before this field existed reads 0 (not garbage past its descriptor). */
  uint32_t num_delay_sites;

  /* OSDI 0.5 — persistent-state array (transition()/slew()/event toolkit):
   * byte offset within the instance + slot count, both guarded against the
   * published descriptor size.  Used to snapshot/restore the array for
   * deferred commit (only accepted steps update the model's history). */
  uint32_t persistent_state_offset;
  uint32_t persistent_state_count;

#ifdef KLU
  uint32_t matrix_ptr_offset;
#endif

} OsdiRegistryEntry;

typedef struct OsdiObjectFile {
  OsdiRegistryEntry *entrys;
  int num_entries;
} OsdiObjectFile;

extern OsdiObjectFile load_object_file(const char *path);
extern SPICEdev *osdi_create_spicedev(const OsdiRegistryEntry *entry);

extern char *inputdir;
