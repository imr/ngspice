/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#pragma once

#include "ngspice/devdefs.h"
#include <stdint.h>

typedef struct OsdiRegistryEntry {
  const void *descriptor;
  uint32_t inst_offset;
  uint32_t dt;
  uint32_t temp;
} OsdiRegistryEntry;

typedef struct OsdiObjectFile {
  OsdiRegistryEntry *entrys;
  int num_entries;
} OsdiObjectFile;

extern OsdiObjectFile load_object_file(const char *path);
extern SPICEdev *osdi_create_spicedev(const OsdiRegistryEntry *entry);

extern char *inputdir;
