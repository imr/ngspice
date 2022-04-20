/*
 * This file is part of the OSDI component of NGSPICE.
 * Copyright© 2022 SemiMod GmbH.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Pascal Kuthe <pascal.kuthe@semimod.de>
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
