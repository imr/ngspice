/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#pragma once

#include <stdint.h>

/**
 * Loads a list of object files into the global OSDI registry.
 * This function must only be called once at program startup.
 *
 * @param PATH *object_files The paths of the object files
 * @param uint32_t num_files The number of object files to load
 * @returns The number of loaded descriptors
 */
extern int osdi_load_devices(void);
extern void osdi_get_info(uint32_t off, uint32_t num_descriptors);
