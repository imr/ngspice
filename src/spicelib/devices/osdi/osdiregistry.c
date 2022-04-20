/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#include "ngspice/memory.h"
#include "ngspice/stringutil.h"
#include "osdidefs.h"

#include "ngspice/iferrmsg.h"
#include "osdi.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <direct.h>
#define getcwd() _getcwd
#elif defined(__GNUC__)
#include <unistd.h>
#endif

// TODO: platform portability

#include "dirent.h"
#include <dlfcn.h>

char *OSDI_Path = NULL;
OsdiRegistryEntry *registry = NULL;
uint32_t registry_off = 0;

/**
 * Calculates the offset that the OSDI instance data has from the beginning of
 * the instance data allocated by ngspice. This offset is non trivial because
 * ngspice must store the terminal pointers before the remaining instance data.
 * As a result the offset is not constant and a variable amount of padding must
 * be inserted to ensure correct alginment.
 */
static size_t calc_osdi_instance_data_off(const OsdiDescriptor *descr) {
  size_t res = sizeof(GENinstance) /* generic data */
               + descr->num_terminals * sizeof(int);
  size_t padding = sizeof(max_align_t) - res % sizeof(max_align_t);
  if (padding == sizeof(max_align_t)) {
    padding = 0;
  }
  return res + padding;
}

/**
 * Loads an object file from the hard drive with the platform equivalent of
 * dlopen. This function checks that the OSDI version of the object file is
 * valid and then writes all data into the `registry`.
 * If any errors occur an appropriate message is written to errMsg
 *
 * @param PATH path A path to the shared object file
 * @param uint32_t* len The amount of entries already written into `registry`
 * @param uint32_t* capacity The amount of space available in `registry` before
 * reallocation is required
 * @returns -1 on error, 0 otherwise
 */
static int load_object_file(char *path, uint32_t *len, uint32_t *capacity) {
  void *handle;
  char *error;
  handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    errMsg = dlerror();
    return -1;
  }
  dlerror(); /* Clear any existing error */

  const uint32_t *OSDI_VERSION_MAJOR = dlsym(handle, "OSDI_VERSION_MAJOR");
  if ((error = dlerror()) != NULL) {
    errMsg = dlerror();
    return -1;
  }

  const uint32_t *OSDI_VERSION_MINOR = dlsym(handle, "OSDI_VERSION_MINOR");
  if ((error = dlerror()) != NULL) {
    errMsg =
        tprintf("Failed to load OSDI object file \"%s\": %s", path, dlerror());
    return -1;
  }

  if (*OSDI_VERSION_MAJOR != OSDI_VERSION_MAJOR_CURR ||
      *OSDI_VERSION_MINOR != OSDI_VERSION_MINOR_CURR) {
    errMsg =
        tprintf("NGSPICE only supports OSDI v%d.%d but \"%s\" targets v%d.%d!",
                OSDI_VERSION_MAJOR_CURR, OSDI_VERSION_MINOR_CURR, path,
                *OSDI_VERSION_MAJOR, *OSDI_VERSION_MINOR);
    return -1;
  }

  const uint32_t *OSDI_NUM_DESCRIPTORS = dlsym(handle, "OSDI_NUM_DESCRIPTORS");
  if ((error = dlerror()) != NULL) {
    errMsg = dlerror();
    return -1;
  }

  const OsdiDescriptor *OSDI_DESCRIPTORS = dlsym(handle, "OSDI_DESCRIPTORS");
  if ((error = dlerror()) != NULL) {
    errMsg = dlerror();
    return -1;
  }

  if ((*len + *OSDI_NUM_DESCRIPTORS) > *capacity) {
    *capacity *= 2;
    registry = realloc(registry, *capacity);
  }

  for (uint32_t i = 0; i < *OSDI_NUM_DESCRIPTORS; i++) {
    const OsdiDescriptor *descr = &OSDI_DESCRIPTORS[i];

    uint32_t dt = descr->num_params + descr->num_opvars;
    uint32_t temp = descr->num_params + descr->num_opvars + 1;
    for (uint32_t param_id = 0; param_id < descr->num_params; param_id++) {
      OsdiParamOpvar *param = &descr->param_opvar[param_id];
      for (uint32_t j = 0; j < 1 + param->num_alias; j++) {
        char *name = param->name[j];
        if (!strcmp(name, "dt")) {
          dt = UINT32_MAX;
        } else if (!strcasecmp(name, "dtemp") || !strcasecmp(name, "dt")) {
          dt = param_id;
        } else if (!strcmp(name, "temp")) {
          temp = UINT32_MAX;
        } else if (!strcasecmp(name, "temp") ||
                   !strcasecmp(name, "temperature")) {
          temp = param_id;
        }
      }
    }

    size_t inst_off = calc_osdi_instance_data_off(descr);
    registry[i + *len] = (OsdiRegistryEntry){
        .descriptor = descr,
        .inst_offset = (uint32_t)inst_off,
        .dt = dt,
        .temp = temp,
    };
  }

  *len += *OSDI_NUM_DESCRIPTORS;

  return 0;
}

static char **osdi_search_dir(char *dir, char **dst, uint32_t *len,
                              uint32_t *cap) {
  DIR *wdir = opendir(dir);
  if (wdir == NULL) {
    return dst;
  }
  struct direct *de;
  while ((de = readdir(wdir)) != NULL) {
    if (!(de->d_ino & (DT_REG | DT_LNK))) {
      continue;
    }
    const char *filename = de->d_name;
    const char *ext = strrchr(filename, '.');

    if (!ext || ext == filename) {
      continue;
    } else {
      ext += 1;
    }
    if (strcmp(ext, "so") != 0) {
      continue;
    }
    if (*len >= *cap) {
      *cap *= 2;
      dst = realloc(dst, *cap * sizeof(char *));
    }
    char *path = tprintf("%s%s%s", dir, DIR_PATHSEP, filename);

    if (de->d_ino & DT_LNK) {
      char *full_path = realpath(path, NULL);
      tfree(path);
      path = full_path;
    }

    dst[*len] = path;
    *len += 1;
  }
  return dst;
}

static char **osdi_search_devices(uint32_t *len) {
  uint32_t cap = 8;
  *len = 0;
  char **dst = TMALLOC(char *, cap);
  dst = osdi_search_dir(OSDI_Path, dst, len, &cap);
  char *cwd = getcwd(NULL, 0);
  if (cwd) {
    char *local_osdi_dir = tprintf("%s%s%s", cwd, DIR_PATHSEP, "osdi");
    dst = osdi_search_dir(local_osdi_dir, dst, len, &cap);
    tfree(cwd);
    tfree(local_osdi_dir);
  }
  return dst;
}

/**
 * Loads a list of object files into the global OSDI registry.
 * This function must only be called once at program startup.
 *
 * @param PATH *object_files The paths of the object files
 * @param uint32_t num_files The number of object files to load
 * @returns The number of loaded descriptors
 */
static int osdi_load_devices_from_files(char **object_files,
                                        uint32_t num_files) {
  uint32_t len = 0;
  uint32_t capacity = num_files + 8;
  registry = TMALLOC(OsdiRegistryEntry, capacity);
  for (uint32_t i = 0; i < num_files; i++) {
    int res = load_object_file(object_files[i], &len, &capacity);
    if (res) {
      return res;
    }
  }
  registry = realloc(registry, len);
  return (int)len;
}

extern int osdi_load_devices(void) {
  uint32_t len = 0;
  char **files = osdi_search_devices(&len);
  int res = osdi_load_devices_from_files(files, len);
  // free paths
  for (uint32_t i = 0; i < len; i++) {
    tfree(files[i]);
  }
  tfree(files);
  return res;
}

size_t osdi_instance_data_off(OsdiRegistryEntry entry);
void *osdi_instance_data(OsdiRegistryEntry entry, GENinstance *inst);
OsdiExtraInstData *osdi_extra_instance_data(OsdiRegistryEntry entry,
                                            GENinstance *inst);
size_t osdi_model_data_off(void);
void *osdi_model_data(GENmodel *model);
void *osdi_model_data_from_inst(GENinstance *inst);
OsdiRegistryEntry osdi_reg_entry_model(GENmodel *model);
OsdiRegistryEntry osdi_reg_entry_inst(GENinstance *inst);
