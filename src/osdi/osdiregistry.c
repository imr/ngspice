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

#include "ngspice/hash.h"
#include "ngspice/memory.h"
#include "ngspice/stringutil.h"
#include "osdidefs.h"

#include <sys/stat.h>

#include "osdi.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if (!defined HAS_WINGUI) && (!defined __MINGW32__) && (!defined _MSC_VER)

#include <dlfcn.h> /* to load libraries*/
#define OPENLIB(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define GET_SYM(lib, sym) dlsym(lib, sym)
#define FREE_DLERR_MSG(msg)

#else /* ifdef HAS_WINGUI */

#undef BOOLEAN
#include <windows.h>
#include <shlwapi.h>
#define OPENLIB(path) LoadLibrary(path)
#define GET_SYM(lib, sym) ((void *)GetProcAddress(lib, sym))
char *dlerror(void);
#define FREE_DLERR_MSG(msg) free_dlerr_msg(msg)
static void free_dlerr_msg(char *msg);

#endif /* ifndef HAS_WINGUI */

char *inputdir = NULL;

/* Returns true if path is an absolute path and false if it is a
 * relative path. No check is done for the existance of the path. */
inline static bool is_absolute_pathname(const char *path) {
#ifdef _WIN32
  return !PathIsRelativeA(path);
#else
  return path[0] == DIR_TERM;
#endif
} /* end of funciton is_absolute_pathname */

/*-------------------------------------------------------------------------*
  Look up the variable sourcepath and try everything in the list in order
  if the file isn't in . and it isn't an abs path name.
  *-------------------------------------------------------------------------*/

static char *resolve_path(const char *name) {
  struct stat st;

#if defined(_WIN32)

  /* If variable 'mingwpath' is set: convert mingw /d/... to d:/... */
  if (cp_getvar("mingwpath", CP_BOOL, NULL, 0) && name[0] == DIR_TERM_LINUX &&
      isalpha_c(name[1]) && name[2] == DIR_TERM_LINUX) {
    DS_CREATE(ds, 100);
    if (ds_cat_str(&ds, name) != 0) {
      fprintf(stderr, "Unable to copy string while resolving path");
      controlled_exit(EXIT_FAILURE);
    }
    char *const buf = ds_get_buf(&ds);
    buf[0] = buf[1];
    buf[1] = ':';
    char *const resolved_path = resolve_path(buf);
    ds_free(&ds);
    return resolved_path;
  }

#endif

  /* just try it */
  if (stat(name, &st) == 0)
    return copy(name);

#if !defined(EXT_ASC) && (defined(__MINGW32__) || defined(_MSC_VER))
  wchar_t wname[BSIZE_SP];
  if (MultiByteToWideChar(CP_UTF8, 0, name, -1, wname,
                          2 * (int)strlen(name) + 1) == 0) {
    fprintf(stderr, "UTF-8 to UTF-16 conversion failed with 0x%x\n",
            GetLastError());
    fprintf(stderr, "%s could not be converted\n", name);
    return NULL;
  }
  if (_waccess(wname, 0) == 0)
    return copy(name);
#endif

  return (char *)NULL;
} /* end of function inp_pathresolve */

static char *resolve_input_path(const char *name) {
  /* if name is an absolute path name,
   *   or if we haven't anything to prepend anyway
   */
  if (is_absolute_pathname(name)) {
    return resolve_path(name);
  }

  if (name[0] == '~' && name[1] == '/') {
    char *const y = cp_tildexpand(name);
    if (y) {
      char *const r = resolve_path(y);
      txfree(y);
      return r;
    }
  }

  /*
   * If called from a script inputdir != NULL so try relativ to that dir
   * Otherwise try relativ to the current workdir and relativ to the
   * executables path
   */

  if (inputdir) {
      DS_CREATE(ds, 100);
      int rc_ds = 0;
      rc_ds |= ds_cat_str(&ds, inputdir);  /* copy the dir name */
      const size_t n = ds_get_length(&ds); /* end of copied dir name */

      /* Append a directory separator if not present already */
      const char ch_last = n > 0 ? inputdir[n - 1] : '\0';
      if (ch_last != DIR_TERM
#ifdef _WIN32
          && ch_last != DIR_TERM_LINUX
#endif
          ) {
          rc_ds |= ds_cat_char(&ds, DIR_TERM);
      }
      rc_ds |= ds_cat_str(&ds, name); /* append the file name */

      if (rc_ds != 0) {
          (void)fprintf(cp_err, "Unable to build \"dir\" path name "
              "in inp_pathresolve_at");
          controlled_exit(EXIT_FAILURE);
      }

      char* const r = resolve_path(ds_get_buf(&ds));
      ds_free(&ds);
      if (r)
          return r;
  }

  if (Spice_Exec_Path && *Spice_Exec_Path) {
      DS_CREATE(ds, 100);
      int rc_ds = 0;
      rc_ds |= ds_cat_str(&ds, Spice_Exec_Path);  /* copy the dir name */
      const size_t n = ds_get_length(&ds); /* end of copied dir name */

      /* Append a directory separator if not present already */
      const char ch_last = n > 0 ? Spice_Exec_Path[n - 1] : '\0';
      if (ch_last != DIR_TERM
#ifdef _WIN32
          && ch_last != DIR_TERM_LINUX
#endif
          ) {
          rc_ds |= ds_cat_char(&ds, DIR_TERM);
      }
      rc_ds |= ds_cat_str(&ds, name); /* append the file name */

      if (rc_ds != 0) {
          (void)fprintf(cp_err, "Unable to build \"dir\" path name "
              "in inp_pathresolve_at");
          controlled_exit(EXIT_FAILURE);
      }

      char* const r = resolve_path(ds_get_buf(&ds));
      ds_free(&ds);
      if (r)
          return r;
  }
  /* no inputdir, or not found relative to inputdir:
   search relative to current working directory */
  DS_CREATE(ds, 100);
  if (ds_cat_printf(&ds, ".%c%s", DIR_TERM, name) != 0) {
      (void)fprintf(cp_err,
          "Unable to build \".\" path name in inp_pathresolve_at");
      controlled_exit(EXIT_FAILURE);
  }
  char* const r = resolve_path(ds_get_buf(&ds));
  ds_free(&ds);
  if (r != (char*)NULL) {
      return r;
  }

  return NULL;
} /* end of function inp_pathresolve_at */

/**
 * Calculates the offset that the OSDI instance data has from the beginning of
 * the instance data allocated by ngspice. This offset is non trivial because
 * ngspice must store the terminal pointers before the remaining instance
 * data. As a result the offset is not constant and a variable amount of
 * padding must be inserted to ensure correct alginment.
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

#define INVALID_OBJECT                                                         \
  (OsdiObjectFile) { .num_entries = -1 }

#define EMPTY_OBJECT                                                           \
  (OsdiObjectFile) {NULL, 0}

#define ERR_AND_RET                                                            \
  error = dlerror();                                                           \
  fprintf(stderr, "Error opening osdi lib \"%s\": %s\n", path, error);                  \
  FREE_DLERR_MSG(error);                                                       \
  return INVALID_OBJECT;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define GET_CONST(name, ty)                                                    \
  sym = GET_SYM(handle, STRINGIFY(name));                                      \
  if (!sym) {                                                                  \
    ERR_AND_RET                                                                \
  }                                                                            \
  const ty name = *((ty *)sym);

#define GET_PTR(name, ty)                                                      \
  sym = GET_SYM(handle, STRINGIFY(name));                                      \
  if (!sym) {                                                                  \
    ERR_AND_RET                                                                \
  }                                                                            \
  const ty *name = (ty *)sym;

#define INIT_CALLBACK(name, ty)                                                \
  sym = GET_SYM(handle, STRINGIFY(name));                                      \
  if (sym) {                                                                   \
    ty *slot = (ty *)sym;                                                      \
    *slot = name;                                                              \
  }

#define IS_LIM_FUN(fun_name, num_args_, val)                                   \
  if (strcmp(lim_table[i].name, fun_name) == 0) {                              \
    if (lim_table[i].num_args == num_args_) {                                  \
      lim_table[i].func_ptr = (void *)val;                                     \
      continue;                                                                \
    } else {                                                                   \
      expected_args = num_args_;                                               \
    }                                                                          \
  }

static NGHASHPTR known_object_files = NULL;
#define DUMMYDATA ((void *)42)
/**
 * Loads an object file from the hard drive with the platform equivalent of
 * dlopen. This function checks that the OSDI version of the object file is
 * valid and then writes all data into the `registry`.
 * If any errors occur an appropriate message is written to errMsg
 *
 * @param PATH path A path to the shared object file
 * @param uint32_t* len The amount of entries already written into `registry`
 * @param uint32_t* capacity The amount of space available in `registry`
 * before reallocation is required
 * @returns -1 on error, 0 otherwise
 */
extern OsdiObjectFile load_object_file(const char *input) {
  void *handle;
  char *error;
  const void *sym;
  /* ensure the hashtable exists */
  if (!known_object_files) {
    known_object_files = nghash_init_pointer(8);
  }
  const char *path = resolve_input_path(input);
  if (!path) {
    fprintf(stderr, "Error opening osdi lib \"%s\": No such file or directory!\n",
           input);
    return INVALID_OBJECT;
  }

  handle = OPENLIB(path);
  if (!handle) {
    ERR_AND_RET
  }

  /* Keep track of loaded shared object files to avoid loading the same model
   * multiple times. We use the handle as a key because the same SO will always
   * return the SAME pointer as long as dlclose is not called.
   * nghash_insert returns NULL if the key (handle) was not already in the table
   * and the data (DUMMYDATA) that was previously insered (!= NULL) otherwise*/
  if (nghash_insert(known_object_files, handle, DUMMYDATA)) {
    txfree(path);
    return EMPTY_OBJECT;
  }

  GET_CONST(OSDI_VERSION_MAJOR, uint32_t);
  GET_CONST(OSDI_VERSION_MINOR, uint32_t);

  if (OSDI_VERSION_MAJOR != OSDI_VERSION_MAJOR_CURR ||
      OSDI_VERSION_MINOR != OSDI_VERSION_MINOR_CURR) {
    printf("NGSPICE only supports OSDI v%d.%d but \"%s\" targets v%d.%d!",
           OSDI_VERSION_MAJOR_CURR, OSDI_VERSION_MINOR_CURR, path,
           OSDI_VERSION_MAJOR, OSDI_VERSION_MINOR);
    txfree(path);
    return INVALID_OBJECT;
  }

  GET_CONST(OSDI_NUM_DESCRIPTORS, uint32_t);
  GET_PTR(OSDI_DESCRIPTORS, OsdiDescriptor);

  INIT_CALLBACK(osdi_log, osdi_log_ptr)

  uint32_t lim_table_len = 0;
  sym = GET_SYM(handle, "OSDI_LIM_TABLE_LEN");
  if (sym) {
    lim_table_len = *((uint32_t *)sym);
  }

  sym = GET_SYM(handle, "OSDI_LIM_TABLE");
  OsdiLimFunction *lim_table = NULL;
  if (sym) {
    lim_table = (OsdiLimFunction *)sym;
  } else {
    lim_table_len = 0;
  }

  for (uint32_t i = 0; i < lim_table_len; i++) {
    int expected_args = -1;
    IS_LIM_FUN("pnjlim", 2, osdi_pnjlim)
    IS_LIM_FUN("limvds", 0, osdi_limvds)
    IS_LIM_FUN("fetlim", 1, osdi_fetlim)
    IS_LIM_FUN("limitlog", 1, osdi_limitlog)
    if (expected_args == -1) {
      printf("warning(osdi): unkown $limit function \"%s\"", lim_table[i].name);
    } else {
      printf("warning(osdi): unexpected number of arguments %i (expected %i) "
             "for \"%s\", ignoring...",
             lim_table[i].num_args, expected_args, lim_table[i].name);
    }
  }

  OsdiRegistryEntry *dst = TMALLOC(OsdiRegistryEntry, OSDI_NUM_DESCRIPTORS);

  for (uint32_t i = 0; i < OSDI_NUM_DESCRIPTORS; i++) {
    const OsdiDescriptor *descr = &OSDI_DESCRIPTORS[i];

    uint32_t dt = descr->num_params + descr->num_opvars;
    bool has_m = false;
    uint32_t temp = descr->num_params + descr->num_opvars + 1;
    for (uint32_t param_id = 0; param_id < descr->num_params; param_id++) {
      OsdiParamOpvar *param = &descr->param_opvar[param_id];
      for (uint32_t j = 0; j < 1 + param->num_alias; j++) {
        char *name = param->name[j];
        if (!strcmp(name, "m")) {
          has_m = true;
        } else if (!strcmp(name, "dt")) {
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
    dst[i] = (OsdiRegistryEntry){
        .descriptor = descr,
        .inst_offset = (uint32_t)inst_off,
        .dt = dt,
        .temp = temp,
        .has_m = has_m,
    };
  }

  txfree(path);
  return (OsdiObjectFile){
      .entrys = dst,
      .num_entries = (int)OSDI_NUM_DESCRIPTORS,
  };
}

inline size_t osdi_instance_data_off(const OsdiRegistryEntry *entry) {
  return entry->inst_offset;
}

inline void *osdi_instance_data(const OsdiRegistryEntry *entry,
                                GENinstance *inst) {
  return (void *)(((char *)inst) + osdi_instance_data_off(entry));
}

inline OsdiExtraInstData *
osdi_extra_instance_data(const OsdiRegistryEntry *entry, GENinstance *inst) {
  OsdiDescriptor *descr = (OsdiDescriptor *)entry->descriptor;
  return (OsdiExtraInstData *)(((char *)inst) + entry->inst_offset +
                               descr->instance_size);
}

inline size_t osdi_model_data_off(void) {
  return offsetof(OsdiModelData, data);
}

inline void *osdi_model_data(GENmodel *model) {
  return (void *)&((OsdiModelData *)model)->data;
}

inline void *osdi_model_data_from_inst(GENinstance *inst) {
  return osdi_model_data(inst->GENmodPtr);
}

inline OsdiRegistryEntry *osdi_reg_entry_model(const GENmodel *model) {
  return (OsdiRegistryEntry *)ft_sim->devices[model->GENmodType]
      ->registry_entry;
}

inline OsdiRegistryEntry *osdi_reg_entry_inst(const GENinstance *inst) {
  return osdi_reg_entry_model(inst->GENmodPtr);
}

#if defined(__MINGW32__) || defined(HAS_WINGUI) || defined(_MSC_VER)

/* For reporting error message if formatting fails */
static const char errstr_fmt[] =
    "Unable to find message in dlerr(). System code = %lu";
static char errstr[sizeof errstr_fmt - 3 + 3 * sizeof(unsigned long)];

#if !defined (XSPICE)
char *dlerror(void) {
  LPVOID lpMsgBuf;

  DWORD rc = FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR)&lpMsgBuf, 0, NULL);

  if (rc == 0) { /* FormatMessage failed */
    (void)sprintf(errstr, errstr_fmt, (unsigned long)GetLastError());
    return errstr;
  }

  return lpMsgBuf; /* Return the formatted message */
} /* end of function dlerror */
#endif

/* Free message related to dynamic loading */
static void free_dlerr_msg(char *msg) {
  if (msg != errstr) { /* msg is an allocation */
    LocalFree(msg);
  }
} /* end of function free_dlerr_msg */

#endif /* Windows emulation of dlerr */
