/*---------------------------------------------------------------------
 * Do not change this data manually,
 * it will be automatically updated by RCS
 *
 * $RCSfile$
 * $Date: 2012-07-19 13:58:39 +0200 (Thu, 19 Jul 2012) $
 * $Revision: 11822 $
 *
 * ----------------------------------------------------------------------*/

#ifndef _SIMKIT_MODELS_LOADER_H
#define _SIMKIT_MODELS_LOADER_H

/**
 * @file
 * This file contains exported functions to load and unload the
 * SiMKit models library.
 */
//#ifdef __cplusplus
//extern "C" {
//#endif

#include <stdio.h>
#include <stdarg.h>

#include "sk.h"

#ifdef _WIN32
#define   snprintf  _snprintf
#define   vsnprintf _vsnprintf
#define   MAXPATHLEN 1024

#ifdef SIMKIT_MODEL_LOAD_IMPL
#define SK_DECL extern "C" __declspec(dllexport) 
#else
#define SK_DECL extern "C" 
#endif

#else
#define SK_DECL extern 
#endif


/**
 * Load the shared library with the NXP models.
 * When the library cannot be loaded the function prints an error to stderr.
 * @return the pointer to the SK_MODELLIB_DESCRIPTOR (NULL on error).
 */

SK_DECL SK_MODELLIB_DESCRIPTOR* SK_load_models(void);

/**
 * Unload a previously loaded shared library.
 * Multiple calls are allowed without any harm.
 */

SK_DECL void SK_unload_models(void);

//#ifdef __cplusplus
//}
//#endif

#endif /* _SIMKIT_MODELS_LOADER_H */
