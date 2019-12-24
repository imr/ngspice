/*============================================================================
FILE  util.c

MEMBER OF process cmpp

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn and Steve Tynor

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains miscellaneous utility functions used in cmpp.

INTERFACES

    init_error()
    print_error()
    str_to_lower()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include  <ctype.h>
#include  <stdarg.h>
#include  <stdbool.h>
#include  <stdio.h>
#include  <stdlib.h>
#include  <string.h>

#if defined(_WIN32)
#include <shlwapi.h> /* for definition of PathIsRelativeA() */
#if defined(_MSC_VER)
#pragma comment(lib, "Shlwapi.lib")
#endif
#endif

#include  "cmpp.h"


/* Using only Unix directory separator since it is used to build directory
 * paths in include files that must work on any supported operating
 * system */
#define DIR_TERM_UNIX '/'


/* *********************************************************************** */

char *prog_name;


inline static bool is_absolute_pathname(const char *path);


/* Initialize external variable prog_name with the name of the program.
 * A copy is not made. */
void init_error(char *program_name)
{
   prog_name = program_name;
} /* end of function init_error */



/* Print an error message to stderr. The message is prefixed with the
 * name of the program and a newline character is added to the end. */
void print_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);

    fprintf(stderr, "%s: ", prog_name);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");

    va_end(ap);
} /* end of function print_error */



/* Convert a string to all lower case */
void str_to_lower(char *s)
{
    int i;
    char c;

    for(i = 0; (c = s[i]) != '\0'; i++) {
        s[i] = tolower_c(c);
    }
} /* end of function str_to_lower */



/* If *path_p is relative, prefix with the value of the CMPP output or
 * input environment variable. Open the file and return the path that
 * was used to open it. */
FILE *fopen_cmpp(const char **path_p, const char *mode)
{
    const char *path = *path_p;
    char *buf = (char *) NULL;

    /* If absoulte path, prefix with CMPP_ODIR/CMPP_IDIR env value */
    if (!is_absolute_pathname(path)) { /* relative path */
        const char *e = getenv((*mode == 'w' || *mode == 'a') ?
                    "CMPP_ODIR" : "CMPP_IDIR");
        if (e) { /* have env var */
            const size_t len_prefix = strlen(e);
            const size_t len_path = strlen(path);
            const size_t n_char = len_prefix + len_path + 1;

            /* Allocate buffer to build full file name */
            if ((buf = (char *) malloc(n_char + 1)) == (char *) NULL) {
                *path_p = (char *) NULL;
                return (FILE *) NULL;
            }

            /* Build the full file name */
            {
                char *p_cur = buf;
                (void) memcpy(p_cur, e, len_prefix);
                p_cur += len_prefix;
                *p_cur++ = DIR_TERM_UNIX;
                (void) memcpy(p_cur, path, len_path + 1);
            }
        } /* end of case that env variable found */
    } /* end of case that path is absolute */

    /* If did not build full file name yet, copy the original
     * name of the file */
    if (buf == (char *) NULL) {
        if ((buf = strdup(path)) == (char *) NULL) { /* failed */
            *path_p = (char *) NULL;
            return (FILE *) NULL;
        }
    }

    /* Return copy of file name and opened file */
    *path_p = buf;
    return fopen(buf, mode);
} /* end of function fopen_cmpp */



/* Returns true if path is an absolute path and false if it is a
 * relative path. No check is done for the existance of the path. */
/*** NOTE: Same as in inpcom.c Currently the cmpp project is "isolated
 * from others. It would be good to make into one function used in common */
inline static bool is_absolute_pathname(const char *path)
{
#ifdef _WIN32
    return !PathIsRelativeA(path);
#else
    return path[0] == DIR_TERM_UNIX;
#endif
} /* end of funciton is_absolute_pathname */



