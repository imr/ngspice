/*============================================================================
FILE  util.c

MEMBER OF process cmpp

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
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
    vprint_error(fmt, ap);
    va_end(ap);
} /* end of function print_error */



void vprint_error(const char *fmt, va_list p_arg)
{
    fprintf(stderr, "%s: ", prog_name);
    vfprintf(stderr, fmt, p_arg);
    fprintf(stderr, "\n");
} /* end of function vprint_error */



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
char *gen_filename(const char *filename, const char *mode)
{
    char *buf = (char *) NULL;

    /* If absoulte path, prefix with CMPP_ODIR/CMPP_IDIR env value */
    if (!is_absolute_pathname(filename)) { /* relative path */
        const char *e = getenv((*mode == 'w' || *mode == 'a') ?
                    "CMPP_ODIR" : "CMPP_IDIR");
        if (e) { /* have env var */
            const size_t len_prefix = strlen(e);
            const size_t len_filename = strlen(filename);
            const size_t n_char = len_prefix + len_filename + 1;

            /* Allocate buffer to build full file name */
            if ((buf = (char *) malloc(n_char + 1)) == (char *) NULL) {
                return (char *) NULL;
            }

            /* Build the full file name */
            {
                char *p_cur = buf;
                (void) memcpy(p_cur, e, len_prefix);
                p_cur += len_prefix;
                *p_cur++ = DIR_TERM_UNIX;
                (void) memcpy(p_cur, filename, len_filename + 1);
            }
        } /* end of case that env variable found */
    } /* end of case that path is absolute */

    /* If did not build full file name yet, make the original
     * name of the file the full file name */
    if (buf == (char *) NULL) {
        if ((buf = strdup(filename)) == (char *) NULL) { /* failed */
            return (char *) NULL;
        }
    }

    return buf;
} /* end of function gen_filename */



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



