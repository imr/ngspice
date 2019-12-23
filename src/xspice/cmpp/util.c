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
#include  <stdio.h>
#include  <stdlib.h>
#include  <string.h>

#include  "cmpp.h"


/* *********************************************************************** */

char *prog_name;


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



/* If *path_p is relative, prefix with the CMPP output or input string
 * build the path and open the file and return the path that was created. */
FILE *fopen_cmpp(const char **path_p, const char *mode)
{
    const char *path = *path_p;

    char buf[MAX_PATH_LEN + 1];

    if (path[0] != '/') {
        const char *e = getenv((*mode == 'w') ? "CMPP_ODIR" : "CMPP_IDIR");
        if (e) {
            if (strlen(e) + 1 + strlen(path) < sizeof(buf)) {
                strcpy(buf, e);
                strcat(buf, "/");
                strcat(buf, path);
                path = buf;
            }
            else {
                path = NULL;
            }
        }
    }

    *path_p = strdup(path);

    return fopen(path, mode);
} /* end of function fopen_cmpp */



