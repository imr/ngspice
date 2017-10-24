/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 *
 * Note that this definition must be the same as struct card in INPdefs.h...
 */

#ifndef ngspice_FTEINP_H
#define ngspice_FTEINP_H

#include "ngspice/inpdefs.h"

/* This struct defines a linked list of lines from a SPICE file. */
struct line {
    int li_linenum;
    int li_linenum_orig;
    char *li_line;
    char *li_error;
    struct line *li_next;
    struct line *li_actual;
    struct nscope *level;
} ;

/* Listing types. */

#define LS_LOGICAL  1
#define LS_PHYSICAL 2
#define LS_DECK  3

#endif

