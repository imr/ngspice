/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Standard definitions. This file serves as the header file for std.c and
 * wlist.c
 */

#ifndef _STD_H_
#define _STD_H_


#ifndef FILE
/* XXX Bogus */
#  include <stdio.h>
#endif

/* FIXME: Split this file and adjust all callers to use new header files */
#if 0
#warning "Please use bool.h, wordlist.h or complex.h rather than cpstd.h"
#endif

#include "bool.h"
#include "wordlist.h"
#include "complex.h"

/* Externs defined in std.c */

extern char *getusername();
extern char *gethome();
extern char *tildexpand();
extern void printnum();
extern int cp_numdgt;
extern void fatal();

extern void cp_printword();

#endif /* _STD_H_*/
