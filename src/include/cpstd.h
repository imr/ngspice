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

typedef int bool;


#ifndef FILE
/* XXX Bogus */
#  include <stdio.h>
#endif

/* Doubly linked lists of words. */

struct wordlist {
    char *wl_word;
    struct wordlist *wl_next;
    struct wordlist *wl_prev;
} ;

typedef struct wordlist wordlist;

/* Complex numbers. */

struct _complex {   /* IBM portability... */
    double cx_real;
    double cx_imag;
} ;

typedef struct _complex complex;

#define realpart(cval)  ((struct _complex *) (cval))->cx_real
#define imagpart(cval)  ((struct _complex *) (cval))->cx_imag

/* Externs defined in std.c */

extern char *getusername();
extern char *gethome();
extern char *tildexpand();
extern char *printnum();
extern int cp_numdgt;
extern void fatal();

/* extern void setenv(); */

extern void cp_printword();

/* Externs from wlist.c */

extern char **wl_mkvec();
extern char *wl_flatten();
extern int wl_length();
extern void wl_free();
extern void wl_print();
extern void wl_sort();
extern wordlist *wl_append();
extern wordlist *wl_build();
extern wordlist *wl_copy();
extern wordlist *wl_range();
extern wordlist *wl_nthelem();
extern wordlist *wl_reverse();
extern wordlist *wl_splice();

#endif /* _STD_H_*/
