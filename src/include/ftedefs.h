/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 *
 * General front end stuff.
 */
#ifndef FTEdefs_h
#define FTEdefs_h

#define DEF_WIDTH   80	/* Line printer width. */
#define DEF_HEIGHT  60  /* Line printer height. */
#define IPOINTMIN   20  /* When we start plotting incremental plots. */
#include "fteparse.h"
#include "fteinp.h"

struct save_info {
    char	*name;
    IFuid	*analysis;
    int		used;
};

#define mylog10(xx) (((xx) > 0.0) ? log10(xx) : (- log10(HUGE)))

#include "fteext.h"

#endif /* FTEdefs_h */
