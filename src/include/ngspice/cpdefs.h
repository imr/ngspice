/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * General stuff for the C-shell parser.
 */

/* Standard definitions */
#ifndef ngspice_CPDEFS_H
#define ngspice_CPDEFS_H

#include "ngspice/cpstd.h"

/* Information about spice commands. */

struct comm {
    /* The name of the command. */
    char *co_comname;

    /* The function that handles the command. */
    void (*co_func) (wordlist *wl);

    /* These can't be used from nutmeg. */
    bool co_spiceonly;

    /* Is this a "major" command? */
    bool co_major;

    /* Bitmasks for command completion. */
    long co_cctypes[4];

    /* print help message on this environment mask */
    unsigned int co_env;

    /* minimum number of arguments required */
    int co_minargs;

    /* maximum number of arguments allowed */
    int co_maxargs;

    /* The fn that prompts the user. */
    void (*co_argfn)(const wordlist *wl, const struct comm *command);

    /* When these are printed, printf(string, av[0]) .. */
    char *co_help;
};

#define LOTS        1000
#define NLOTS       10000

/* The history list. Things get put here after the first (basic) parse.
 * The word list will change later, so be sure to copy it.
 */

struct histent {
    int hi_event;
    wordlist *hi_wlist;
    struct histent *hi_next;
    struct histent *hi_prev;
};



/* The values returned by cp_usrset(). */

enum {
    US_OK = 1,      /* Either not relevant or nothing special. */
    US_READONLY,    /* Complain and don't set this var. */
    US_DONTRECORD,  /* OK, but don't keep track of this one. */
    US_SIMVAR,      /* OK, recorded in options struct */
    US_NOSIMVAR,    /* Not OK, simulation param but circuit not loaded */
};

/* Aliases. These will be expanded if the word is the first in an input
 * line. The substitution string may contain arg selectors.
 */

struct alias {
    char *al_name;        /* The word to be substituted for. */
    wordlist *al_text;  /* What to substitute for it. */
    struct alias *al_next;
    struct alias *al_prev;
} ;


#define CT_ALIASES  1
#define CT_LABEL    15

/* Get all the extern definitions... */

#include "ngspice/cpextern.h"

#endif
