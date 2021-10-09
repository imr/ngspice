#ifndef ngspice_PLOT_H
#define ngspice_PLOT_H

#include "ngspice/wordlist.h"
#include "ngspice/bool.h"
#include "ngspice/dvec.h"
#include "ngspice/hash.h"

struct ccom;

/* The information for a particular set of vectors that come from one
 * plot.  */
struct plot {
    char *pl_title;		/* The title card. */
    char *pl_date;		/* Date. */
    char *pl_name;		/* The plot name. */
    char *pl_typename;		/* Tran1, op2, etc. */
    struct dvec *pl_dvecs;	/* The data vectors in this plot. */
    struct dvec *pl_scale;	/* The "scale" for the rest of the vectors. */
    struct plot *pl_next;	/* List of plots. */
    NGHASHPTR pl_lookup_table;	/* for quick lookup of vectors */
    wordlist *pl_commands;	/* Commands to execute for this plot. */
    struct variable *pl_env;	/* The 'environment' for this plot. */
    struct ccom *pl_ccom;	/* The ccom struct for this plot. */
    bool pl_written;		/* Some or all of the vecs have been saved. */
    bool pl_lookup_valid;	/* vector lookup table valid */
    int pl_ndims;		/* Number of dimensions */
    int pl_xdim2d;		/* 2D Cider plot x dimension */
    int pl_ydim2d;		/* 2D Cider plot y dimension */
} ;


#endif
