
/* RCS Info: $Revision$ on $Date$
 *         $Source$
 * Copyright (c) 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
 *
 * Structures used for representing spice data in nutmeg.
 */

#ifndef FTEdata_h
#define FTEdata_h

#include "cpstd.h"      /* for struct complex */
#include "fteconst.h"

/* A (possibly multi-dimensional) data vector.  The data is represented
 * internally by a 1-d array.  The number of dimensions and the size
 * of each dimension is recorded, along with v_length, the total size of
 * the array.  If the dimensionality is 0 or 1, v_length is significant
 * instead of v_numdims and v_dims, and the vector is handled in the old
 * manner.
 */

#define MAXDIMS 8

struct dvec {
    char *v_name;		/* Same as so_vname. */
    int v_type;			/* Same as so_vtype. */
    short v_flags;		/* Flags (a combination of VF_*). */
    double *v_realdata;		/* Real data. */
    complex *v_compdata;	/* Complex data. */
    double v_minsignal;		/* Minimum value to plot. */
    double v_maxsignal;		/* Maximum value to plot. */
    GRIDTYPE v_gridtype;	/* One of GRID_*. */
    PLOTTYPE v_plottype;	/* One of PLOT_*. */
    int v_length;		/* Length of the vector. */
    int v_rlength;		/* How much space we really have. */
    int v_outindex;		/* Index if writedata is building the
				   vector. */
    int v_linestyle;		/* What line style we are using. */
    int v_color;		/* What color we are using. */
    char *v_defcolor;		/* The name of a color to use. */
    int v_numdims;		/* How many dims -- 0 = scalar (len = 1). */
    int v_dims[MAXDIMS];	/* The actual size in each dimension. */
    struct plot *v_plot;	/* The plot structure (if it has one). */
    struct dvec *v_next;	/* Link for list of plot vectors. */
    struct dvec *v_link2;	/* Extra link for things like print. */
    struct dvec *v_scale;	/* If this has a non-standard scale... */
} ;

#define isreal(v)   ((v)->v_flags & VF_REAL)
#define iscomplex(v)    ((v)->v_flags & VF_COMPLEX)

/* The information for a particular set of vectors that come from one
 * plot.
 */

struct plot {
    char *pl_title;     /* The title card. */
    char *pl_date;      /* Date. */
    char *pl_name;      /* The plot name. */
    char *pl_typename;  /* Tran1, op2, etc. */
    struct dvec *pl_dvecs;  /* The data vectors in this plot. */
    struct dvec *pl_scale;  /* The "scale" for the rest of the vectors. */
    struct plot *pl_next;   /* List of plots. */
    wordlist *pl_commands;  /* Commands to execute for this plot. */
    struct variable *pl_env;/* The 'environment' for this plot. */
    char *pl_ccom;      /* The ccom struct for this plot. */
    bool pl_written;    /* Some or all of the vecs have been saved. */
    int pl_ndims;    /* Number of dimensions */
} ;

#endif /* FTEdata_h */
