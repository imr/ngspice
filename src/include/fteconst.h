/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Vector types. */

#ifndef _CONSTANTS_
#define _CONSTANTS_

#define SV_NOTYPE   0
#define SV_TIME  1
#define SV_FREQUENCY    2
#define SV_VOLTAGE  3
#define SV_CURRENT  4
#define SV_OUTPUT_N_DENS 5
#define SV_OUTPUT_NOISE  6
#define SV_INPUT_N_DENS 7
#define SV_INPUT_NOISE  8
#define SV_POLE     9
#define SV_ZERO     10
#define SV_SPARAM   11

#ifdef notdef
#define SV_OUTPUT_NOISE 5
#define SV_INPUT_NOISE  6
#define SV_HD2    7
#define SV_HD3    8
#define SV_DIM2  9
#define SV_SIM2  10
#define SV_DIM3  11
#define SV_POLE     12
#define SV_ZERO     13
#define SV_SPARAM   14
#endif

/* Dvec flags. */

#define VF_REAL     (1 << 0) /* The data is real. */
#define VF_COMPLEX  (1 << 1) /* The data is complex. */
#define VF_ACCUM    (1 << 2) /* writedata should save this vector. */
#define VF_PLOT     (1 << 3) /* writedata should incrementally plot it. */
#define VF_PRINT    (1 << 4) /* writedata should print this vector. */
#define VF_MINGIVEN (1 << 5) /* The v_minsignal value is valid. */
#define VF_MAXGIVEN (1 << 6) /* The v_maxsignal value is valid. */
#define VF_PERMANENT    (1 << 7) /* Don't garbage collect this vector. */

/* Grid types. */

/*
#define GRID_NONE   0
#define GRID_LIN    1
#define GRID_LOGLOG 2
#define GRID_XLOG   3
#define GRID_YLOG   4
#define GRID_POLAR  5
#define GRID_SMITH  6
*/

/* SMITHGRID is only a smith grid, SMITH transforms the data */
typedef enum {
    GRID_NONE = 0, GRID_LIN = 1, GRID_LOGLOG = 2, GRID_XLOG = 3,
    GRID_YLOG = 4, GRID_POLAR = 5, GRID_SMITH = 6, GRID_SMITHGRID = 7
} GRIDTYPE;

/* Plot types. */

/*
#define PLOT_LIN    0
#define PLOT_COMB   1
#define PLOT_POINT  2
*/

typedef enum {
    PLOT_LIN = 0, PLOT_COMB = 1, PLOT_POINT = 2
} PLOTTYPE;

/* The types for command completion keywords. Note that these constants
 * are built into cmdtab.c, so DON'T change them unless you want to
 * change all of the bitmasks in cp_coms.
 * Note that this is spice- and nutmeg- dependent.
 */

#define CT_FILENAME     0
#define CT_CKTNAMES     2
#define CT_COMMANDS     3
#define CT_DBNUMS       4
#define CT_DEVNAMES     5
#define CT_LISTINGARGS  6
#define CT_NODENAMES    7
#define CT_PLOT         8
#define CT_PLOTKEYWORDS 9
#define CT_RUSEARGS     10
#define CT_STOPARGS     11
#define CT_UDFUNCS      12
#define CT_VARIABLES    13
#define CT_VECTOR       14
#define CT_TYPENAMES    16

#endif

