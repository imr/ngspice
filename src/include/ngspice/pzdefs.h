#ifndef ngspice_PZDEFS_H
#define ngspice_PZDEFS_H

    /* structure used to describe an PZ analysis to be performed */

#include "ngspice/jobdefs.h"
#include <math.h>
#include "ngspice/complex.h"
#include "ngspice/typedefs.h"

struct PZtrial {
    SPcomplex	s, f_raw, f_def;
    PZtrial	*next, *prev;
    int		mag_raw, mag_def;
    int		multiplicity;
    int		flags;
    int		seq_num;
    int		count;
};

struct PZAN {
    int JOBtype;
    JOB *JOBnextJob;
    IFuid JOBname;
    int PZin_pos;
    int PZin_neg;
    int PZout_pos;
    int PZout_neg;
    int PZinput_type;
    int PZwhich;
    int PZnumswaps;
    int PZbalance_col;
    int PZsolution_col;
    PZtrial *PZpoleList;
    PZtrial *PZzeroList;
    int PZnPoles;
    int PZnZeros;
    double *PZdrive_pptr;
    double *PZdrive_nptr;
};

#define PZ_DO_POLES	0x1
#define PZ_DO_ZEROS	0x2
#define PZ_IN_VOL	1
#define PZ_IN_CUR	2

enum {
    PZ_NODEI = 1,
    PZ_NODEG,
    PZ_NODEJ,
    PZ_NODEK,
    PZ_V,
    PZ_I,
    PZ_POL,
    PZ_ZER,
    PZ_PZ,
};

#endif
