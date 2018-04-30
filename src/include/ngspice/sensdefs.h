/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#ifndef ngspice_SENSDEFS_H
#define ngspice_SENSDEFS_H

#include "ngspice/jobdefs.h"

    /* structure used to describe an Adjoint Sensitivity analysis */

typedef struct st_sens SENS_AN;
typedef struct st_devsen DevSenList;
typedef struct st_modsen ModSenList;
typedef struct st_devsen ParamSenList;
typedef struct st_nodes Nodes;
typedef struct st_output output;

struct st_sens {
    int		JOBtype;
    JOB		*JOBnextJob;    /* pointer to next thing to do */
    char	*JOBname;      /* name of this job */

    DevSenList	*first;
    double	start_freq;
    double	stop_freq;

    int		step_type;
    int		n_freq_steps;

    CKTnode	*output_pos, *output_neg;
    IFuid	output_src;
    char	*output_name;
    int		output_volt;

    double	deftol;
    double	defperturb;
    unsigned int pct_flag :1;

};

struct st_output {
	int	type;
	int	pos, neg;
};

struct st_nodes {
	int	pos, neg;
};

struct st_paramsenlist {
	ParamSenList	*next;
	int		param_no;
	double		delta, tol;
};

struct st_modsenlist {
	ModSenList	*next;
	int		mod_no;
	ParamSenList	*first;
};

struct st_devsenlist {
	DevSenList	*next;
	int		dev_no;
	ModSenList	*first;
};

/* va, with prototypes */
extern int SENSask(CKTcircuit *,JOB *,int ,IFvalue *);
extern int SENSsetParam(CKTcircuit *,JOB *,int ,IFvalue *);
extern int sens_sens(CKTcircuit *,int);

enum {
    SENS_POS = 2,
    SENS_NEG,
    SENS_SRC,
    SENS_NAME,
};

enum {
    SENS_START = 10,
    SENS_STOP,
    SENS_STEPS,
    SENS_DECADE,
    SENS_OCTAVE,
    SENS_LINEAR,
    SENS_DC,
    SENS_DEFTOL,
    SENS_DEFPERTURB,
    SENS_DEVDEFTOL,
    SENS_DEVDEFPERT,
    SENS_TYPE,
    SENS_DEVICE,
};

enum {
    SENS_PARAM = 24,
    SENS_TOL,
    SENS_PERT,
};

#endif /*DEFS*/

