/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "devdefs.h"
#include "ifsim.h"
#include <stdio.h>
#include "inddefs.h"
#include "suffix.h"

IFparm INDpTable[] = { /* parameters */ 
 IOPAP("inductance",IND_IND,    IF_REAL,"Inductance of inductor"),
 IOPAU("ic",        IND_IC,     IF_REAL,"Initial current through inductor"),
 IP( "sens_ind", IND_IND_SENS,IF_FLAG,
        "flag to request sensitivity WRT inductance"),
 OP( "flux",      IND_FLUX,   IF_REAL,"Flux through inductor"),
 OP( "v",         IND_VOLT,   IF_REAL,"Terminal voltage of inductor"),
 OPR("volt",      IND_VOLT,   IF_REAL,""),
 OP( "i",   	  IND_CURRENT,IF_REAL,"Current through the inductor"),
 OPR( "current",  IND_CURRENT,IF_REAL,""),
 OP( "p",         IND_POWER,  IF_REAL,
        "instantaneous power dissipated by the inductor"),
 OPU( "sens_dc", IND_QUEST_SENS_DC,     IF_REAL, "dc sensitivity sensitivity"),
 OPU( "sens_real", IND_QUEST_SENS_REAL, IF_REAL, "real part of ac sensitivity"),
 OPU( "sens_imag", IND_QUEST_SENS_IMAG, IF_REAL, 
        "dc sensitivity and imag part of ac sensitivty"),
 OPU( "sens_mag",  IND_QUEST_SENS_MAG,  IF_REAL, "sensitivity of AC magnitude"),
 OPU( "sens_ph",   IND_QUEST_SENS_PH,   IF_REAL, "sensitivity of AC phase"),
 OPU( "sens_cplx", IND_QUEST_SENS_CPLX, IF_COMPLEX,    "ac sensitivity")
};

char *INDnames[] = {
    "L+",
    "L-"
};


int	INDnSize = NUMELEMS(INDnames);
int	INDpTSize = NUMELEMS(INDpTable);
int	INDmPTSize = 0;
int	INDiSize = sizeof(INDinstance);
int	INDmSize = sizeof(INDmodel);

#ifdef MUTUAL

IFparm MUTpTable[] = { /* parameters */ 
 IOPAP( "k", MUT_COEFF, IF_REAL    , "Mutual inductance"),
 IOPR( "coefficient", MUT_COEFF, IF_REAL    , ""),
 IOP( "inductor1", MUT_IND1,  IF_INSTANCE, "First coupled inductor"),
 IOP( "inductor2", MUT_IND2,  IF_INSTANCE, "Second coupled inductor"),
 IP( "sens_coeff", MUT_COEFF_SENS, IF_FLAG,    
        "flag to request sensitivity WRT coupling factor"),
 OPU( "sens_dc",   MUT_QUEST_SENS_DC,   IF_REAL, "dc sensitivity "),
 OPU( "sens_real", MUT_QUEST_SENS_REAL, IF_REAL, "real part of ac sensitivity"),
 OPU( "sens_imag", MUT_QUEST_SENS_IMAG, IF_REAL, 
        "dc sensitivity and imag part of ac sensitivty"),
 OPU( "sens_mag", MUT_QUEST_SENS_MAG,  IF_REAL, "sensitivity of AC magnitude"),
 OPU( "sens_ph",  MUT_QUEST_SENS_PH,   IF_REAL, "sensitivity of AC phase"),
 OPU( "sens_cplx",  MUT_QUEST_SENS_CPLX, IF_COMPLEX,  "ac sensitivity")
};

int	MUTnSize = NUMELEMS(INDnames);
int	MUTpTSize = NUMELEMS(MUTpTable);
int	MUTmPTSize = 0;
int	MUTiSize = sizeof(INDinstance);
int	MUTmSize = sizeof(INDmodel);

#endif /*MUTUAL*/
