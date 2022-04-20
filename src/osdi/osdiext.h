/*
 * Copyright© 2022 SemiMod UG. All rights reserved.
 */

#pragma once

#include "ngspice/gendefs.h"
#include "ngspice/smpdefs.h"
#include <stdint.h>

#include "ngspice/osdiitf.h"

extern int OSDImParam(int, IFvalue *, GENmodel *);
extern int OSDIparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int OSDIsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int OSDIunsetup(GENmodel *, CKTcircuit *);
extern int OSDIask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int OSDIload(GENmodel *, CKTcircuit *);
extern int OSDItemp(GENmodel *, CKTcircuit *);
extern int OSDIacLoad(GENmodel *, CKTcircuit *);
extern int OSDItrunc(GENmodel *, CKTcircuit *, double *);
extern int OSDIpzLoad(GENmodel*, CKTcircuit*, SPcomplex*);

/* extern int OSDIconvTest(GENmodel*,CKTcircuit*); */
/* extern int OSDImDelete(GENmodel*); */
/* extern int OSDIgetic(GENmodel*,CKTcircuit*); */
/* extern int OSDImAsk(CKTcircuit*,GENmodel*,int,IFvalue*); */
/* extern int OSDInoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*); */
/* extern int OSDIsoaCheck(CKTcircuit *, GENmodel *); */
