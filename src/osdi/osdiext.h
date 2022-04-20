/* 
 * This file is part of the OSDI component of NGSPICE.
 * Copyright© 2022 SemiMod GmbH.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Pascal Kuthe <pascal.kuthe@semimod.de>
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
