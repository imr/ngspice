/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
License: 3-clause BSD
**********/
#ifndef __HICUMEXT_H
#define __HICUMEXT_H


extern int HICUMacLoad(GENmodel *,CKTcircuit*);
extern int HICUMask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HICUMconvTest(GENmodel*,CKTcircuit*);
extern int HICUMdelete(GENinstance*);
extern int HICUMgetic(GENmodel*,CKTcircuit*);
//extern int HICUMload(GENmodel*,CKTcircuit*);//moved to hicumL2.hpp
extern int HICUMmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int HICUMmParam(int,IFvalue*,GENmodel*);
extern int HICUMparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int HICUMpzLoad(GENmodel*, CKTcircuit*, SPcomplex*);
extern int HICUMsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HICUMunsetup(GENmodel*,CKTcircuit*);
// extern int HICUMtemp(GENmodel*,CKTcircuit*); // moved to hicum2temp.hpp
extern int HICUMtrunc(GENmodel*,CKTcircuit*,double*);
extern int HICUMnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int HICUMsoaCheck(CKTcircuit *, GENmodel *);

#endif
