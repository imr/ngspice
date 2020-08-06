/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Yuhua Cheng to use BSIM3v3 in Spice3f5 (Jan. 1997)
File: bsim3ext.h
**********/

extern int BSIM3SIMDacLoad(GENmodel *,CKTcircuit*);
extern int BSIM3SIMDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3SIMDconvTest(GENmodel *,CKTcircuit*);
extern int BSIM3SIMDgetic(GENmodel*,CKTcircuit*);
extern int BSIM3SIMDload(GENmodel*,CKTcircuit*);
extern int BSIM3SIMDloadSel(GENmodel*,CKTcircuit*);
extern int BSIM3SIMDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3SIMDmDelete(GENmodel*);
extern int BSIM3SIMDmParam(int,IFvalue*,GENmodel*);
extern void BSIM3SIMDmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3SIMDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3SIMDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3SIMDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3SIMDtemp(GENmodel*,CKTcircuit*);
extern int BSIM3SIMDtrunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3SIMDnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3SIMDunsetup(GENmodel*,CKTcircuit*);
extern int BSIM3SIMDsoaCheck(CKTcircuit *, GENmodel *);
