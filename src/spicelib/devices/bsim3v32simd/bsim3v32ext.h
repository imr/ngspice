/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Yuhua Cheng to use BSIM3v3 in Spice3f5 (Jan. 1997)
Modified by Paolo Nenzi 2002
File: bsim3ext.h
**********/

extern int BSIM3v32SIMDacLoad(GENmodel *,CKTcircuit*);
extern int BSIM3v32SIMDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3v32SIMDconvTest(GENmodel *,CKTcircuit*);
extern int BSIM3v32SIMDgetic(GENmodel*,CKTcircuit*);
extern int BSIM3v32SIMDload(GENmodel*,CKTcircuit*);
extern int BSIM3v32SIMDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3v32SIMDmDelete(GENmodel*);
extern int BSIM3v32SIMDmParam(int,IFvalue*,GENmodel*);
extern void BSIM3v32SIMDmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*);
extern int BSIM3v32SIMDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3v32SIMDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3v32SIMDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3v32SIMDtemp(GENmodel*,CKTcircuit*);
extern int BSIM3v32SIMDtrunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3v32SIMDnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3v32SIMDunsetup(GENmodel*,CKTcircuit*);
extern int BSIM3v32SIMDsoaCheck(CKTcircuit *, GENmodel *);
#ifdef BSIM3v32SIMD
extern int BSIM3v32SIMDloadSel(GENmodel*,CKTcircuit*);
#endif
