/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Paolo Nenzi 2002
File: bsim3v1sext.h
**********/

extern int BSIM3v1SacLoad(GENmodel *,CKTcircuit*);
extern int BSIM3v1Sask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3v1SconvTest(GENmodel *,CKTcircuit*);
extern int BSIM3v1Sdelete(GENmodel*,IFuid,GENinstance**);
extern void BSIM3v1Sdestroy(GENmodel**);
extern int BSIM3v1Sgetic(GENmodel*,CKTcircuit*);
extern int BSIM3v1Sload(GENmodel*,CKTcircuit*);
extern int BSIM3v1SmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3v1SmDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM3v1SmParam(int,IFvalue*,GENmodel*);
extern void BSIM3v1SmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3v1Sparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3v1SpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3v1Ssetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3v1Stemp(GENmodel*,CKTcircuit*);
extern int BSIM3v1Strunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3v1Snoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3v1Sunsetup(GENmodel *, CKTcircuit *);
