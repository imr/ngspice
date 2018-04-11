/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: bsim3v0ext.h
**********/

extern int BSIM3v0acLoad(GENmodel *,CKTcircuit*);
extern int BSIM3v0ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3v0convTest(GENmodel *,CKTcircuit*);
extern int BSIM3v0getic(GENmodel*,CKTcircuit*);
extern int BSIM3v0load(GENmodel*,CKTcircuit*);
extern int BSIM3v0mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3v0mParam(int,IFvalue*,GENmodel*);
extern void BSIM3v0mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3v0param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3v0pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3v0setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3v0temp(GENmodel*,CKTcircuit*);
extern int BSIM3v0trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3v0noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int  BSIM3v0unsetup(GENmodel *, CKTcircuit *);

