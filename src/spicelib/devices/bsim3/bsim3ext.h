/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Yuhua Cheng to use BSIM3v3 in Spice3f5 (Jan. 1997)
File: bsim3ext.h
**********/

extern int BSIM3acLoad(GENmodel *,CKTcircuit*);
extern int BSIM3ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3convTest(GENmodel *,CKTcircuit*);
extern int BSIM3getic(GENmodel*,CKTcircuit*);
extern int BSIM3load(GENmodel*,CKTcircuit*);
extern int BSIM3mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3mDelete(GENmodel*);
extern int BSIM3mParam(int,IFvalue*,GENmodel*);
extern void BSIM3mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3temp(GENmodel*,CKTcircuit*);
extern int BSIM3trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3unsetup(GENmodel*,CKTcircuit*);
extern int BSIM3soaCheck(CKTcircuit *, GENmodel *);
