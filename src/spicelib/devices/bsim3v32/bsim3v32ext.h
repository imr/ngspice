/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Yuhua Cheng to use BSIM3v3 in Spice3f5 (Jan. 1997)
Modified by Paolo Nenzi 2002
File: bsim3ext.h
**********/

extern int BSIM3v32acLoad(GENmodel *,CKTcircuit*);
extern int BSIM3v32ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3v32convTest(GENmodel *,CKTcircuit*);
extern int BSIM3v32getic(GENmodel*,CKTcircuit*);
extern int BSIM3v32load(GENmodel*,CKTcircuit*);
extern int BSIM3v32mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3v32mDelete(GENmodel*);
extern int BSIM3v32mParam(int,IFvalue*,GENmodel*);
extern void BSIM3v32mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*);
extern int BSIM3v32param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3v32pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3v32setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3v32temp(GENmodel*,CKTcircuit*);
extern int BSIM3v32trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3v32noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3v32unsetup(GENmodel*,CKTcircuit*);
extern int BSIM3v32soaCheck(CKTcircuit *, GENmodel *);
