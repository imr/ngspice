/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Paolo Nenzi 2002
File: bsim3v1aext.h
**********/

extern int BSIM3v1AacLoad(GENmodel *,CKTcircuit*);
extern int BSIM3v1Aask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3v1AconvTest(GENmodel *,CKTcircuit*);
extern int BSIM3v1Adelete(GENmodel*,IFuid,GENinstance**);
extern void BSIM3v1Adestroy(GENmodel**);
extern int BSIM3v1Agetic(GENmodel*,CKTcircuit*);
extern int BSIM3v1Aload(GENmodel*,CKTcircuit*);
extern int BSIM3v1AmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3v1AmDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM3v1AmParam(int,IFvalue*,GENmodel*);
extern void BSIM3v1AmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3v1Aparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3v1ApzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3v1Asetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3v1Atemp(GENmodel*,CKTcircuit*);
extern int BSIM3v1Atrunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3v1Anoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3v1Aunsetup(GENmodel *, CKTcircuit *);
