/***********************************************************************
 HiSIM v1.1.0
 File: hsm1ext.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

extern int HSM1acLoad(GENmodel *,CKTcircuit*);
extern int HSM1ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HSM1convTest(GENmodel *,CKTcircuit*);
extern int HSM1delete(GENmodel*,IFuid,GENinstance**);
extern void HSM1destroy(GENmodel**);
extern int HSM1getic(GENmodel*,CKTcircuit*);
extern int HSM1load(GENmodel*,CKTcircuit*);
extern int HSM1mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int HSM1mDelete(GENmodel**,IFuid,GENmodel*);
extern int HSM1mParam(int,IFvalue*,GENmodel*);
extern void HSM1mosCap(CKTcircuit*, double, double, double, double*,
        double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int HSM1param(int,IFvalue*,GENinstance*,IFvalue*);
extern int HSM1pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int HSM1setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HSM1unsetup(GENmodel*,CKTcircuit*);
extern int HSM1temp(GENmodel*,CKTcircuit*);
extern int HSM1trunc(GENmodel*,CKTcircuit*,double*);
extern int HSM1noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
