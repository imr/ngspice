/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1ext.h of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

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
