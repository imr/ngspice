/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hsm2ext.h

 Date : 2014.6.5

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

extern int HSM2acLoad(GENmodel *,CKTcircuit*);
extern int HSM2ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HSM2convTest(GENmodel *,CKTcircuit*);
extern int HSM2getic(GENmodel*,CKTcircuit*);
extern int HSM2load(GENmodel*,CKTcircuit*);
extern int HSM2mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int HSM2mDelete(GENmodel*);
extern int HSM2mParam(int,IFvalue*,GENmodel*);
extern void HSM2mosCap(CKTcircuit*, double, double, double, double*,
        double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int HSM2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int HSM2pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int HSM2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HSM2unsetup(GENmodel*,CKTcircuit*);
extern int HSM2temp(GENmodel*,CKTcircuit*);
extern int HSM2trunc(GENmodel*,CKTcircuit*,double*);
extern int HSM2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int HSM2soaCheck(CKTcircuit *, GENmodel *);
