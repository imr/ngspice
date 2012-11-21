/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 7  REVISION : 0 ) Beta
 
 FILE : hsm2ext.h

 Date : 2012.10.25

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

extern int HSM2acLoad(GENmodel *,CKTcircuit*);
extern int HSM2ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HSM2convTest(GENmodel *,CKTcircuit*);
extern int HSM2delete(GENmodel*,IFuid,GENinstance**);
extern void HSM2destroy(GENmodel**);
extern int HSM2getic(GENmodel*,CKTcircuit*);
extern int HSM2load(GENmodel*,CKTcircuit*);
extern int HSM2mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int HSM2mDelete(GENmodel**,IFuid,GENmodel*);
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
