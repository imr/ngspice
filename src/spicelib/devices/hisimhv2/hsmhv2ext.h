/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvext.h

 DATE : 2014.6.11

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

extern int HSMHV2acLoad(GENmodel *,CKTcircuit*);
extern int HSMHV2ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HSMHV2convTest(GENmodel *,CKTcircuit*);
extern int HSMHV2getic(GENmodel*,CKTcircuit*);
extern int HSMHV2load(GENmodel*,CKTcircuit*);
extern int HSMHV2mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int HSMHV2mParam(int,IFvalue*,GENmodel*);
extern void HSMHV2mosCap(CKTcircuit*, double, double, double, double*,
        double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int HSMHV2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int HSMHV2pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int HSMHV2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HSMHV2unsetup(GENmodel*,CKTcircuit*);
extern int HSMHV2temp(GENmodel*,CKTcircuit*);
extern int HSMHV2trunc(GENmodel*,CKTcircuit*,double*);
extern int HSMHV2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int HSMHV2soaCheck(CKTcircuit *, GENmodel *);
