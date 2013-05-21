/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 3 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvext.h

 DATE : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

extern int HSMHVacLoad(GENmodel *,CKTcircuit*);
extern int HSMHVask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HSMHVconvTest(GENmodel *,CKTcircuit*);
extern int HSMHVdelete(GENmodel*,IFuid,GENinstance**);
extern void HSMHVdestroy(GENmodel**);
extern int HSMHVgetic(GENmodel*,CKTcircuit*);
extern int HSMHVload(GENmodel*,CKTcircuit*);
extern int HSMHVmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int HSMHVmDelete(GENmodel**,IFuid,GENmodel*);
extern int HSMHVmParam(int,IFvalue*,GENmodel*);
extern void HSMHVmosCap(CKTcircuit*, double, double, double, double*,
        double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int HSMHVparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int HSMHVpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int HSMHVsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HSMHVunsetup(GENmodel*,CKTcircuit*);
extern int HSMHVtemp(GENmodel*,CKTcircuit*);
extern int HSMHVtrunc(GENmodel*,CKTcircuit*,double*);
extern int HSMHVnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int HSMHVnodeIsNonLinear (GENmodel *, CKTcircuit *) ;
