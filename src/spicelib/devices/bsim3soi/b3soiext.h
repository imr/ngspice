/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soiext.h
Modified by Paolo Nenzi 2002
**********/

extern int B3SOIacLoad(GENmodel *,CKTcircuit*);
extern int B3SOIask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B3SOIconvTest(GENmodel *,CKTcircuit*);
extern int B3SOIdelete(GENmodel*,IFuid,GENinstance**);
extern void B3SOIdestroy(GENmodel**);
extern int B3SOIgetic(GENmodel*,CKTcircuit*);
extern int B3SOIload(GENmodel*,CKTcircuit*);
extern int B3SOImAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B3SOImDelete(GENmodel**,IFuid,GENmodel*);
extern int B3SOImParam(int,IFvalue*,GENmodel*);
extern void B3SOImosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B3SOIparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int B3SOIpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B3SOIsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B3SOItemp(GENmodel*,CKTcircuit*);
extern int B3SOItrunc(GENmodel*,CKTcircuit*,double*);
extern int B3SOInoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int B3SOIunsetup(GENmodel*,CKTcircuit*);
