/**********
Copyright 2010 Regents of the University of California.  All rights reserved.
Author: 2005 Hui Wan (based on Samuel Fung's b3soiext.h)
Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
Authors: 2009- Tanvir Morshed, Ali Niknejad, Chenming Hu.
Authors: 2010- Tanvir Morshed, Ali Niknejad, Chenming Hu.
File: b4soiext.h
**********/

extern int B4SOIacLoad(GENmodel *,CKTcircuit*);
extern int B4SOIask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B4SOIconvTest(GENmodel *,CKTcircuit*);
extern int B4SOIgetic(GENmodel*,CKTcircuit*);
extern int B4SOIload(GENmodel*,CKTcircuit*);
extern int B4SOImAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B4SOImDelete(GENmodel*);
extern int B4SOImParam(int,IFvalue*,GENmodel*);
extern void B4SOImosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B4SOIparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int B4SOIpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B4SOIsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B4SOItemp(GENmodel*,CKTcircuit*);
extern int B4SOItrunc(GENmodel*,CKTcircuit*,double*);
extern int B4SOInoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int B4SOIunsetup(GENmodel*,CKTcircuit*);
extern int B4SOIsoaCheck(CKTcircuit *, GENmodel *);
