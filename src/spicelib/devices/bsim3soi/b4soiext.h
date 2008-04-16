/**********
Copyright 2005 Regents of the University of California.  All rights reserved.
Author: 2005 Hui Wan (based on Samuel Fung's b3soiext.h)
File: b4soiext.h
**********/

#ifdef __STDC__
extern int B4SOIacLoad(GENmodel *,CKTcircuit*);
extern int B4SOIask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B4SOIconvTest(GENmodel *,CKTcircuit*);
extern int B4SOIdelete(GENmodel*,IFuid,GENinstance**);
extern void B4SOIdestroy(GENmodel**);
extern int B4SOIgetic(GENmodel*,CKTcircuit*);
extern int B4SOIload(GENmodel*,CKTcircuit*);
extern int B4SOImAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B4SOImDelete(GENmodel**,IFuid,GENmodel*);
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

#else /* stdc */
extern int B4SOIacLoad();
extern int B4SOIdelete();
extern void B4SOIdestroy();
extern int B4SOIgetic();
extern int B4SOIload();
extern int B4SOImDelete();
extern int B4SOIask();
extern int B4SOImAsk();
extern int B4SOIconvTest();
extern int B4SOItemp();
extern int B4SOImParam();
extern void B4SOImosCap();
extern int B4SOIparam();
extern int B4SOIpzLoad();
extern int B4SOIsetup();
extern int B4SOItrunc();
extern int B4SOInoise();
extern int B4SOIunsetup();

#endif /* stdc */

