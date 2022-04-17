/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

extern int EKVacLoad(GENmodel *,CKTcircuit*);
extern int EKVask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int EKVdelete(GENinstance*);
//extern void EKVdestroy(GENmodel**);
extern int EKVgetic(GENmodel*,CKTcircuit*);
extern int EKVload(GENmodel*,CKTcircuit*);
extern int EKVmAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
//extern int EKVmDelete(GENmodel*);
extern int EKVmParam(int,IFvalue*,GENmodel*);
extern int EKVparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int EKVsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int EKVunsetup(GENmodel*,CKTcircuit*);
extern int EKVtemp(GENmodel*,CKTcircuit*);
extern int EKVtrunc(GENmodel*,CKTcircuit*,double*);
extern int EKVconvTest(GENmodel*,CKTcircuit*);
extern int EKVnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
