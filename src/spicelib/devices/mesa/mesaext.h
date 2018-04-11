/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

extern int MESAacLoad(GENmodel*,CKTcircuit*);
extern int MESAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MESAgetic(GENmodel*,CKTcircuit*);
extern int MESAload(GENmodel*,CKTcircuit*);
extern int MESAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int MESAmParam(int,IFvalue*,GENmodel*);
extern int MESAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int MESApzLoad(GENmodel*,CKTcircuit*, SPcomplex*);
extern int MESAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MESAtemp(GENmodel*,CKTcircuit*);
extern int MESAtrunc(GENmodel*,CKTcircuit*,double*);
extern int MESAunsetup(GENmodel*,CKTcircuit*);
