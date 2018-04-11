/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int MOS6acLoad(GENmodel *,CKTcircuit*);
extern int MOS6ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS6delete(GENinstance*);
extern int MOS6getic(GENmodel*,CKTcircuit*);
extern int MOS6load(GENmodel*,CKTcircuit*);
extern int MOS6mAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
extern int MOS6mParam(int,IFvalue*,GENmodel*);
extern int MOS6param(int,IFvalue*,GENinstance*,IFvalue*);
extern int MOS6pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MOS6setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MOS6unsetup(GENmodel*,CKTcircuit*);
extern int MOS6temp(GENmodel*,CKTcircuit*);
extern int MOS6trunc(GENmodel*,CKTcircuit*,double*);
extern int MOS6convTest(GENmodel*,CKTcircuit*);
