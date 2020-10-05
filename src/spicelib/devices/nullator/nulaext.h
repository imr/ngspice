/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int NULAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int NULAfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int NULAload(GENmodel*,CKTcircuit*);
extern int NULAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int NULAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int NULAunsetup(GENmodel*,CKTcircuit*);
extern int NULApzLoad(GENmodel*,CKTcircuit*,SPcomplex*);

/*
extern int NULAsAcLoad(GENmodel*,CKTcircuit*);
extern int NULAsLoad(GENmodel*,CKTcircuit*);
extern int NULAsSetup(SENstruct*,GENmodel*);
extern void NULAsPrint(GENmodel*,CKTcircuit*);
*/
