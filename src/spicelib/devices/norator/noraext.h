/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int NORAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int NORAfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int NORAload(GENmodel*,CKTcircuit*);
extern int NORAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int NORAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int NORAunsetup(GENmodel*,CKTcircuit*);
extern int NORApzLoad(GENmodel*,CKTcircuit*,SPcomplex*);

/*
extern int NORAsAcLoad(GENmodel*,CKTcircuit*);
extern int NORAsLoad(GENmodel*,CKTcircuit*);
extern int NORAsSetup(SENstruct*,GENmodel*);
extern void NORAsPrint(GENmodel*,CKTcircuit*);
*/
