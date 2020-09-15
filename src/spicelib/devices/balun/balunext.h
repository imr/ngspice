/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int BALUNask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int BALUNload(GENmodel*,CKTcircuit*);
extern int BALUNparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int BALUNpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BALUNsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BALUNunsetup(GENmodel*,CKTcircuit*);

