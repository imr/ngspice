/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int VCCSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VCCSload(GENmodel*,CKTcircuit*);
extern int VCCSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VCCSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VCCSsAcLoad(GENmodel*,CKTcircuit*);
extern int VCCSsLoad(GENmodel*,CKTcircuit*);
extern int VCCSsSetup(SENstruct*,GENmodel*);
extern void VCCSsPrint(GENmodel*,CKTcircuit*);
extern int VCCSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
