/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int VCCSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VCCSdelete(GENmodel*,IFuid,GENinstance**);
extern void VCCSdestroy(GENmodel**);
extern int VCCSload(GENmodel*,CKTcircuit*);
extern int VCCSmDelete(GENmodel**,IFuid,GENmodel*);
extern int VCCSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VCCSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VCCSsAcLoad(GENmodel*,CKTcircuit*);
extern int VCCSsLoad(GENmodel*,CKTcircuit*);
extern int VCCSsSetup(SENstruct*,GENmodel*);
extern void VCCSsPrint(GENmodel*,CKTcircuit*);
extern int VCCSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
#else /* stdc */
extern int VCCSask();
extern int VCCSdelete();
extern void VCCSdestroy();
extern int VCCSload();
extern int VCCSmDelete();
extern int VCCSparam();
extern int VCCSpzLoad();
extern int VCCSsAcLoad();
extern int VCCSsLoad();
extern int VCCSsSetup();
extern void VCCSsPrint();
extern int VCCSsetup();
#endif /* stdc */
