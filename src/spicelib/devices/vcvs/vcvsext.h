/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int VCVSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VCVSdelete(GENinstance*);
extern void VCVSdestroy(void);
extern int VCVSfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int VCVSload(GENmodel*,CKTcircuit*);
extern int VCVSmDelete(GENmodel*);
extern int VCVSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VCVSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VCVSsAcLoad(GENmodel*,CKTcircuit*);
extern int VCVSsLoad(GENmodel*,CKTcircuit*);
extern int VCVSsSetup(SENstruct*,GENmodel*);
extern void VCVSsPrint(GENmodel*,CKTcircuit*);
extern int VCVSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VCVSunsetup(GENmodel*,CKTcircuit*);

#ifdef KLU
extern int VCVSbindCSC (GENmodel*, CKTcircuit*) ;
extern int VCVSbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int VCVSbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
