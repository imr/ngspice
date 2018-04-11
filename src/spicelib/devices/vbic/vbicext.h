/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/
#ifndef __VBICEXT_H
#define __VBICEXT_H


extern int VBICacLoad(GENmodel *,CKTcircuit*);
extern int VBICask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int VBICconvTest(GENmodel*,CKTcircuit*);
extern int VBICdelete(GENinstance*);
extern int VBICgetic(GENmodel*,CKTcircuit*);
extern int VBICload(GENmodel*,CKTcircuit*);
extern int VBICmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int VBICmParam(int,IFvalue*,GENmodel*);
extern int VBICparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VBICpzLoad(GENmodel*, CKTcircuit*, SPcomplex*);
extern int VBICsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VBICunsetup(GENmodel*,CKTcircuit*);
extern int VBICtemp(GENmodel*,CKTcircuit*);
extern int VBICtrunc(GENmodel*,CKTcircuit*,double*);
extern int VBICnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int VBICsoaCheck(CKTcircuit *, GENmodel *);

#endif
