/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
#ifndef __BJT2EXT_H
#define __BJT2EXT_H


extern int BJT2acLoad(GENmodel *,CKTcircuit*);
extern int BJT2ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BJT2convTest(GENmodel*,CKTcircuit*);
extern int BJT2delete(GENmodel*,IFuid,GENinstance**);
extern void BJT2destroy(GENmodel**);
extern int BJT2getic(GENmodel*,CKTcircuit*);
extern int BJT2load(GENmodel*,CKTcircuit*);
extern int BJT2mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int BJT2mDelete(GENmodel**,IFuid,GENmodel*);
extern int BJT2mParam(int,IFvalue*,GENmodel*);
extern int BJT2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BJT2pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BJT2sAcLoad(GENmodel*,CKTcircuit*);
extern int BJT2sLoad(GENmodel*,CKTcircuit*);
extern void BJT2sPrint(GENmodel*,CKTcircuit*);
extern int BJT2sSetup(SENstruct*,GENmodel*);
extern int BJT2sUpdate(GENmodel*,CKTcircuit*);
extern int BJT2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BJT2unsetup(GENmodel*,CKTcircuit*);
extern int BJT2temp(GENmodel*,CKTcircuit*);
extern int BJT2trunc(GENmodel*,CKTcircuit*,double*);
extern int BJT2disto(int,GENmodel*,CKTcircuit*);
extern int BJT2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BJT2dSetup(GENmodel*, register CKTcircuit*);

#endif
