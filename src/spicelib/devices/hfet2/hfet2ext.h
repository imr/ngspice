/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

extern int  HFET2acLoad(GENmodel*,CKTcircuit*);
extern int  HFET2ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int  HFET2getic(GENmodel*,CKTcircuit*);
extern int  HFET2load(GENmodel*,CKTcircuit*);
extern int  HFET2mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int  HFET2mParam(int,IFvalue*,GENmodel*);
extern int  HFET2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int  HFET2pzLoad(GENmodel*, CKTcircuit*, SPcomplex*);
extern int  HFET2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int  HFET2temp(GENmodel*,CKTcircuit*);
extern int  HFET2trunc(GENmodel*,CKTcircuit*,double*);
extern int  HFET2unsetup( GENmodel*,CKTcircuit*);
