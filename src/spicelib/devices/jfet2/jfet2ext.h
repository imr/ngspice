/**********
Based on jfetext.h
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/

extern int JFET2acLoad(GENmodel*,CKTcircuit*);
extern int JFET2ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int JFET2getic(GENmodel*,CKTcircuit*);
extern int JFET2load(GENmodel*,CKTcircuit*);
extern int JFET2mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int JFET2mParam(int,IFvalue*,GENmodel*);
extern int JFET2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int JFET2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int JFET2unsetup(GENmodel*,CKTcircuit*);
extern int JFET2temp(GENmodel*,CKTcircuit*);
extern int JFET2trunc(GENmodel*,CKTcircuit*,double*);
extern int JFET2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
