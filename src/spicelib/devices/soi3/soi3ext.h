/**********
STAG version 2.6
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

#ifdef __STDC__
extern int SOI3acLoad(GENmodel *,CKTcircuit*);
extern int SOI3ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int SOI3delete(GENmodel*,IFuid,GENinstance**);
extern void SOI3destroy(GENmodel**);
extern int SOI3getic(GENmodel*,CKTcircuit*);
extern int SOI3load(GENmodel*,CKTcircuit*);
extern int SOI3mAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
extern int SOI3mDelete(GENmodel**,IFuid,GENmodel*);
extern int SOI3mParam(int,IFvalue*,GENmodel*);
extern void SOI3cap(double,double,double,
                    double*,double*,double*,double*,double*,double*,
                    double*,double*,double*,double*,
                    double*,double*,double*,double*,double*,double*,double*,double*,
                    double*,double*,double*,double*,double*,double*,double*,double*,double*);
extern void SOI3capEval(CKTcircuit*,double*,double*,
                        double,double,double,double,
                        double,double,double,double,
                        double,double,double,double,
                        double,double,double,double,double,
                        double,double,
                        double*,double*,double*,double*,
                        double*,double*,double*,double*,
                        double*,double*,double*,double*,
                        double*,double*,double*,double*,double*,
                        double*,double*,double*,
                        double*,double*,
                        double*,double*,double*,double*,double*);
extern int SOI3param(int,IFvalue*,GENinstance*,IFvalue*);
/* extern int SOI3pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);  |
extern int SOI3sAcLoad(GENmodel*,CKTcircuit*);               |
extern int SOI3sLoad(GENmodel*,CKTcircuit*);                 | Ignored
extern void SOI3sPrint(GENmodel*,CKTcircuit*);               |
extern int SOI3sSetup(SENstruct*,GENmodel*);                 |
extern int SOI3sUpdate(GENmodel*,CKTcircuit*);               | */
extern int SOI3setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int SOI3unsetup(GENmodel*,CKTcircuit*);
extern int SOI3temp(GENmodel*,CKTcircuit*);
extern int SOI3trunc(GENmodel*,CKTcircuit*,double*);

#ifdef SIMETRIX_VERSION
/* NTL modification 27.4.98 - add lastAttempt */
extern int SOI3convTest(GENmodel*,CKTcircuit*,int lastAttempt);
/* end NTL modification */
#else /* SIMETRIX_VERSION */
extern int SOI3convTest(GENmodel*,CKTcircuit*);
#endif /* SIMETRIX_VERSION */

/* extern int SOI3disto(int,GENmodel*,CKTcircuit*); */
extern int SOI3noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int SOI3acLoad();
extern int SOI3ask();
extern int SOI3delete();
extern void SOI3destroy();
extern int SOI3getic();
extern int SOI3load();
extern int SOI3mAsk();
extern int SOI3mDelete();
extern int SOI3mParam();
extern void SOI3cap();
extern void SOI3capEval();
extern int SOI3param();
/* extern int SOI3pzLoad(); |
extern int SOI3sAcLoad();   |
extern int SOI3sLoad();     |  ignored
extern void SOI3sPrint();   |
extern int SOI3sSetup();    |
extern int SOI3sUpdate();  */
extern int SOI3setup();
extern int SOI3unsetup();
extern int SOI3temp();
extern int SOI3trunc();
extern int SOI3convTest();
/* extern int SOI3disto(); */
extern int SOI3noise();
#endif /* stdc */
