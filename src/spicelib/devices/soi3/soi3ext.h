/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White,
						 and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

extern int SOI3acLoad(GENmodel *,CKTcircuit*);
extern int SOI3ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int SOI3getic(GENmodel*,CKTcircuit*);
extern int SOI3load(GENmodel*,CKTcircuit*);
extern int SOI3mAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
extern int SOI3mParam(int,IFvalue*,GENmodel*);
extern void SOI3cap(double,double,double,
                    	double*,double*,double*,double*,
                    	double*,double*,double*,double*,
                    	double*,double*,double*,double*,
                    	double*,double*,double*,double*,double*,
                     double*,double*,double*,double*,double*,
                    	double*,double*,double*,double*,double*,
                    	double*,double*,double*,double*,double*);
extern void SOI3capEval(CKTcircuit*,double*,double*,
							double,double,double,double,double,
                     double,double,double,double,double,
                     double,double,double,double,double,
                     double,double,double,double,double,
                     double,double,double,double,double,
                     double*,double*,double*,double*,double*,
                     double*,double*,double*,double*,double*,
							double*,double*,double*,double*,double*,
                     double*,double*,double*,double*,double*,
                     double*,double*,double*,double*,double*,
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
extern int SOI3convTest(GENmodel*,CKTcircuit*);

/* extern int SOI3disto(int,GENmodel*,CKTcircuit*); */
extern int SOI3noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
