/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soipdext.h
**********/

#ifdef __STDC__
extern int B3SOIPDacLoad (GENmodel *, CKTcircuit *);
extern int B3SOIPDask (CKTcircuit *, GENinstance *, int, IFvalue *,
		       IFvalue *);
extern int B3SOIPDconvTest (GENmodel *, CKTcircuit *);
extern int B3SOIPDdelete (GENmodel *, IFuid, GENinstance **);
extern void B3SOIPDdestroy (GENmodel **);
extern int B3SOIPDgetic (GENmodel *, CKTcircuit *);
extern int B3SOIPDload (GENmodel *, CKTcircuit *);
extern int B3SOIPDmAsk (CKTcircuit *, GENmodel *, int, IFvalue *);
extern int B3SOIPDmDelete (GENmodel **, IFuid, GENmodel *);
extern int B3SOIPDmParam (int, IFvalue *, GENmodel *);
extern void B3SOIPDmosCap (CKTcircuit *, double, double, double, double,
			   double, double, double, double, double, double,
			   double, double, double, double, double, double,
			   double, double *, double *, double *, double *,
			   double *, double *, double *, double *, double *,
			   double *, double *, double *, double *, double *,
			   double *, double *);
extern int B3SOIPDparam (int, IFvalue *, GENinstance *, IFvalue *);
extern int B3SOIPDpzLoad (GENmodel *, CKTcircuit *, SPcomplex *);
extern int B3SOIPDsetup (SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int B3SOIPDtemp (GENmodel *, CKTcircuit *);
extern int B3SOIPDtrunc (GENmodel *, CKTcircuit *, double *);
extern int B3SOIPDnoise (int, int, GENmodel *, CKTcircuit *, Ndata *,
			 double *);
extern int B3SOIPDunsetup (GENmodel *, CKTcircuit *);

#else /* stdc */
extern int B3SOIPDacLoad ();
extern int B3SOIPDdelete ();
extern void B3SOIPDdestroy ();
extern int B3SOIPDgetic ();
extern int B3SOIPDload ();
extern int B3SOIPDmDelete ();
extern int B3SOIPDask ();
extern int B3SOIPDmAsk ();
extern int B3SOIPDconvTest ();
extern int B3SOIPDtemp ();
extern int B3SOIPDmParam ();
extern void B3SOIPDmosCap ();
extern int B3SOIPDparam ();
extern int B3SOIPDpzLoad ();
extern int B3SOIPDsetup ();
extern int B3SOIPDtrunc ();
extern int B3SOIPDnoise ();
extern int B3SOIPDunsetup ();

#endif /* stdc */
