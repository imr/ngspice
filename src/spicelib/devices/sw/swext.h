/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon M. Jacobs
Modified: 2000 AlansFixes
**********/

extern int SWacLoad(GENmodel *, CKTcircuit *);
extern int SWask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int SWload(GENmodel *, CKTcircuit *);
extern int SWmAsk(CKTcircuit *, GENmodel *, int, IFvalue *);
extern int SWmParam(int, IFvalue *, GENmodel *);
extern int SWparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int SWpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int SWsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int SWnoise(int, int, GENmodel *, CKTcircuit *, Ndata *, double *);
extern int SWtrunc(GENmodel *, CKTcircuit *, double *);
