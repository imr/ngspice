/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon M. Jacobs
Modified: 2000 AlansFixes
**********/

extern int CSWask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int CSWacLoad(GENmodel *, CKTcircuit *);
extern int CSWload(GENmodel *, CKTcircuit *);
extern int CSWmAsk(CKTcircuit *, GENmodel *, int, IFvalue *);
extern int CSWmParam(int, IFvalue *, GENmodel *);
extern int CSWparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int CSWpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int CSWsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int CSWnoise(int, int, GENmodel *, CKTcircuit *, Ndata *, double *);
extern int CSWtrunc(GENmodel *, CKTcircuit *, double *);
