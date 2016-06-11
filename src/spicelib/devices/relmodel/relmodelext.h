/**********
Author: Francesco Lannutti - July 2015
**********/

extern int RELMODELcalculateAging (GENinstance *, int, CKTcircuit *, double, double, unsigned int) ;
extern int RELMODELcalculateFitting (unsigned int, unsigned int, double, double *, double *, double *) ;
extern int RELMODELmAsk (CKTcircuit *, GENmodel *, int, IFvalue *) ;
extern int RELMODELmDelete (GENmodel **, IFuid, GENmodel *) ;
extern int RELMODELmParam (int, IFvalue *, GENmodel *) ;
extern int RELMODELsetup (SMPmatrix *, GENmodel *, CKTcircuit *, int *) ;
