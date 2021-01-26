/*************
 * Header file for outitf.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_OUTITF_H
#define ngspice_OUTITF_H

#include "ngspice/typedefs.h"


typedef struct dataDesc {
    char *name;                 /* The name of the vector. */
    int type;                   /* The type. */
    GRIDTYPE gtype;             /* default plot scale */
    bool regular;               /* Is this given to us? */
    int outIndex;               /* If regular then the index. */
    char *specName;             /* The device name if special. */
    char *specParamName;        /* The parameter name if special. */
    int specIndex;              /* For sensitivity, if special. */
    int specType;
    GENinstance *specFast;
    int refIndex;               /* The index of our ref vector. */
    struct dvec *vec;
} dataDesc;


struct runDesc {
    void *analysis;
    CKTcircuit *circuit;
    char *name;
    char *type;
    int numData;
    int refIndex;
    dataDesc *data;
    bool writeOut;
    bool windowed;
    bool binary;
    struct plot *runPlot;
    FILE *fp;
    long pointPos;              /* where to write pointCount */
    int pointCount;
    int isComplex;
    int windowCount;
    int maxData;
};


int OUTpBeginPlot(CKTcircuit *circuitPtr, JOB *analysisPtr,
                  IFuid analName,
                  IFuid refName, int refType,
                  int numNames, IFuid *dataNames, int dataType, runDesc **plotPtr);
int OUTwBeginPlot(CKTcircuit *circuitPtr, JOB *analysisPtr,
                  IFuid analName,
                  IFuid refName, int refType,
                  int numNames, IFuid *dataNames, int dataType, runDesc **plotPtr);
int OUTpData(runDesc *plotPtr, IFvalue *refValue, IFvalue *valuePtr);
int OUTwReference(runDesc *plotPtr, IFvalue *valuePtr, void **refPtr);
int OUTwData(runDesc *plotPtr, int dataIndex, IFvalue *valuePtr, void *refPtr);
int OUTwEnd(runDesc *plotPtr);
int OUTendPlot(runDesc *plotPtr);
int OUTbeginDomain(runDesc *plotPtr, IFuid refName, int refType, IFvalue *outerRefValue);
int OUTendDomain(runDesc *plotPtr);
int OUTattributes(runDesc *plotPtr, IFuid varName, int param, IFvalue *value);
int OUTstopnow(void);
void OUTerror(int flags, char *format, IFuid *names);

#ifdef __GNUC__
void OUTerrorf(int, const char *fmt, ...)  __attribute__ ((format (__printf__, 2, 3)));
#else
void OUTerrorf(int, const char *fmt, ...);
#endif


#endif
