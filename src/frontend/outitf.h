/*************
 * Header file for outitf.c
 * 1999 E. Rouat
 ************/

#ifndef OUTITF_H_INCLUDED
#define OUTITF_H_INCLUDED
typedef struct dataDesc {
    char *name;     /* The name of the vector. */
    int type;       /* The type. */
    int	gtype;		/* default plot scale */
    bool regular;       /* Is this given to us? */
    int outIndex;       /* If regular then the index. */
    char *specName;     /* The device name if special. */
    char *specParamName;    /* The parameter name if special. */
    int specIndex;      /* For sensitivity, if special. */
    int specType;
    void *specFast;
    int refIndex;       /* The index of our ref vector. */
    struct dvec *vec;
} dataDesc;

typedef struct runDesc {
    void *analysis;
    void *circuit;
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
    long pointPos;          /* where to write pointCount */
    int pointCount;
    int isComplex;
    int windowCount;
} runDesc;

int OUTpBeginPlot(void *circuitPtr, void *analysisPtr, IFuid analName, IFuid refName, 
		  int refType, int numNames, IFuid *dataNames, int dataType, void **plotPtr);
int OUTwBeginPlot(void *circuitPtr, void *analysisPtr, IFuid analName, IFuid refName, 
		  int refType, int numNames, IFuid *dataNames, int dataType, void **plotPtr);
int OUTpData(void *plotPtr, IFvalue *refValue, IFvalue *valuePtr);
int OUTwReference(void *plotPtr, IFvalue *valuePtr, void **refPtr);
int OUTwData(void *plotPtr, int dataIndex, IFvalue *valuePtr, void *refPtr);
int OUTwEnd(void *plotPtr);
int OUTendPlot(void *plotPtr);
int OUTbeginDomain(void *plotPtr, char *refName, int refType, IFvalue *outerRefValue);
int OUTendDomain(void *plotPtr);
int OUTattributes(void *plotPtr, char *varName, int param, IFvalue *value);
int OUTstopnow(void);
void OUTerror(int flags, char *format, IFuid *names);


#endif
