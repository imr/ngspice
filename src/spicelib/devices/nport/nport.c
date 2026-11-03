/**********
Enhancement-242: native n-port device -- parameter tables.
**********/

#include "ngspice/ngspice.h"
#include "nportdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"

/* instance parameters: none (everything comes from the .model / fit file) */
IFparm NPORTpTable[] = {
    OPU("nports_i", NPORT_NPORTS, IF_INTEGER, "number of ports")
};

/* model parameters */
IFparm NPORTmPTable[] = {
    IP("nport", NPORT_MOD_NPORT, IF_FLAG,    "native n-port rational device"),
    IP("file",  NPORT_MOD_FILE,  IF_STRING,  "path to the .nport fit file"),
    OP("nports", NPORT_NPORTS,   IF_INTEGER, "number of ports"),
    OP("npoles", NPORT_NPOLES,   IF_INTEGER, "number of poles")
};

int NPORTpTSize  = NUMELEMS(NPORTpTable);
int NPORTmPTSize = NUMELEMS(NPORTmPTable);
int NPORTiSize   = sizeof(NPORTinstance);
int NPORTmSize   = sizeof(NPORTmodel);
