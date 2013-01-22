#ifndef ngspice_COMPLETION_H
#define ngspice_COMPLETION_H

/* The types for command completion keywords. Note that these
 * constants are built into cmdtab.c, so DON'T change them unless you
 * want to change all of the bitmasks in cp_coms.  Note that this is
 * spice- and nutmeg- dependent.  */

#define CT_FILENAME     0
#define CT_CKTNAMES     2
#define CT_COMMANDS     3
#define CT_DBNUMS       4
#define CT_DEVNAMES     5
#define CT_LISTINGARGS  6
#define CT_NODENAMES    7
#define CT_PLOT         8
#define CT_PLOTKEYWORDS 9
#define CT_RUSEARGS     10
#define CT_STOPARGS     11
#define CT_UDFUNCS      12
#define CT_VARIABLES    13
#define CT_VECTOR       14
#define CT_TYPENAMES    16

#endif
