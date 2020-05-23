/**********
Author: 2010 Paolo Nenzi
**********/

#ifndef ngspice_FTEOPTDEFS_H
#define ngspice_FTEOPTDEFS_H

    /* Structure used to describe the frontend statistics to be collected */
    /* This is similar to the STATististics in optdefs.h but collects     */
    /* statistics pertaining to ngspice frontend                          */

typedef struct sFTESTATistics {

    int FTESTATdeckNumLines;    /* number of lines in spice deck */

    double FTESTATnetLoadTime;  /* total time required to load the spice deck */
    double FTESTATnetPrepTime;  /* total time required to do subcircuit and numparam expansion */
    double FTESTATnetParseTime; /* total time required to parse the netlist */
} FTESTATistics;


enum {
    FTEOPT_NLDECK = 1,
    FTEOPT_NLT,
    FTEOPT_PRT,
    FTEOPT_NPT,
};

#endif
