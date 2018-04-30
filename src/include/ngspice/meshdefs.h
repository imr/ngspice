/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Author:  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_MESHDEFS_H
#define ngspice_MESHDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sMESHcard {
    struct sMESHcard *MESHnextCard;
    double MESHlocation;
    double MESHwidth;
    double MESHlocStart;
    double MESHlocEnd;
    double MESHhStart;
    double MESHhEnd;
    double MESHhMax;
    double MESHratio;
    int MESHnumber;
    unsigned int MESHlocationGiven : 1;
    unsigned int MESHwidthGiven : 1;
    unsigned int MESHhStartGiven : 1;
    unsigned int MESHhEndGiven : 1;
    unsigned int MESHhMaxGiven : 1;
    unsigned int MESHratioGiven : 1;
    unsigned int MESHnumberGiven : 1;
} MESHcard;

/* MESH parameters */
enum {
    MESH_NUMBER = 1,
    MESH_LOCATION,
    MESH_WIDTH,
    MESH_H_START,
    MESH_H_END,
    MESH_H_MAX,
    MESH_RATIO,
};

#endif
