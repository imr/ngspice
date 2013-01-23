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
#define MESH_NUMBER	1
#define MESH_LOCATION	2
#define MESH_WIDTH	3
#define MESH_H_START	4
#define MESH_H_END	5
#define MESH_H_MAX	6
#define MESH_RATIO	7

#endif
