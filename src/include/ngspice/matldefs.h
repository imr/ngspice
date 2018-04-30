/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors :  1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_MATLDEFS_H
#define ngspice_MATLDEFS_H

/* Data Structures and Definitions for Device Simulation Cards */

typedef struct sMATLcard {
    struct sMATLcard *MATLnextCard;
    int    MATLnumber;
    int    MATLmaterial;
    double MATLpermittivity;
    double MATLaffinity;
    double MATLnc0;
    double MATLnv0;
    double MATLeg0;
    double MATLdEgdT;
    double MATLtrefEg;
    double MATLdEgdN;
    double MATLnrefEg;
    double MATLdEgdP;
    double MATLprefEg;
    double MATLtaun0;
    double MATLtaup0;
    double MATLnrefSRHn;
    double MATLnrefSRHp;
    double MATLcnAug;
    double MATLcpAug;
    double MATLaRichN;
    double MATLaRichP;
    unsigned int MATLnumberGiven : 1;
    unsigned int MATLmaterialGiven : 1;
    unsigned int MATLpermittivityGiven : 1;
    unsigned int MATLaffinityGiven : 1;
    unsigned int MATLnc0Given : 1;
    unsigned int MATLnv0Given : 1;
    unsigned int MATLeg0Given : 1;
    unsigned int MATLdEgdTGiven : 1;
    unsigned int MATLtrefEgGiven : 1;
    unsigned int MATLdEgdNGiven : 1;
    unsigned int MATLnrefEgGiven : 1;
    unsigned int MATLdEgdPGiven : 1;
    unsigned int MATLprefEgGiven : 1;
    unsigned int MATLtaun0Given : 1;
    unsigned int MATLtaup0Given : 1;
    unsigned int MATLnrefSRHnGiven : 1;
    unsigned int MATLnrefSRHpGiven : 1;
    unsigned int MATLcnAugGiven : 1;
    unsigned int MATLcpAugGiven : 1;
    unsigned int MATLaRichNGiven : 1;
    unsigned int MATLaRichPGiven : 1;
    unsigned int MATLtnomGiven : 1;
} MATLcard;

/* MATL parameters */
enum {
    MATL_NC0 = 1,
    MATL_NV0,
    MATL_EG0,
    MATL_DEGDT,
    MATL_TREF_EG,
    MATL_DEGDN,
    MATL_NREF_EG,
    MATL_DEGDP,
    MATL_PREF_EG,
    MATL_AFFIN,
    MATL_PERMIT,
    MATL_TAUN0,
    MATL_TAUP0,
    MATL_NSRHN,
    MATL_NSRHP,
    MATL_CNAUG,
    MATL_CPAUG,
    MATL_ARICHN,
    MATL_ARICHP,
    MATL_INSULATOR,
    MATL_OXIDE,
    MATL_NITRIDE,
    MATL_SEMICON,
    MATL_SILICON,
    MATL_POLYSIL,
    MATL_GAAS,
    MATL_NUMBER,
    MATL_DEGDC,
    MATL_CREF_EG,
};

#endif
