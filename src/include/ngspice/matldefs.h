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
#define MATL_NC0	1
#define MATL_NV0	2
#define MATL_EG0	3
#define MATL_DEGDT	4
#define MATL_TREF_EG	5
#define MATL_DEGDN	6
#define MATL_NREF_EG	7
#define MATL_DEGDP	8
#define MATL_PREF_EG	9
#define MATL_AFFIN	10
#define MATL_PERMIT	11
#define MATL_TAUN0	12
#define MATL_TAUP0	13
#define MATL_NSRHN	14
#define MATL_NSRHP	15
#define MATL_CNAUG	16
#define MATL_CPAUG	17
#define MATL_ARICHN	18
#define MATL_ARICHP	19
#define MATL_INSULATOR  20
#define MATL_OXIDE	21
#define MATL_NITRIDE	22
#define MATL_SEMICON	23
#define MATL_SILICON	24
#define MATL_POLYSIL	25
#define MATL_GAAS	26
#define MATL_NUMBER	27
#define MATL_DEGDC	28
#define MATL_CREF_EG	29

#endif
