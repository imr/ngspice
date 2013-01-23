/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

/*
 * Doping Profile Data Structures
 */
#ifndef ngspice_PROFILE_H
#define ngspice_PROFILE_H

typedef struct sDOPprofile {
    int type;                       /* Primary profile type UNIF, EXP, etc */
    int latType;                    /* Lateral profile type UNIF, EXP, etc */
    int rotate;			    /* Is lat prof rotation of primary? */
    int numDomains;                 /* Size of vector of domains to dope */
    int *domains;                   /* Vector of domains to dope */
    double param[10];
    struct sDOPprofile *next;
} DOPprofile;

#define CONC        param[1]
#define PEAK_CONC   param[1]
#define IMPID       param[1]
#define X_LOW       param[2]
#define X_HIGH      param[3]
#define Y_LOW       param[4]
#define Y_HIGH      param[5]
#define RANGE       param[6]
#define LOCATION    param[6]
#define CHAR_LENGTH param[7]
#define DIRECTION   param[8]
#define LAT_RATIO   param[9]

/* Structure for holding lookup tables of concentrations */
typedef struct sDOPtable {
    int impId;			    /* id of impurity */
    double **dopData;		    /* Array of doping values */
    struct sDOPtable *next;        /* Pointer to next table in list */
} DOPtable;

/* define impurity types */
#define IMP_BORON	1
#define IMP_PHOSPHORUS	2
#define IMP_ARSENIC	3
#define IMP_ANTIMONY	4
#define IMP_N_TYPE	5
#define IMP_P_TYPE	6

#endif
