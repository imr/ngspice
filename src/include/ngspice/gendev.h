/**********
Copyright 1992 Regents of the University of California. All rights reserved.
Authors: 1992 David Gates
**********/

/* Member of CIDER device simulator
 * Version: 1b1
 */

#ifndef ngspice_GENDEV_H
#define ngspice_GENDEV_H

#include "ngspice/numenum.h"

typedef struct sStatInfo
{
  double setupTime[NUM_STATTYPES];
  double loadTime[NUM_STATTYPES];
  double orderTime[NUM_STATTYPES];
  double factorTime[NUM_STATTYPES];
  double solveTime[NUM_STATTYPES];
  double updateTime[NUM_STATTYPES];
  double checkTime[NUM_STATTYPES];
  double miscTime[NUM_STATTYPES];
  double totalTime[NUM_STATTYPES];
  double lteTime;
  int numIters[NUM_STATTYPES];
} StatInfo;
typedef struct sStatInfo ONEstats;
typedef struct sStatInfo TWOstats;
typedef struct sStatInfo STATstats;

/* Transient analysis information transferred via this structure. */
typedef struct sTranInfo {
    int method;                /* integration method */
    int order;                 /* integration order  */
    int maxOrder;              /* maximum order to be used */
    double lteCoeff;           /* coefficient for calculating LTE */
    double intCoeff[7];        /* array of integration coefficients */
    double predCoeff[7];       /* array of predicted coefficients */
    double *delta;             /* array of the time deltas */
} TranInfo;
typedef struct sTranInfo ONEtranInfo;
typedef struct sTranInfo TWOtranInfo;

/* Mesh coordinates transferred via this structure */
typedef struct sCoordInfo {
    struct sCoordInfo *next;       /* pointer to next mesh info */
    int number;                    /* number/position in list of coordinates */
    double location;               /* location of node */
} CoordInfo;
typedef struct sCoordInfo ONEcoord;
typedef struct sCoordInfo TWOcoord;
typedef struct sCoordInfo MESHcoord;

/* Generic vertex structure */
typedef struct sVertexInfo {
    struct sVertexInfo *next;      /* pointer to next vertex */
    int ix;                        /* the x coordinate */
    int iy;                        /* the y coordinate */
} VertexInfo;
typedef struct sVertexInfo ONEvertex;
typedef struct sVertexInfo TWOvertex;

/* Generic box structure that other box-shaped things are derived from */
typedef struct sBoxInfo {
    struct sBoxInfo *next;         /* pointer to next box */
    int ixLo;                      /* the low x coordinate */
    int iyLo;                      /* the low y coordinate */
    int ixHi;                      /* the high x coordinate */
    int iyHi;                      /* the high y coordinate */
} BoxInfo;
typedef struct sBoxInfo ONEbox;
typedef struct sBoxInfo TWObox;

/* Structure for domains */
typedef struct sDomainInfo {
    struct sDomainInfo *next;      /* pointer to next domain */
    int ixLo;                      /* the low x coordinate */
    int iyLo;                      /* the low y coordinate */
    int ixHi;                      /* the high x coordinate */
    int iyHi;                      /* the high y coordinate */
    int id;                        /* ID number of domain */
    int material;                  /* ID of material used by domain */
} DomainInfo;
typedef struct sDomainInfo ONEdomain;
typedef struct sDomainInfo TWOdomain;
typedef struct sDomainInfo DOMNdomain;

/* Structure used for electrodes */
typedef struct sElectrodeInfo {
    struct sElectrodeInfo *next;   /* pointer to next electrode */
    int ixLo;                      /* the low x coordinate */
    int iyLo;                      /* the low y coordinate */
    int ixHi;                      /* the high x coordinate */
    int iyHi;                      /* the high y coordinate */
    int id;                        /* ID number */
    double workf;                  /* electrode work function */
} ElectrodeInfo;
typedef struct sElectrodeInfo ONEelectrode;
typedef struct sElectrodeInfo TWOelectrode;
typedef struct sElectrodeInfo ELCTelectrode;

/* Structure used for boundaries and interfaces */
typedef struct sBoundaryInfo {
    struct sBoundaryInfo *next;    /* pointer to next boundary */
    int ixLo;                      /* the low x coordinate */
    int iyLo;                      /* the low y coordinate */
    int ixHi;                      /* the high x coordinate */
    int iyHi;                      /* the high y coordinate */
    int domain;			   /* ID of primary domain */
    int neighbor;		   /* ID of neighbor domain */
    double qf;			   /* fixed charge density */
    double sn;			   /* elec surface recomb velocity */
    double sp;			   /* hole surface recomb velocity */
    double layer;		   /* surface layer width */
} BoundaryInfo;
typedef struct sBoundaryInfo ONEboundary;
typedef struct sBoundaryInfo TWOboundary;
typedef struct sBoundaryInfo BDRYboundary;

#endif
