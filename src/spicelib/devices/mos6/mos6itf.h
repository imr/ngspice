/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 T. Sakurai
Modified: 1999 Paolo Nenzi
**********/
#ifdef DEV_mos6

#ifndef DEV_MOS6
#define DEV_MOS6

#include "mos6ext.h"
extern IFparm MOS6pTable[ ];
extern IFparm MOS6mPTable[ ];
extern char *MOS6names[ ];
extern int     MOS6nSize;
extern int     MOS6pTSize;
extern int     MOS6mPTSize;
extern int MOS6iSize;
extern int MOS6mSize;

SPICEdev MOS6info = {
    {
        "Mos6",
        "Level 6 MOSfet model with Meyer capacitance model",

        &MOS6nSize,
        &MOS6nSize,
        MOS6names,

        &MOS6pTSize,
        MOS6pTable,

        &MOS6mPTSize,
        MOS6mPTable,
	DEV_DEFAULT
    },

    MOS6param,
    MOS6mParam,
    MOS6load,
    MOS6setup,
    MOS6unsetup,
    NULL, /* PZsetup routine */
    MOS6temp,
    MOS6trunc,
    NULL,
    NULL, /* MOS6acLoad, XXX */
    NULL,
    MOS6destroy,
    NULL, /* DELETES */
    NULL,/* DELETES */
    MOS6getic,
    MOS6ask,
    MOS6mAsk,
    NULL, /*MOS6pzLoad, XXX */
#ifdef NEWCONV
    MOS6convTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */

    NULL /* MOS6sSetup */,
    NULL /* MOS6sLoad */,
    NULL /* MOS6sUpdate */,
    NULL /* MOS6sAcLoad */,
    NULL /* MOS6sPrint */,
    NULL,
	NULL, /* Distortion routine */
	NULL, /* Noise routine */


    &MOS6iSize,
    &MOS6mSize
};

#endif
#endif
