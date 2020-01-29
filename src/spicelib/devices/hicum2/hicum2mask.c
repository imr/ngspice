/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "hicumdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define MIN_R 0.001

/*ARGSUSED*/
int
HICUMmAsk(CKTcircuit *ckt, GENmodel *instPtr, int which, IFvalue *value)
{
    HICUMmodel *here = (HICUMmodel*)instPtr;

    NG_IGNORE(ckt);

    switch(which) {

//Circuit simulator specific parameters
        case HICUM_MOD_TYPE:
            if (here->HICUMtype == NPN)
                value->sValue = "npn";
            else
                value->sValue = "pnp";
            return(OK);
        case HICUM_MOD_TNOM:
            value->rValue = here->HICUMtnom;
            return(OK);

//Transfer current
        case HICUM_MOD_C10:
            value->rValue = here->HICUMc10;
            return(OK);
        case HICUM_MOD_QP0:
            value->rValue = here->HICUMqp0;
            return(OK);
        case HICUM_MOD_ICH:
            value->rValue = here->HICUMich;
            return(OK);
        case HICUM_MOD_HF0:
            value->rValue = here->HICUMhf0;
            return(OK);
        case HICUM_MOD_HFE:
            value->rValue = here->HICUMhfe;
            return(OK);
        case HICUM_MOD_HFC:
            value->rValue = here->HICUMhfc;
            return(OK);
        case HICUM_MOD_HJEI:
            value->rValue = here->HICUMhjei;
            return(OK);
        case HICUM_MOD_AHJEI:
            value->rValue = here->HICUMahjei;
            return(OK);
        case HICUM_MOD_RHJEI:
            value->rValue = here->HICUMrhjei;
            return(OK);
        case HICUM_MOD_HJCI:
            value->rValue = here->HICUMhjci;
            return(OK);

//Base-Emitter diode:
        case HICUM_MOD_IBEIS:
            value->rValue = here->HICUMibeis;
            return(OK);
        case HICUM_MOD_MBEI:
            value->rValue = here->HICUMmbei;
            return(OK);
        case HICUM_MOD_IREIS:
            value->rValue = here->HICUMireis;
            return(OK);
        case HICUM_MOD_MREI:
            value->rValue = here->HICUMmrei;
            return(OK);
        case HICUM_MOD_IBEPS:
            value->rValue = here->HICUMibeps;
            return(OK);
        case HICUM_MOD_MBEP:
            value->rValue = here->HICUMmbep;
            return(OK);
        case HICUM_MOD_IREPS:
            value->rValue = here->HICUMireps;
            return(OK);
        case HICUM_MOD_MREP:
            value->rValue = here->HICUMmrep;
            return(OK);
        case HICUM_MOD_MCF:
            value->rValue = here->HICUMmcf;
            return(OK);

//Transit time for excess recombination current at b-c barrier
        case HICUM_MOD_TBHREC:
            value->rValue = here->HICUMtbhrec;
            return(OK);

//Base-Collector diode currents
        case HICUM_MOD_IBCIS:
            value->rValue = here->HICUMibcis;
            return(OK);
        case HICUM_MOD_MBCI:
            value->rValue = here->HICUMmbci;
            return(OK);
        case HICUM_MOD_IBCXS:
            value->rValue = here->HICUMibcxs;
            return(OK);
        case HICUM_MOD_MBCX:
            value->rValue = here->HICUMmbcx;
            return(OK);

//Base-Emitter tunneling current
        case HICUM_MOD_IBETS:
            value->rValue = here->HICUMibets;
            return(OK);
        case HICUM_MOD_ABET:
            value->rValue = here->HICUMabet;
            return(OK);
        case HICUM_MOD_TUNODE:
            value->rValue = here->HICUMtunode = value->iValue;
            return(OK);

//Base-Collector avalanche current
        case HICUM_MOD_FAVL:
            value->rValue = here->HICUMfavl;
            return(OK);
        case HICUM_MOD_QAVL:
            value->rValue = here->HICUMqavl;
            return(OK);
        case HICUM_MOD_KAVL:
            value->rValue = here->HICUMkavl;
            return(OK);
        case HICUM_MOD_ALFAV:
            value->rValue = here->HICUMalfav;
            return(OK);
        case HICUM_MOD_ALQAV:
            value->rValue = here->HICUMalqav;
            return(OK);
        case HICUM_MOD_ALKAV:
            value->rValue = here->HICUMalkav;
            return(OK);

//Series resistances
        case HICUM_MOD_RBI0:
            value->rValue = here->HICUMrbi0;
            if (here->HICUMrbi0 < MIN_R) here->HICUMrbi0 = MIN_R;
            return(OK);
        case HICUM_MOD_RBX:
            value->rValue = here->HICUMrbx;
            if (here->HICUMrbx < MIN_R) here->HICUMrbx = MIN_R;
            return(OK);
        case HICUM_MOD_FGEO:
            value->rValue = here->HICUMfgeo;
            return(OK);
        case HICUM_MOD_FDQR0:
            value->rValue = here->HICUMfdqr0;
            return(OK);
        case HICUM_MOD_FCRBI:
            value->rValue = here->HICUMfcrbi;
            return(OK);
        case HICUM_MOD_FQI:
            value->rValue = here->HICUMfqi;
            return(OK);
        case HICUM_MOD_RE:
            value->rValue = here->HICUMre;
            if (here->HICUMre < MIN_R) here->HICUMre = MIN_R;
            return(OK);
        case HICUM_MOD_RCX:
            value->rValue = here->HICUMrcx;
            if (here->HICUMrcx < MIN_R) here->HICUMrcx = MIN_R;
            return(OK);

//Substrate transistor
        case HICUM_MOD_ITSS:
            value->rValue = here->HICUMitss;
            return(OK);
        case HICUM_MOD_MSF:
            value->rValue = here->HICUMmsf;
            return(OK);
        case HICUM_MOD_ISCS:
            value->rValue = here->HICUMiscs;
            return(OK);
        case HICUM_MOD_MSC:
            value->rValue = here->HICUMmsc;
            return(OK);
        case HICUM_MOD_TSF:
            value->rValue = here->HICUMtsf;
            return(OK);

//Intra-device substrate coupling
        case HICUM_MOD_RSU:
            value->rValue = here->HICUMrsu;
            if (here->HICUMrsu < MIN_R) here->HICUMrsu = MIN_R;
            return(OK);
        case HICUM_MOD_CSU:

//Depletion Capacitances
        case HICUM_MOD_CJEI0:
            value->rValue = here->HICUMcjei0;
            return(OK);
        case HICUM_MOD_VDEI:
            value->rValue = here->HICUMvdei;
            return(OK);
        case HICUM_MOD_ZEI:
            value->rValue = here->HICUMzei;
            return(OK);
        case HICUM_MOD_AJEI:
            value->rValue = here->HICUMajei;
            return(OK);
        case HICUM_MOD_CJEP0:
            value->rValue = here->HICUMcjep0;
            return(OK);
        case HICUM_MOD_VDEP:
            value->rValue = here->HICUMvdep;
            return(OK);
        case HICUM_MOD_ZEP:
            value->rValue = here->HICUMzep;
            return(OK);
        case HICUM_MOD_AJEP:
            value->rValue = here->HICUMajep;
            return(OK);
        case HICUM_MOD_CJCI0:
            value->rValue = here->HICUMcjci0;
            return(OK);
        case HICUM_MOD_VDCI:
            value->rValue = here->HICUMvdci;
            return(OK);
        case HICUM_MOD_ZCI:
            value->rValue = here->HICUMzci;
            return(OK);
        case HICUM_MOD_VPTCI:
            value->rValue = here->HICUMvptci;
            return(OK);
        case HICUM_MOD_CJCX0:
            value->rValue = here->HICUMcjcx0;
            return(OK);
        case HICUM_MOD_VDCX:
            value->rValue = here->HICUMvdcx;
            return(OK);
        case HICUM_MOD_ZCX:
            value->rValue = here->HICUMzcx;
            return(OK);
        case HICUM_MOD_VPTCX:
            value->rValue = here->HICUMvptcx;
            return(OK);
        case HICUM_MOD_FBCPAR:
            value->rValue = here->HICUMfbcpar;
            return(OK);
        case HICUM_MOD_FBEPAR:
            value->rValue = here->HICUMfbepar;
            return(OK);
        case HICUM_MOD_CJS0:
            value->rValue = here->HICUMcjs0;
            return(OK);
        case HICUM_MOD_VDS:
            value->rValue = here->HICUMvds;
            return(OK);
        case HICUM_MOD_ZS:
            value->rValue = here->HICUMzs;
            return(OK);
        case HICUM_MOD_VPTS:
            value->rValue = here->HICUMvpts;
            return(OK);
        case HICUM_MOD_CSCP0:
            value->rValue = here->HICUMcscp0;
            return(OK);
        case HICUM_MOD_VDSP:
            value->rValue = here->HICUMvdsp;
            return(OK);
        case HICUM_MOD_ZSP:
            value->rValue = here->HICUMzsp;
            return(OK);
        case HICUM_MOD_VPTSP:
            value->rValue = here->HICUMvptsp;
            return(OK);

//Diffusion Capacitances
        case HICUM_MOD_T0:
            value->rValue = here->HICUMt0;
            return(OK);
        case HICUM_MOD_DT0H:
            value->rValue = here->HICUMdt0h;
            return(OK);
        case HICUM_MOD_TBVL:
            value->rValue = here->HICUMtbvl;
            return(OK);
        case HICUM_MOD_TEF0:
            value->rValue = here->HICUMtef0;
            return(OK);
        case HICUM_MOD_GTFE:
            value->rValue = here->HICUMgtfe;
            return(OK);
        case HICUM_MOD_THCS:
            value->rValue = here->HICUMthcs;
            return(OK);
        case HICUM_MOD_AHC:
            value->rValue = here->HICUMahc;
            return(OK);
        case HICUM_MOD_FTHC:
            value->rValue = here->HICUMfthc;
            return(OK);
        case HICUM_MOD_RCI0:
            value->rValue = here->HICUMrci0;
            return(OK);
        case HICUM_MOD_VLIM:
            value->rValue = here->HICUMvlim;
            return(OK);
        case HICUM_MOD_VCES:
            value->rValue = here->HICUMvces;
            return(OK);
        case HICUM_MOD_VPT:
            value->rValue = here->HICUMvpt;
            return(OK);
        case HICUM_MOD_AICK:
            value->rValue = here->HICUMaick;
            return(OK);
        case HICUM_MOD_DELCK:
            value->rValue = here->HICUMdelck;
            return(OK);
        case HICUM_MOD_TR:
            value->rValue = here->HICUMtr;
            return(OK);
        case HICUM_MOD_VCBAR:
            value->rValue = here->HICUMvcbar;
            return(OK);
        case HICUM_MOD_ICBAR:
            value->rValue = here->HICUMicbar;
            return(OK);
        case HICUM_MOD_ACBAR:
            value->rValue = here->HICUMacbar;
            return(OK);

//Isolation Capacitances
        case HICUM_MOD_CBEPAR:
            value->rValue = here->HICUMcbepar;
            return(OK);
        case HICUM_MOD_CBCPAR:
            value->rValue = here->HICUMcbcpar;
            return(OK);

//Non-quasi-static Effect
        case HICUM_MOD_ALQF:
            value->rValue = here->HICUMalqf;
            return(OK);
        case HICUM_MOD_ALIT:
            value->rValue = here->HICUMalit;
            return(OK);
        case HICUM_MOD_FLNQS:
            value->iValue = here->HICUMflnqs;
            return(OK);
  
//Noise
        case HICUM_MOD_KF:
            value->rValue = here->HICUMkf;
            return(OK);
        case HICUM_MOD_AF:
            value->rValue = here->HICUMaf;
            return(OK);
        case HICUM_MOD_CFBE:
            value->rValue = here->HICUMcfbe;
            return(OK);
        case HICUM_MOD_FLCONO:
            value->iValue = here->HICUMflcono;
            return(OK);
        case HICUM_MOD_KFRE:
            value->rValue = here->HICUMkfre;
            return(OK);
        case HICUM_MOD_AFRE:
            value->rValue = here->HICUMafre;
            return(OK);

//Lateral Geometry Scaling (at high current densities)
        case HICUM_MOD_LATB:
            value->rValue = here->HICUMlatb;
            return(OK);
        case HICUM_MOD_LATL:
            value->rValue = here->HICUMlatl;
            return(OK);

//Temperature dependence
        case HICUM_MOD_VGB:
            value->rValue = here->HICUMvgb;
            return(OK);
        case HICUM_MOD_ALT0:
            value->rValue = here->HICUMalt0;
            return(OK);
        case HICUM_MOD_KT0:
            value->rValue = here->HICUMkt0;
            return(OK);
        case HICUM_MOD_ZETACI:
            value->rValue = here->HICUMzetaci;
            return(OK);
        case HICUM_MOD_ALVS:
            value->rValue = here->HICUMalvs;
            return(OK);
        case HICUM_MOD_ALCES:
            value->rValue = here->HICUMalces;
            return(OK);
        case HICUM_MOD_ZETARBI:
            value->rValue = here->HICUMzetarbi;
            return(OK);
        case HICUM_MOD_ZETARBX:
            value->rValue = here->HICUMzetarbx;
            return(OK);
        case HICUM_MOD_ZETARCX:
            value->rValue = here->HICUMzetarcx;
            return(OK);
        case HICUM_MOD_ZETARE:
            value->rValue = here->HICUMzetare;
            return(OK);
        case HICUM_MOD_ZETACX:
            value->rValue = here->HICUMzetacx;
            return(OK);
        case HICUM_MOD_VGE:
            value->rValue = here->HICUMvge;
            return(OK);
        case HICUM_MOD_VGC:
            value->rValue = here->HICUMvgc;
            return(OK);
        case HICUM_MOD_VGS:
            value->rValue = here->HICUMvgs;
            return(OK);
        case HICUM_MOD_F1VG:
            value->rValue = here->HICUMf1vg;
            return(OK);
        case HICUM_MOD_F2VG:
            value->rValue = here->HICUMf2vg;
            return(OK);
        case HICUM_MOD_ZETACT:
            value->rValue = here->HICUMzetact;
            return(OK);
        case HICUM_MOD_ZETABET:
            value->rValue = here->HICUMzetabet;
            return(OK);
        case HICUM_MOD_ALB:
            value->rValue = here->HICUMalb;
            return(OK);
        case HICUM_MOD_DVGBE:
            value->rValue = here->HICUMdvgbe;
            return(OK);
        case HICUM_MOD_ZETAHJEI:
            value->rValue = here->HICUMzetahjei;
            return(OK);
        case HICUM_MOD_ZETAVGBE:
            value->rValue = here->HICUMzetavgbe;
            return(OK);

//Self-Heating
        case HICUM_MOD_FLSH:
            value->iValue = here->HICUMflsh;
            return(OK);
        case HICUM_MOD_RTH:
            value->rValue = here->HICUMrth;
            return(OK);
        case HICUM_MOD_ZETARTH:
            value->rValue = here->HICUMzetarth;
            return(OK);
        case HICUM_MOD_ALRTH:
            value->rValue = here->HICUMalrth;
            return(OK);
        case HICUM_MOD_CTH:
            value->rValue = here->HICUMcth;
            return(OK);

//Compatibility with V2.1
        case HICUM_MOD_FLCOMP:
            value->rValue = here->HICUMflcomp;
            return(OK);

//SOA-check
        case HICUM_MOD_VBE_MAX:
            value->rValue = here->HICUMvbeMax;
            return(OK);
        case HICUM_MOD_VBC_MAX:
            value->rValue = here->HICUMvbcMax;
            return(OK);
        case HICUM_MOD_VCE_MAX:
            value->rValue = here->HICUMvceMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

