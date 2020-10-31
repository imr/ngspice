/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define MIN_R 0.001

/*ARGSUSED*/
int
HICUMmAsk(CKTcircuit *ckt, GENmodel *instPtr, int which, IFvalue *value)
{
    HICUMmodel *model = (HICUMmodel*)instPtr;

    NG_IGNORE(ckt);

    switch(which) {

//Circuit simulator specific parameters
        case HICUM_MOD_TYPE:
            if (model->HICUMtype == NPN)
                value->sValue = "npn";
            else
                value->sValue = "pnp";
            return(OK);
        case HICUM_MOD_TNOM:
            value->rValue = model->HICUMtnom;
            return(OK);

        case HICUM_MOD_VERSION :
            value->sValue = model->HICUMversion;
            return(OK);

//Transfer current
        case HICUM_MOD_C10:
            value->rValue = model->HICUMc10;
            return(OK);
        case HICUM_MOD_QP0:
            value->rValue = model->HICUMqp0;
            return(OK);
        case HICUM_MOD_ICH:
            value->rValue = model->HICUMich;
            return(OK);
        case HICUM_MOD_HF0:
            value->rValue = model->HICUMhf0;
            return(OK);
        case HICUM_MOD_HFE:
            value->rValue = model->HICUMhfe;
            return(OK);
        case HICUM_MOD_HFC:
            value->rValue = model->HICUMhfc;
            return(OK);
        case HICUM_MOD_HJEI:
            value->rValue = model->HICUMhjei;
            return(OK);
        case HICUM_MOD_AHJEI:
            value->rValue = model->HICUMahjei;
            return(OK);
        case HICUM_MOD_RHJEI:
            value->rValue = model->HICUMrhjei;
            return(OK);
        case HICUM_MOD_HJCI:
            value->rValue = model->HICUMhjci;
            return(OK);

//Base-Emitter diode:
        case HICUM_MOD_IBEIS:
            value->rValue = model->HICUMibeis;
            return(OK);
        case HICUM_MOD_MBEI:
            value->rValue = model->HICUMmbei;
            return(OK);
        case HICUM_MOD_IREIS:
            value->rValue = model->HICUMireis;
            return(OK);
        case HICUM_MOD_MREI:
            value->rValue = model->HICUMmrei;
            return(OK);
        case HICUM_MOD_IBEPS:
            value->rValue = model->HICUMibeps;
            return(OK);
        case HICUM_MOD_MBEP:
            value->rValue = model->HICUMmbep;
            return(OK);
        case HICUM_MOD_IREPS:
            value->rValue = model->HICUMireps;
            return(OK);
        case HICUM_MOD_MREP:
            value->rValue = model->HICUMmrep;
            return(OK);
        case HICUM_MOD_MCF:
            value->rValue = model->HICUMmcf;
            return(OK);
        case HICUM_MOD_TBHREC:
            value->rValue = model->HICUMtbhrec;
            return(OK);
        case HICUM_MOD_IBCIS:
            value->rValue = model->HICUMibcis;
            return(OK);
        case HICUM_MOD_MBCI:
            value->rValue = model->HICUMmbci;
            return(OK);
        case HICUM_MOD_IBCXS:
            value->rValue = model->HICUMibcxs;
            return(OK);
        case HICUM_MOD_MBCX:
            value->rValue = model->HICUMmbcx;
            return(OK);
        case HICUM_MOD_IBETS:
            value->rValue = model->HICUMibets;
            return(OK);
        case HICUM_MOD_ABET:
            value->rValue = model->HICUMabet;
            return(OK);
        case HICUM_MOD_TUNODE:
            value->rValue = model->HICUMtunode = value->iValue;
            return(OK);
        case HICUM_MOD_FAVL:
            value->rValue = model->HICUMfavl;
            return(OK);
        case HICUM_MOD_QAVL:
            value->rValue = model->HICUMqavl;
            return(OK);
        case HICUM_MOD_KAVL:
            value->rValue = model->HICUMkavl;
            return(OK);
        case HICUM_MOD_ALFAV:
            value->rValue = model->HICUMalfav;
            return(OK);
        case HICUM_MOD_ALQAV:
            value->rValue = model->HICUMalqav;
            return(OK);
        case HICUM_MOD_ALKAV:
            value->rValue = model->HICUMalkav;
            return(OK);
        case HICUM_MOD_RBI0:
            value->rValue = model->HICUMrbi0;
            return(OK);
        case HICUM_MOD_RBX:
            value->rValue = model->HICUMrbx;
            return(OK);
        case HICUM_MOD_FGEO:
            value->rValue = model->HICUMfgeo;
            return(OK);
        case HICUM_MOD_FDQR0:
            value->rValue = model->HICUMfdqr0;
            return(OK);
        case HICUM_MOD_FCRBI:
            value->rValue = model->HICUMfcrbi;
            return(OK);
        case HICUM_MOD_FQI:
            value->rValue = model->HICUMfqi;
            return(OK);
        case HICUM_MOD_RE:
            value->rValue = model->HICUMre;
            return(OK);
        case HICUM_MOD_RCX:
            value->rValue = model->HICUMrcx;
            return(OK);
        case HICUM_MOD_ITSS:
            value->rValue = model->HICUMitss;
            return(OK);
        case HICUM_MOD_MSF:
            value->rValue = model->HICUMmsf;
            return(OK);
        case HICUM_MOD_ISCS:
            value->rValue = model->HICUMiscs;
            return(OK);
        case HICUM_MOD_MSC:
            value->rValue = model->HICUMmsc;
            return(OK);
        case HICUM_MOD_TSF:
            value->rValue = model->HICUMtsf;
            return(OK);
        case HICUM_MOD_RSU:
            value->rValue = model->HICUMrsu;
            return(OK);
        case HICUM_MOD_CSU:
            value->rValue = model->HICUMcsu;
            return(OK);
        case HICUM_MOD_CJEI0:
            value->rValue = model->HICUMcjei0;
            return(OK);
        case HICUM_MOD_VDEI:
            value->rValue = model->HICUMvdei;
            return(OK);
        case HICUM_MOD_ZEI:
            value->rValue = model->HICUMzei;
            return(OK);
        case HICUM_MOD_AJEI:
            value->rValue = model->HICUMajei;
            return(OK);
        case HICUM_MOD_CJEP0:
            value->rValue = model->HICUMcjep0;
            return(OK);
        case HICUM_MOD_VDEP:
            value->rValue = model->HICUMvdep;
            return(OK);
        case HICUM_MOD_ZEP:
            value->rValue = model->HICUMzep;
            return(OK);
        case HICUM_MOD_AJEP:
            value->rValue = model->HICUMajep;
            return(OK);
        case HICUM_MOD_CJCI0:
            value->rValue = model->HICUMcjci0;
            return(OK);
        case HICUM_MOD_VDCI:
            value->rValue = model->HICUMvdci;
            return(OK);
        case HICUM_MOD_ZCI:
            value->rValue = model->HICUMzci;
            return(OK);
        case HICUM_MOD_VPTCI:
            value->rValue = model->HICUMvptci;
            return(OK);
        case HICUM_MOD_CJCX0:
            value->rValue = model->HICUMcjcx0;
            return(OK);
        case HICUM_MOD_VDCX:
            value->rValue = model->HICUMvdcx;
            return(OK);
        case HICUM_MOD_ZCX:
            value->rValue = model->HICUMzcx;
            return(OK);
        case HICUM_MOD_VPTCX:
            value->rValue = model->HICUMvptcx;
            return(OK);
        case HICUM_MOD_FBCPAR:
            value->rValue = model->HICUMfbcpar;
            return(OK);
        case HICUM_MOD_FBEPAR:
            value->rValue = model->HICUMfbepar;
            return(OK);
        case HICUM_MOD_CJS0:
            value->rValue = model->HICUMcjs0;
            return(OK);
        case HICUM_MOD_VDS:
            value->rValue = model->HICUMvds;
            return(OK);
        case HICUM_MOD_ZS:
            value->rValue = model->HICUMzs;
            return(OK);
        case HICUM_MOD_VPTS:
            value->rValue = model->HICUMvpts;
            return(OK);
        case HICUM_MOD_CSCP0:
            value->rValue = model->HICUMcscp0;
            return(OK);
        case HICUM_MOD_VDSP:
            value->rValue = model->HICUMvdsp;
            return(OK);
        case HICUM_MOD_ZSP:
            value->rValue = model->HICUMzsp;
            return(OK);
        case HICUM_MOD_VPTSP:
            value->rValue = model->HICUMvptsp;
            return(OK);
        case HICUM_MOD_T0:
            value->rValue = model->HICUMt0;
            return(OK);
        case HICUM_MOD_DT0H:
            value->rValue = model->HICUMdt0h;
            return(OK);
        case HICUM_MOD_TBVL:
            value->rValue = model->HICUMtbvl;
            return(OK);
        case HICUM_MOD_TEF0:
            value->rValue = model->HICUMtef0;
            return(OK);
        case HICUM_MOD_GTFE:
            value->rValue = model->HICUMgtfe;
            return(OK);
        case HICUM_MOD_THCS:
            value->rValue = model->HICUMthcs;
            return(OK);
        case HICUM_MOD_AHC:
            value->rValue = model->HICUMahc;
            return(OK);
        case HICUM_MOD_FTHC:
            value->rValue = model->HICUMfthc;
            return(OK);
        case HICUM_MOD_RCI0:
            value->rValue = model->HICUMrci0;
            return(OK);
        case HICUM_MOD_VLIM:
            value->rValue = model->HICUMvlim;
            return(OK);
        case HICUM_MOD_VCES:
            value->rValue = model->HICUMvces;
            return(OK);
        case HICUM_MOD_VPT:
            value->rValue = model->HICUMvpt;
            return(OK);
        case HICUM_MOD_AICK:
            value->rValue = model->HICUMaick;
            return(OK);
        case HICUM_MOD_DELCK:
            value->rValue = model->HICUMdelck;
            return(OK);
        case HICUM_MOD_TR:
            value->rValue = model->HICUMtr;
            return(OK);
        case HICUM_MOD_VCBAR:
            value->rValue = model->HICUMvcbar;
            return(OK);
        case HICUM_MOD_ICBAR:
            value->rValue = model->HICUMicbar;
            return(OK);
        case HICUM_MOD_ACBAR:
            value->rValue = model->HICUMacbar;
            return(OK);
        case HICUM_MOD_CBEPAR:
            value->rValue = model->HICUMcbepar;
            return(OK);
        case HICUM_MOD_CBCPAR:
            value->rValue = model->HICUMcbcpar;
            return(OK);
        case HICUM_MOD_ALQF:
            value->rValue = model->HICUMalqf;
            return(OK);
        case HICUM_MOD_ALIT:
            value->rValue = model->HICUMalit;
            return(OK);
        case HICUM_MOD_FLNQS:
            value->iValue = model->HICUMflnqs;
            return(OK);
        case HICUM_MOD_KF:
            value->rValue = model->HICUMkf;
            return(OK);
        case HICUM_MOD_AF:
            value->rValue = model->HICUMaf;
            return(OK);
        case HICUM_MOD_CFBE:
            value->rValue = model->HICUMcfbe;
            return(OK);
        case HICUM_MOD_FLCONO:
            value->iValue = model->HICUMflcono;
            return(OK);
        case HICUM_MOD_KFRE:
            value->rValue = model->HICUMkfre;
            return(OK);
        case HICUM_MOD_AFRE:
            value->rValue = model->HICUMafre;
            return(OK);
        case HICUM_MOD_LATB:
            value->rValue = model->HICUMlatb;
            return(OK);
        case HICUM_MOD_LATL:
            value->rValue = model->HICUMlatl;
            return(OK);
        case HICUM_MOD_VGB:
            value->rValue = model->HICUMvgb;
            return(OK);
        case HICUM_MOD_ALT0:
            value->rValue = model->HICUMalt0;
            return(OK);
        case HICUM_MOD_KT0:
            value->rValue = model->HICUMkt0;
            return(OK);
        case HICUM_MOD_ZETACI:
            value->rValue = model->HICUMzetaci;
            return(OK);
        case HICUM_MOD_ALVS:
            value->rValue = model->HICUMalvs;
            return(OK);
        case HICUM_MOD_ALCES:
            value->rValue = model->HICUMalces;
            return(OK);
        case HICUM_MOD_ZETARBI:
            value->rValue = model->HICUMzetarbi;
            return(OK);
        case HICUM_MOD_ZETARBX:
            value->rValue = model->HICUMzetarbx;
            return(OK);
        case HICUM_MOD_ZETARCX:
            value->rValue = model->HICUMzetarcx;
            return(OK);
        case HICUM_MOD_ZETARE:
            value->rValue = model->HICUMzetare;
            return(OK);
        case HICUM_MOD_ZETACX:
            value->rValue = model->HICUMzetacx;
            return(OK);
        case HICUM_MOD_VGE:
            value->rValue = model->HICUMvge;
            return(OK);
        case HICUM_MOD_VGC:
            value->rValue = model->HICUMvgc;
            return(OK);
        case HICUM_MOD_VGS:
            value->rValue = model->HICUMvgs;
            return(OK);
        case HICUM_MOD_F1VG:
            value->rValue = model->HICUMf1vg;
            return(OK);
        case HICUM_MOD_F2VG:
            value->rValue = model->HICUMf2vg;
            return(OK);
        case HICUM_MOD_ZETACT:
            value->rValue = model->HICUMzetact;
            return(OK);
        case HICUM_MOD_ZETABET:
            value->rValue = model->HICUMzetabet;
            return(OK);
        case HICUM_MOD_ALB:
            value->rValue = model->HICUMalb;
            return(OK);
        case HICUM_MOD_DVGBE:
            value->rValue = model->HICUMdvgbe;
            return(OK);
        case HICUM_MOD_ZETAHJEI:
            value->rValue = model->HICUMzetahjei;
            return(OK);
        case HICUM_MOD_ZETAVGBE:
            value->rValue = model->HICUMzetavgbe;
            return(OK);
        case HICUM_MOD_FLSH:
            value->iValue = model->HICUMflsh;
            return(OK);
        case HICUM_MOD_RTH:
            value->rValue = model->HICUMrth;
            return(OK);
        case HICUM_MOD_ZETARTH:
            value->rValue = model->HICUMzetarth;
            return(OK);
        case HICUM_MOD_ALRTH:
            value->rValue = model->HICUMalrth;
            return(OK);
        case HICUM_MOD_CTH:
            value->rValue = model->HICUMcth;
            return(OK);
        case HICUM_MOD_FLCOMP:
            value->rValue = model->HICUMflcomp;
            return(OK);
        case HICUM_MOD_VBE_MAX:
            value->rValue = model->HICUMvbeMax;
            return(OK);
        case HICUM_MOD_VBC_MAX:
            value->rValue = model->HICUMvbcMax;
            return(OK);
        case HICUM_MOD_VCE_MAX:
            value->rValue = model->HICUMvceMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

