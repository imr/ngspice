/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
**********/

/*
 * This routine sets model parameters for
 * HICUMs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
HICUMmParam(int param, IFvalue *value, GENmodel *inModel)
{
    HICUMmodel *model = (HICUMmodel*)inModel;

    switch(param) {

//Circuit simulator specific parameters

        case HICUM_MOD_NPN:
            if(value->iValue) {
                model->HICUMtype = NPN;
            }
            break;
        case HICUM_MOD_PNP:
            if(value->iValue) {
                model->HICUMtype = PNP;
            }
            break;

        case HICUM_MOD_TNOM:
            model->HICUMtnom = value->rValue+CONSTCtoK;
            model->HICUMtnomGiven = TRUE;
            break;

        case HICUM_MOD_VERSION :
            model->HICUMversion = value->sValue;
            model->HICUMversionGiven = TRUE;
            break;

//Transfer current
        case HICUM_MOD_C10:
            model->HICUMc10 = value->rValue;
            model->HICUMc10Given = TRUE;
            break;
        case HICUM_MOD_QP0:
            model->HICUMqp0 = value->rValue;
            model->HICUMqp0Given = TRUE;
            break;
        case HICUM_MOD_ICH:
            model->HICUMich = value->rValue;
            model->HICUMichGiven = TRUE;
            break;
        case HICUM_MOD_HF0:
            model->HICUMhf0 = value->rValue;
            model->HICUMhf0Given = TRUE;
            break;
        case HICUM_MOD_HFE:
            model->HICUMhfe = value->rValue;
            model->HICUMhfeGiven = TRUE;
            break;
        case HICUM_MOD_HFC:
            model->HICUMhfc = value->rValue;
            model->HICUMhfcGiven = TRUE;
            break;
        case HICUM_MOD_HJEI:
            model->HICUMhjei = value->rValue;
            model->HICUMhjeiGiven = TRUE;
            break;
        case HICUM_MOD_AHJEI:
            model->HICUMahjei = value->rValue;
            model->HICUMahjeiGiven = TRUE;
            break;
        case HICUM_MOD_RHJEI:
            model->HICUMrhjei = value->rValue;
            model->HICUMrhjeiGiven = TRUE;
            break;
        case HICUM_MOD_HJCI:
            model->HICUMhjci = value->rValue;
            model->HICUMhjciGiven = TRUE;
            break;

//Base-Emitter diode:
        case HICUM_MOD_IBEIS:
            model->HICUMibeis = value->rValue;
            model->HICUMibeisGiven = TRUE;
            break;
        case HICUM_MOD_MBEI:
            model->HICUMmbei = value->rValue;
            model->HICUMmbeiGiven = TRUE;
            break;
        case HICUM_MOD_IREIS:
            model->HICUMireis = value->rValue;
            model->HICUMireisGiven = TRUE;
            break;
        case HICUM_MOD_MREI:
            model->HICUMmrei = value->rValue;
            model->HICUMmreiGiven = TRUE;
            break;
        case HICUM_MOD_IBEPS:
            model->HICUMibeps = value->rValue;
            model->HICUMibepsGiven = TRUE;
            break;
        case HICUM_MOD_MBEP:
            model->HICUMmbep = value->rValue;
            model->HICUMmbepGiven = TRUE;
            break;
        case HICUM_MOD_IREPS:
            model->HICUMireps = value->rValue;
            model->HICUMirepsGiven = TRUE;
            break;
        case HICUM_MOD_MREP:
            model->HICUMmrep = value->rValue;
            model->HICUMmrepGiven = TRUE;
            break;
        case HICUM_MOD_MCF:
            model->HICUMmcf = value->rValue;
            model->HICUMmcfGiven = TRUE;
            break;

//Transit time for excess recombination current at b-c barrier
        case HICUM_MOD_TBHREC:
            model->HICUMtbhrec = value->rValue;
            model->HICUMtbhrecGiven = TRUE;
            break;

//Base-Collector diode currents
        case HICUM_MOD_IBCIS:
            model->HICUMibcis = value->rValue;
            model->HICUMibcisGiven = TRUE;
            break;
        case HICUM_MOD_MBCI:
            model->HICUMmbci = value->rValue;
            model->HICUMmbciGiven = TRUE;
            break;
        case HICUM_MOD_IBCXS:
            model->HICUMibcxs = value->rValue;
            model->HICUMibcxsGiven = TRUE;
            break;
        case HICUM_MOD_MBCX:
            model->HICUMmbcx = value->rValue;
            model->HICUMmbcxGiven = TRUE;
            break;

//Base-Emitter tunneling current
        case HICUM_MOD_IBETS:
            model->HICUMibets = value->rValue;
            model->HICUMibetsGiven = TRUE;
            break;
        case HICUM_MOD_ABET:
            model->HICUMabet = value->rValue;
            model->HICUMabetGiven = TRUE;
            break;
        case HICUM_MOD_TUNODE:
            model->HICUMtunode = value->iValue;
            model->HICUMtunodeGiven = TRUE;
            break;

//Base-Collector avalanche current
        case HICUM_MOD_FAVL:
            model->HICUMfavl = value->rValue;
            model->HICUMfavlGiven = TRUE;
            break;
        case HICUM_MOD_QAVL:
            model->HICUMqavl = value->rValue;
            model->HICUMqavlGiven = TRUE;
            break;
        case HICUM_MOD_KAVL:
            model->HICUMkavl = value->rValue;
            model->HICUMkavlGiven = TRUE;
            break;
        case HICUM_MOD_ALFAV:
            model->HICUMalfav = value->rValue;
            model->HICUMalfavGiven = TRUE;
            break;
        case HICUM_MOD_ALQAV:
            model->HICUMalqav = value->rValue;
            model->HICUMalqavGiven = TRUE;
            break;
        case HICUM_MOD_ALKAV:
            model->HICUMalkav = value->rValue;
            model->HICUMalkavGiven = TRUE;
            break;

//Series resistances
        case HICUM_MOD_RBI0:
            model->HICUMrbi0 = value->rValue;
            model->HICUMrbi0Given = TRUE;
            break;
        case HICUM_MOD_RBX:
            model->HICUMrbx = value->rValue;
            model->HICUMrbxGiven = TRUE;
            break;
        case HICUM_MOD_FGEO:
            model->HICUMfgeo = value->rValue;
            model->HICUMfgeoGiven = TRUE;
            break;
        case HICUM_MOD_FDQR0:
            model->HICUMfdqr0 = value->rValue;
            model->HICUMfdqr0Given = TRUE;
            break;
        case HICUM_MOD_FCRBI:
            model->HICUMfcrbi = value->rValue;
            model->HICUMfcrbiGiven = TRUE;
            break;
        case HICUM_MOD_FQI:
            model->HICUMfqi = value->rValue;
            model->HICUMfqiGiven = TRUE;
            break;
        case HICUM_MOD_RE:
            model->HICUMre = value->rValue;
            model->HICUMreGiven = TRUE;
            break;
        case HICUM_MOD_RCX:
            model->HICUMrcx = value->rValue;
            model->HICUMrcxGiven = TRUE;
            break;

//Substrate transistor
        case HICUM_MOD_ITSS:
            model->HICUMitss = value->rValue;
            model->HICUMitssGiven = TRUE;
            break;
        case HICUM_MOD_MSF:
            model->HICUMmsf = value->rValue;
            model->HICUMmsfGiven = TRUE;
            break;
        case HICUM_MOD_ISCS:
            model->HICUMiscs = value->rValue;
            model->HICUMiscsGiven = TRUE;
            break;
        case HICUM_MOD_MSC:
            model->HICUMmsc = value->rValue;
            model->HICUMmscGiven = TRUE;
            break;
        case HICUM_MOD_TSF:
            model->HICUMtsf = value->rValue;
            model->HICUMtsfGiven = TRUE;
            break;

//Intra-device substrate coupling
        case HICUM_MOD_RSU:
            model->HICUMrsu = value->rValue;
            model->HICUMrsuGiven = TRUE;
            break;
        case HICUM_MOD_CSU:
            model->HICUMcsu = value->rValue;
            model->HICUMcsuGiven = TRUE;
            break;

//Depletion Capacitances
        case HICUM_MOD_CJEI0:
            model->HICUMcjei0 = value->rValue;
            model->HICUMcjei0Given = TRUE;
            break;
        case HICUM_MOD_VDEI:
            model->HICUMvdei = value->rValue;
            model->HICUMvdeiGiven = TRUE;
            break;
        case HICUM_MOD_ZEI:
            model->HICUMzei = value->rValue;
            model->HICUMzeiGiven = TRUE;
            break;
        case HICUM_MOD_AJEI:
            model->HICUMajei = value->rValue;
            model->HICUMajeiGiven = TRUE;
            break;
        case HICUM_MOD_CJEP0:
            model->HICUMcjep0 = value->rValue;
            model->HICUMcjep0Given = TRUE;
            break;
        case HICUM_MOD_VDEP:
            model->HICUMvdep = value->rValue;
            model->HICUMvdepGiven = TRUE;
            break;
        case HICUM_MOD_ZEP:
            model->HICUMzep = value->rValue;
            model->HICUMzepGiven = TRUE;
            break;
        case HICUM_MOD_AJEP:
            model->HICUMajep = value->rValue;
            model->HICUMajepGiven = TRUE;
            break;
        case HICUM_MOD_CJCI0:
            model->HICUMcjci0 = value->rValue;
            model->HICUMcjci0Given = TRUE;
            break;
        case HICUM_MOD_VDCI:
            model->HICUMvdci = value->rValue;
            model->HICUMvdciGiven = TRUE;
            break;
        case HICUM_MOD_ZCI:
            model->HICUMzci = value->rValue;
            model->HICUMzciGiven = TRUE;
            break;
        case HICUM_MOD_VPTCI:
            model->HICUMvptci = value->rValue;
            model->HICUMvptciGiven = TRUE;
            break;
        case HICUM_MOD_CJCX0:
            model->HICUMcjcx0 = value->rValue;
            model->HICUMcjcx0Given = TRUE;
            break;
        case HICUM_MOD_VDCX:
            model->HICUMvdcx = value->rValue;
            model->HICUMvdcxGiven = TRUE;
            break;
        case HICUM_MOD_ZCX:
            model->HICUMzcx = value->rValue;
            model->HICUMzcxGiven = TRUE;
            break;
        case HICUM_MOD_VPTCX:
            model->HICUMvptcx = value->rValue;
            model->HICUMvptcxGiven = TRUE;
            break;
        case HICUM_MOD_FBCPAR:
            model->HICUMfbcpar = value->rValue;
            model->HICUMfbcparGiven = TRUE;
            break;
        case HICUM_MOD_FBEPAR:
            model->HICUMfbepar = value->rValue;
            model->HICUMfbeparGiven = TRUE;
            break;
        case HICUM_MOD_CJS0:
            model->HICUMcjs0 = value->rValue;
            model->HICUMcjs0Given = TRUE;
            break;
        case HICUM_MOD_VDS:
            model->HICUMvds = value->rValue;
            model->HICUMvdsGiven = TRUE;
            break;
        case HICUM_MOD_ZS:
            model->HICUMzs = value->rValue;
            model->HICUMzsGiven = TRUE;
            break;
        case HICUM_MOD_VPTS:
            model->HICUMvpts = value->rValue;
            model->HICUMvptsGiven = TRUE;
            break;
        case HICUM_MOD_CSCP0:
            model->HICUMcscp0 = value->rValue;
            model->HICUMcscp0Given = TRUE;
            break;
        case HICUM_MOD_VDSP:
            model->HICUMvdsp = value->rValue;
            model->HICUMvdspGiven = TRUE;
            break;
        case HICUM_MOD_ZSP:
            model->HICUMzsp = value->rValue;
            model->HICUMzspGiven = TRUE;
            break;
        case HICUM_MOD_VPTSP:
            model->HICUMvptsp = value->rValue;
            model->HICUMvptspGiven = TRUE;
            break;

//Diffusion Capacitances
        case HICUM_MOD_T0:
            model->HICUMt0 = value->rValue;
            model->HICUMt0Given = TRUE;
            break;
        case HICUM_MOD_DT0H:
            model->HICUMdt0h = value->rValue;
            model->HICUMdt0hGiven = TRUE;
            break;
        case HICUM_MOD_TBVL:
            model->HICUMtbvl = value->rValue;
            model->HICUMtbvlGiven = TRUE;
            break;
        case HICUM_MOD_TEF0:
            model->HICUMtef0 = value->rValue;
            model->HICUMtef0Given = TRUE;
            break;
        case HICUM_MOD_GTFE:
            model->HICUMgtfe = value->rValue;
            model->HICUMgtfeGiven = TRUE;
            break;
        case HICUM_MOD_THCS:
            model->HICUMthcs = value->rValue;
            model->HICUMthcsGiven = TRUE;
            break;
        case HICUM_MOD_AHC:
            model->HICUMahc = value->rValue;
            model->HICUMahcGiven = TRUE;
            break;
        case HICUM_MOD_FTHC:
            model->HICUMfthc = value->rValue;
            model->HICUMfthcGiven = TRUE;
            break;
        case HICUM_MOD_RCI0:
            model->HICUMrci0 = value->rValue;
            model->HICUMrci0Given = TRUE;
            break;
        case HICUM_MOD_VLIM:
            model->HICUMvlim = value->rValue;
            model->HICUMvlimGiven = TRUE;
            break;
        case HICUM_MOD_VCES:
            model->HICUMvces = value->rValue;
            model->HICUMvcesGiven = TRUE;
            break;
        case HICUM_MOD_VPT:
            model->HICUMvpt = value->rValue;
            model->HICUMvptGiven = TRUE;
            break;
        case HICUM_MOD_AICK:
            model->HICUMaick = value->rValue;
            model->HICUMaickGiven = TRUE;
            break;
        case HICUM_MOD_DELCK:
            model->HICUMdelck = value->rValue;
            model->HICUMdelckGiven = TRUE;
            break;
        case HICUM_MOD_TR:
            model->HICUMtr = value->rValue;
            model->HICUMtrGiven = TRUE;
            break;
        case HICUM_MOD_VCBAR:
            model->HICUMvcbar = value->rValue;
            model->HICUMvcbarGiven = TRUE;
            break;
        case HICUM_MOD_ICBAR:
            model->HICUMicbar = value->rValue;
            model->HICUMicbarGiven = TRUE;
            break;
        case HICUM_MOD_ACBAR:
            model->HICUMacbar = value->rValue;
            model->HICUMacbarGiven = TRUE;
            break;

//Isolation Capacitances
        case HICUM_MOD_CBEPAR:
            model->HICUMcbepar = value->rValue;
            model->HICUMcbeparGiven = TRUE;
            break;
        case HICUM_MOD_CBCPAR:
            model->HICUMcbcpar = value->rValue;
            model->HICUMcbcparGiven = TRUE;
            break;

//Non-quasi-static Effect
        case HICUM_MOD_ALQF:
            model->HICUMalqf = value->rValue;
            model->HICUMalqfGiven = TRUE;
            break;
        case HICUM_MOD_ALIT:
            model->HICUMalit = value->rValue;
            model->HICUMalitGiven = TRUE;
            break;
        case HICUM_MOD_FLNQS:
            model->HICUMflnqs = value->iValue;
            model->HICUMflnqsGiven = TRUE;
            break;

//Noise
        case HICUM_MOD_KF:
            model->HICUMkf = value->rValue;
            model->HICUMkfGiven = TRUE;
            break;
        case HICUM_MOD_AF:
            model->HICUMaf = value->rValue;
            model->HICUMafGiven = TRUE;
            break;
        case HICUM_MOD_CFBE:
            model->HICUMcfbe = value->iValue;
            model->HICUMcfbeGiven = TRUE;
            break;
        case HICUM_MOD_FLCONO:
            model->HICUMflcono = value->iValue;
            model->HICUMflconoGiven = TRUE;
            break;
        case HICUM_MOD_KFRE:
            model->HICUMkfre = value->rValue;
            model->HICUMkfreGiven = TRUE;
            break;
        case HICUM_MOD_AFRE:
            model->HICUMafre = value->rValue;
            model->HICUMafreGiven = TRUE;
            break;

//Lateral Geometry Scaling (at high current densities)
        case HICUM_MOD_LATB:
            model->HICUMlatb = value->rValue;
            model->HICUMlatbGiven = TRUE;
            break;
        case HICUM_MOD_LATL:
            model->HICUMlatl = value->rValue;
            model->HICUMlatlGiven = TRUE;
            break;

//Temperature dependence
        case HICUM_MOD_VGB:
            model->HICUMvgb = value->rValue;
            model->HICUMvgbGiven = TRUE;
            break;
        case HICUM_MOD_ALT0:
            model->HICUMalt0 = value->rValue;
            model->HICUMalt0Given = TRUE;
            break;
        case HICUM_MOD_KT0:
            model->HICUMkt0 = value->rValue;
            model->HICUMkt0Given = TRUE;
            break;
        case HICUM_MOD_ZETACI:
            model->HICUMzetaci = value->rValue;
            model->HICUMzetaciGiven = TRUE;
            break;
        case HICUM_MOD_ALVS:
            model->HICUMalvs = value->rValue;
            model->HICUMalvsGiven = TRUE;
            break;
        case HICUM_MOD_ALCES:
            model->HICUMalces = value->rValue;
            model->HICUMalcesGiven = TRUE;
            break;
        case HICUM_MOD_ZETARBI:
            model->HICUMzetarbi = value->rValue;
            model->HICUMzetarbiGiven = TRUE;
            break;
        case HICUM_MOD_ZETARBX:
            model->HICUMzetarbx = value->rValue;
            model->HICUMzetarbxGiven = TRUE;
            break;
        case HICUM_MOD_ZETARCX:
            model->HICUMzetarcx = value->rValue;
            model->HICUMzetarcxGiven = TRUE;
            break;
        case HICUM_MOD_ZETARE:
            model->HICUMzetare = value->rValue;
            model->HICUMzetareGiven = TRUE;
            break;
        case HICUM_MOD_ZETACX:
            model->HICUMzetacx = value->rValue;
            model->HICUMzetacxGiven = TRUE;
            break;
        case HICUM_MOD_VGE:
            model->HICUMvge = value->rValue;
            model->HICUMvgeGiven = TRUE;
            break;
        case HICUM_MOD_VGC:
            model->HICUMvgc = value->rValue;
            model->HICUMvgcGiven = TRUE;
            break;
        case HICUM_MOD_VGS:
            model->HICUMvgs = value->rValue;
            model->HICUMvgsGiven = TRUE;
            break;
        case HICUM_MOD_F1VG:
            model->HICUMf1vg = value->rValue;
            model->HICUMf1vgGiven = TRUE;
            break;
        case HICUM_MOD_F2VG:
            model->HICUMf2vg = value->rValue;
            model->HICUMf2vgGiven = TRUE;
            break;
        case HICUM_MOD_ZETACT:
            model->HICUMzetact = value->rValue;
            model->HICUMzetactGiven = TRUE;
            break;
        case HICUM_MOD_ZETABET:
            model->HICUMzetabet = value->rValue;
            model->HICUMzetabetGiven = TRUE;
            break;
        case HICUM_MOD_ALB:
            model->HICUMalb = value->rValue;
            model->HICUMalbGiven = TRUE;
            break;
        case HICUM_MOD_DVGBE:
            model->HICUMdvgbe = value->rValue;
            model->HICUMdvgbeGiven = TRUE;
            break;
        case HICUM_MOD_ZETAHJEI:
            model->HICUMzetahjei = value->rValue;
            model->HICUMzetahjeiGiven = TRUE;
            break;
        case HICUM_MOD_ZETAVGBE:
            model->HICUMzetavgbe = value->rValue;
            model->HICUMzetavgbeGiven = TRUE;
            break;

//Self-Heating
        case HICUM_MOD_FLSH:
            model->HICUMflsh = value->iValue;
            model->HICUMflshGiven = TRUE;
            break;
        case HICUM_MOD_RTH:
            model->HICUMrth = value->rValue;
            model->HICUMrthGiven = TRUE;
            break;
        case HICUM_MOD_ZETARTH:
            model->HICUMzetarth = value->rValue;
            model->HICUMzetarthGiven = TRUE;
            break;
        case HICUM_MOD_ALRTH:
            model->HICUMalrth = value->rValue;
            model->HICUMalrthGiven = TRUE;
            break;
        case HICUM_MOD_CTH:
            model->HICUMcth = value->rValue;
            model->HICUMcthGiven = TRUE;
            break;

//Compatibility with V2.1
        case HICUM_MOD_FLCOMP:
            model->HICUMflcomp = value->rValue;
            model->HICUMflcompGiven = TRUE;
            break;

//SOA-check
        case HICUM_MOD_VBE_MAX:
            model->HICUMvbeMax = value->rValue;
            model->HICUMvbeMaxGiven = TRUE;
            break;
        case HICUM_MOD_VBC_MAX:
            model->HICUMvbcMax = value->rValue;
            model->HICUMvbcMaxGiven = TRUE;
            break;
        case HICUM_MOD_VCE_MAX:
            model->HICUMvceMax = value->rValue;
            model->HICUMvceMaxGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
