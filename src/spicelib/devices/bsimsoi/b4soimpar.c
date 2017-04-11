/***  B4SOI 12/16/2010 Released by Tanvir Morshed    ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soimpar.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * Authors: 2010- Tanvir Morshed, Ali Niknejad, Chenming Hu.
 * File: b4soimpar.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 * Modified by Tanvir Morshed 12/16/2010
 **********/

#include "ngspice/ngspice.h"

#include "b4soidef.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B4SOImParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    B4SOImodel *mod = (B4SOImodel*)inMod;
    switch(param)
    {  
        case  B4SOI_MOD_MOBMOD :
            mod->B4SOImobMod = value->iValue;
            mod->B4SOImobModGiven = TRUE;
            break;
        case  B4SOI_MOD_BINUNIT :
            mod->B4SOIbinUnit = value->iValue;
            mod->B4SOIbinUnitGiven = TRUE;
            break;
        case  B4SOI_MOD_PARAMCHK :
            mod->B4SOIparamChk = value->iValue;
            mod->B4SOIparamChkGiven = TRUE;
            break;
        case  B4SOI_MOD_CAPMOD :
            mod->B4SOIcapMod = value->iValue;
            mod->B4SOIcapModGiven = TRUE;
            break;
        case  B4SOI_MOD_SHMOD :
            mod->B4SOIshMod = value->iValue;
            mod->B4SOIshModGiven = TRUE;
            break;

/*        case  B4SOI_MOD_NOIMOD :
            mod->B4SOInoiMod = value->iValue;
            mod->B4SOInoiModGiven = TRUE;
            break;  v3.2 */

        case  B4SOI_MOD_VERSION :
            mod->B4SOIversion = value->rValue;
            mod->B4SOIversionGiven = TRUE;
            break;
                case  B4SOI_MOD_MTRLMOD :
            mod->B4SOImtrlMod = value->iValue;
            mod->B4SOImtrlModGiven = TRUE;
            break;
                case  B4SOI_MOD_VGSTCVMOD :
            mod->B4SOIvgstcvMod = value->iValue;
            mod->B4SOIvgstcvModGiven = TRUE;
            break;        
                case  B4SOI_MOD_GIDLMOD :
            mod->B4SOIgidlMod = value->iValue;
            mod->B4SOIgidlModGiven = TRUE;
            break;
                case  B4SOI_MOD_IIIMOD :
            mod->B4SOIiiiMod = value->iValue;
            mod->B4SOIiiiModGiven = TRUE;
            break;
        case  B4SOI_MOD_TOX :
            mod->B4SOItox = value->rValue;
            mod->B4SOItoxGiven = TRUE;
            break;
        case  B4SOI_MOD_TOXP :
            mod->B4SOItoxp = value->rValue;
            mod->B4SOItoxpGiven = TRUE;
            break;
        case  B4SOI_MOD_LEFFEOT :
            mod->B4SOIleffeot = value->rValue;
            mod->B4SOIleffeotGiven = TRUE;
            break;
        case  B4SOI_MOD_WEFFEOT :
            mod->B4SOIweffeot = value->rValue;
            mod->B4SOIweffeotGiven = TRUE;
            break;
        case  B4SOI_MOD_VDDEOT :
            mod->B4SOIvddeot = value->rValue;
            mod->B4SOIvddeotGiven = TRUE;
            break;
        case  B4SOI_MOD_TEMPEOT :
            mod->B4SOItempeot = value->rValue;
            mod->B4SOItempeotGiven = TRUE;
            break;
        case  B4SOI_MOD_ADOS :
            mod->B4SOIados = value->rValue;
            mod->B4SOIadosGiven = TRUE;
            break;
        case  B4SOI_MOD_BDOS :
            mod->B4SOIbdos = value->rValue;
            mod->B4SOIbdosGiven = TRUE;
            break;
        case B4SOI_MOD_EPSRGATE:
            mod->B4SOIepsrgate = value->rValue;
            mod->B4SOIepsrgateGiven = TRUE;
                break;
        case B4SOI_MOD_PHIG:
            mod->B4SOIphig = value->rValue;
            mod->B4SOIphigGiven = TRUE;
                break;
        case B4SOI_MOD_EASUB:
            mod->B4SOIeasub = value->rValue;
            mod->B4SOIeasubGiven = TRUE;
            break;                
                
        case  B4SOI_MOD_TOXM :
            mod->B4SOItoxm = value->rValue;
            mod->B4SOItoxmGiven = TRUE;
            break; /* v3.2 */
                /*4.1        */
        case  B4SOI_MOD_EOT :
            mod->B4SOIeot = value->rValue;
            mod->B4SOIeotGiven = TRUE;
            break;
        case  B4SOI_MOD_EPSROX :
            mod->B4SOIepsrox = value->rValue;
            mod->B4SOIepsroxGiven = TRUE;
            break;
        case B4SOI_MOD_EPSRSUB:
            mod->B4SOIepsrsub = value->rValue;
            mod->B4SOIepsrsubGiven = TRUE;
            break;
        case B4SOI_MOD_NI0SUB:
            mod->B4SOIni0sub = value->rValue;
            mod->B4SOIni0subGiven = TRUE;
            break;
        case B4SOI_MOD_BG0SUB:
            mod->B4SOIbg0sub = value->rValue;
            mod->B4SOIbg0subGiven = TRUE;
            break;
        case B4SOI_MOD_TBGASUB:
            mod->B4SOItbgasub = value->rValue;
            mod->B4SOItbgasubGiven = TRUE;
            break;
        case B4SOI_MOD_TBGBSUB:
            mod->B4SOItbgbsub = value->rValue;
            mod->B4SOItbgbsubGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_DTOXCV :
            mod->B4SOIdtoxcv = value->rValue;
            mod->B4SOIdtoxcvGiven = TRUE;
            break;

        case  B4SOI_MOD_CDSC :
            mod->B4SOIcdsc = value->rValue;
            mod->B4SOIcdscGiven = TRUE;
            break;
        case  B4SOI_MOD_CDSCB :
            mod->B4SOIcdscb = value->rValue;
            mod->B4SOIcdscbGiven = TRUE;
            break;

        case  B4SOI_MOD_CDSCD :
            mod->B4SOIcdscd = value->rValue;
            mod->B4SOIcdscdGiven = TRUE;
            break;

        case  B4SOI_MOD_CIT :
            mod->B4SOIcit = value->rValue;
            mod->B4SOIcitGiven = TRUE;
            break;
        case  B4SOI_MOD_NFACTOR :
            mod->B4SOInfactor = value->rValue;
            mod->B4SOInfactorGiven = TRUE;
            break;
        case B4SOI_MOD_VSAT:
            mod->B4SOIvsat = value->rValue;
            mod->B4SOIvsatGiven = TRUE;
            break;
        case B4SOI_MOD_A0:
            mod->B4SOIa0 = value->rValue;
            mod->B4SOIa0Given = TRUE;
            break;
        
        case B4SOI_MOD_AGS:
            mod->B4SOIags= value->rValue;
            mod->B4SOIagsGiven = TRUE;
            break;
        
        case B4SOI_MOD_A1:
            mod->B4SOIa1 = value->rValue;
            mod->B4SOIa1Given = TRUE;
            break;
        case B4SOI_MOD_A2:
            mod->B4SOIa2 = value->rValue;
            mod->B4SOIa2Given = TRUE;
            break;
        case B4SOI_MOD_AT:
            mod->B4SOIat = value->rValue;
            mod->B4SOIatGiven = TRUE;
            break;
        case B4SOI_MOD_KETA:
            mod->B4SOIketa = value->rValue;
            mod->B4SOIketaGiven = TRUE;
            break;    
        case B4SOI_MOD_NSUB:
            mod->B4SOInsub = value->rValue;
            mod->B4SOInsubGiven = TRUE;
            break;
        case B4SOI_MOD_NPEAK:
            mod->B4SOInpeak = value->rValue;
            mod->B4SOInpeakGiven = TRUE;
        /* Bug # 22 Jul09 Proper limiting conditions are specified in the B4SOIcheck.c file*/
            /* if (mod->B4SOInpeak > 1.0e20)                                        
                mod->B4SOInpeak *= 1.0e-6; */
            break;
        case B4SOI_MOD_NSD:
            mod->B4SOInsd = value->rValue;
            mod->B4SOInsdGiven = TRUE;
           /* if (mod->B4SOInsd > 1.0e23)
                mod->B4SOInsd *= 1.0e-6;  */                                        /* Bug # 22 Jul09 Proper limiting conditions are specified in the B4SOIcheck.c file*/
            break;
        case B4SOI_MOD_NGATE:
            mod->B4SOIngate = value->rValue;
            mod->B4SOIngateGiven = TRUE;
            /* if (mod->B4SOIngate > 1.0e23)
                mod->B4SOIngate *= 1.0e-6;                         */                                        /* Bug # 22 Jul09 Proper limiting conditions are specified in the B4SOIcheck.c file*/
            break;
        case B4SOI_MOD_GAMMA1:
            mod->B4SOIgamma1 = value->rValue;
            mod->B4SOIgamma1Given = TRUE;
            break;
        case B4SOI_MOD_GAMMA2:
            mod->B4SOIgamma2 = value->rValue;
            mod->B4SOIgamma2Given = TRUE;
            break;
        case B4SOI_MOD_VBX:
            mod->B4SOIvbx = value->rValue;
            mod->B4SOIvbxGiven = TRUE;
            break;
        case B4SOI_MOD_VBM:
            mod->B4SOIvbm = value->rValue;
            mod->B4SOIvbmGiven = TRUE;
            break;
        case B4SOI_MOD_XT:
            mod->B4SOIxt = value->rValue;
            mod->B4SOIxtGiven = TRUE;
            break;
        case  B4SOI_MOD_K1:
            mod->B4SOIk1 = value->rValue;
            mod->B4SOIk1Given = TRUE;
            break;
        case  B4SOI_MOD_KT1:
            mod->B4SOIkt1 = value->rValue;
            mod->B4SOIkt1Given = TRUE;
            break;
        case  B4SOI_MOD_KT1L:
            mod->B4SOIkt1l = value->rValue;
            mod->B4SOIkt1lGiven = TRUE;
            break;
        case  B4SOI_MOD_KT2:
            mod->B4SOIkt2 = value->rValue;
            mod->B4SOIkt2Given = TRUE;
            break;
        case  B4SOI_MOD_K2:
            mod->B4SOIk2 = value->rValue;
            mod->B4SOIk2Given = TRUE;
            break;
        case  B4SOI_MOD_K3:
            mod->B4SOIk3 = value->rValue;
            mod->B4SOIk3Given = TRUE;
            break;
        case  B4SOI_MOD_K3B:
            mod->B4SOIk3b = value->rValue;
            mod->B4SOIk3bGiven = TRUE;
            break;
        case  B4SOI_MOD_LPE0:
            mod->B4SOIlpe0 = value->rValue;
            mod->B4SOIlpe0Given = TRUE;
            break;
        case  B4SOI_MOD_LPEB:        /* v4.0 for Vth */
            mod->B4SOIlpeb = value->rValue;
            mod->B4SOIlpebGiven = TRUE;
            break;
        case  B4SOI_MOD_W0:
            mod->B4SOIw0 = value->rValue;
            mod->B4SOIw0Given = TRUE;
            break;
        case  B4SOI_MOD_DVT0:               
            mod->B4SOIdvt0 = value->rValue;
            mod->B4SOIdvt0Given = TRUE;
            break;
        case  B4SOI_MOD_DVT1:             
            mod->B4SOIdvt1 = value->rValue;
            mod->B4SOIdvt1Given = TRUE;
            break;
        case  B4SOI_MOD_DVT2:             
            mod->B4SOIdvt2 = value->rValue;
            mod->B4SOIdvt2Given = TRUE;
            break;
        case  B4SOI_MOD_DVT0W:               
            mod->B4SOIdvt0w = value->rValue;
            mod->B4SOIdvt0wGiven = TRUE;
            break;
        case  B4SOI_MOD_DVT1W:             
            mod->B4SOIdvt1w = value->rValue;
            mod->B4SOIdvt1wGiven = TRUE;
            break;
        case  B4SOI_MOD_DVT2W:             
            mod->B4SOIdvt2w = value->rValue;
            mod->B4SOIdvt2wGiven = TRUE;
            break;
        case  B4SOI_MOD_DROUT:             
            mod->B4SOIdrout = value->rValue;
            mod->B4SOIdroutGiven = TRUE;
            break;
        case  B4SOI_MOD_DSUB:             
            mod->B4SOIdsub = value->rValue;
            mod->B4SOIdsubGiven = TRUE;
            break;
        case B4SOI_MOD_VTH0:
            mod->B4SOIvth0 = value->rValue;
            mod->B4SOIvth0Given = TRUE;
            break;
        case B4SOI_MOD_VFB:
            mod->B4SOIvfb = value->rValue;
            mod->B4SOIvfbGiven = TRUE;
            break; /* v4.1 */
        case B4SOI_MOD_UA:
            mod->B4SOIua = value->rValue;
            mod->B4SOIuaGiven = TRUE;
            break;
        case B4SOI_MOD_UA1:
            mod->B4SOIua1 = value->rValue;
            mod->B4SOIua1Given = TRUE;
            break;
        case B4SOI_MOD_UB:
            mod->B4SOIub = value->rValue;
            mod->B4SOIubGiven = TRUE;
            break;
        case B4SOI_MOD_UB1:
            mod->B4SOIub1 = value->rValue;
            mod->B4SOIub1Given = TRUE;
            break;
        case B4SOI_MOD_UC:
            mod->B4SOIuc = value->rValue;
            mod->B4SOIucGiven = TRUE;
            break;
        case B4SOI_MOD_UC1:
            mod->B4SOIuc1 = value->rValue;
            mod->B4SOIuc1Given = TRUE;
            break;
        case  B4SOI_MOD_U0 :
            mod->B4SOIu0 = value->rValue;
            mod->B4SOIu0Given = TRUE;
            break;
        case  B4SOI_MOD_UTE :
            mod->B4SOIute = value->rValue;
            mod->B4SOIuteGiven = TRUE;
            break;
/*4.1 mobmod=4*/
case B4SOI_MOD_UD:
            mod->B4SOIud = value->rValue;
            mod->B4SOIudGiven = TRUE;
            break;
        case B4SOI_MOD_LUD:
            mod->B4SOIlud = value->rValue;
            mod->B4SOIludGiven = TRUE;
            break;
        case B4SOI_MOD_WUD:
            mod->B4SOIwud = value->rValue;
            mod->B4SOIwudGiven = TRUE;
            break;
        case B4SOI_MOD_PUD:
            mod->B4SOIpud = value->rValue;
            mod->B4SOIpudGiven = TRUE;
            break;
        case B4SOI_MOD_UD1:
            mod->B4SOIud1 = value->rValue;
            mod->B4SOIud1Given = TRUE;
            break;
        case B4SOI_MOD_LUD1:
            mod->B4SOIlud1 = value->rValue;
            mod->B4SOIlud1Given = TRUE;
            break;
        case B4SOI_MOD_WUD1:
            mod->B4SOIwud1 = value->rValue;
            mod->B4SOIwud1Given = TRUE;
            break;
        case B4SOI_MOD_PUD1:
            mod->B4SOIpud1 = value->rValue;
            mod->B4SOIpud1Given = TRUE;
            break;
        case B4SOI_MOD_EU:
            mod->B4SOIeu = value->rValue;
            mod->B4SOIeuGiven = TRUE;
            break;
        case B4SOI_MOD_LEU:
            mod->B4SOIleu = value->rValue;
            mod->B4SOIleuGiven = TRUE;
            break;
        case B4SOI_MOD_WEU:
            mod->B4SOIweu = value->rValue;
            mod->B4SOIweuGiven = TRUE;
            break;
        case B4SOI_MOD_PEU:
            mod->B4SOIpeu = value->rValue;
            mod->B4SOIpeuGiven = TRUE;
            break;
        case B4SOI_MOD_UCS:
            mod->B4SOIucs = value->rValue;
            mod->B4SOIucsGiven = TRUE;
                        break;
        case B4SOI_MOD_LUCS:
            mod->B4SOIlucs = value->rValue;
            mod->B4SOIlucsGiven = TRUE;
                        break;
        case B4SOI_MOD_WUCS:
            mod->B4SOIwucs = value->rValue;
            mod->B4SOIwucsGiven = TRUE;
                        break;
        case B4SOI_MOD_PUCS:
            mod->B4SOIpucs = value->rValue;
            mod->B4SOIpucsGiven = TRUE;
                        break; /* Bug fix # 31 Jul09 */        
        case B4SOI_MOD_UCSTE:
            mod->B4SOIucste = value->rValue;
            mod->B4SOIucsteGiven = TRUE;
                        break;
        case B4SOI_MOD_LUCSTE:
            mod->B4SOIlucste = value->rValue;
            mod->B4SOIlucsteGiven = TRUE;
                        break;
        case B4SOI_MOD_WUCSTE:
            mod->B4SOIwucste = value->rValue;
            mod->B4SOIwucsteGiven = TRUE;
                        break;
        case B4SOI_MOD_PUCSTE:
            mod->B4SOIpucste = value->rValue;
            mod->B4SOIpucsteGiven = TRUE;
                        break;
        case B4SOI_MOD_VOFF:
            mod->B4SOIvoff = value->rValue;
            mod->B4SOIvoffGiven = TRUE;
            break;
        case  B4SOI_MOD_DELTA :
            mod->B4SOIdelta = value->rValue;
            mod->B4SOIdeltaGiven = TRUE;
            break;
        case B4SOI_MOD_RDSW:
            mod->B4SOIrdsw = value->rValue;
            mod->B4SOIrdswGiven = TRUE;
            break;                     
        case B4SOI_MOD_RSW:
            mod->B4SOIrsw = value->rValue;
            mod->B4SOIrswGiven = TRUE;
            break;                     
        case B4SOI_MOD_RDW:
            mod->B4SOIrdw = value->rValue;
            mod->B4SOIrdwGiven = TRUE;
            break;                     
        case B4SOI_MOD_RSWMIN:
            mod->B4SOIrswmin = value->rValue;
            mod->B4SOIrswminGiven = TRUE;
            break;                     
        case B4SOI_MOD_RDWMIN:
            mod->B4SOIrdwmin = value->rValue;
            mod->B4SOIrdwminGiven = TRUE;
            break;                     
        case B4SOI_MOD_PRWG:
            mod->B4SOIprwg = value->rValue;
            mod->B4SOIprwgGiven = TRUE;
            break;                     
        case B4SOI_MOD_PRWB:
            mod->B4SOIprwb = value->rValue;
            mod->B4SOIprwbGiven = TRUE;
            break;                     
        case B4SOI_MOD_PRT:
            mod->B4SOIprt = value->rValue;
            mod->B4SOIprtGiven = TRUE;
            break;                     
        case B4SOI_MOD_ETA0:
            mod->B4SOIeta0 = value->rValue;
            mod->B4SOIeta0Given = TRUE;
            break;                 
        case B4SOI_MOD_ETAB:
            mod->B4SOIetab = value->rValue;
            mod->B4SOIetabGiven = TRUE;
            break;                 
        case B4SOI_MOD_PCLM:
            mod->B4SOIpclm = value->rValue;
            mod->B4SOIpclmGiven = TRUE;
            break;                 
        case B4SOI_MOD_PDIBL1:
            mod->B4SOIpdibl1 = value->rValue;
            mod->B4SOIpdibl1Given = TRUE;
            break;                 
        case B4SOI_MOD_PDIBL2:
            mod->B4SOIpdibl2 = value->rValue;
            mod->B4SOIpdibl2Given = TRUE;
            break;                 
        case B4SOI_MOD_PDIBLB:
            mod->B4SOIpdiblb = value->rValue;
            mod->B4SOIpdiblbGiven = TRUE;
            break;                 
        case B4SOI_MOD_PVAG:
            mod->B4SOIpvag = value->rValue;
            mod->B4SOIpvagGiven = TRUE;
            break;                 
        case  B4SOI_MOD_WR :
            mod->B4SOIwr = value->rValue;
            mod->B4SOIwrGiven = TRUE;
            break;
        case  B4SOI_MOD_DWG :
            mod->B4SOIdwg = value->rValue;
            mod->B4SOIdwgGiven = TRUE;
            break;
        case  B4SOI_MOD_DWB :
            mod->B4SOIdwb = value->rValue;
            mod->B4SOIdwbGiven = TRUE;
            break;
        case  B4SOI_MOD_B0 :
            mod->B4SOIb0 = value->rValue;
            mod->B4SOIb0Given = TRUE;
            break;
        case  B4SOI_MOD_B1 :
            mod->B4SOIb1 = value->rValue;
            mod->B4SOIb1Given = TRUE;
            break;
        case  B4SOI_MOD_ALPHA0 :
            mod->B4SOIalpha0 = value->rValue;
            mod->B4SOIalpha0Given = TRUE;
            break;

        case  B4SOI_MOD_CGSL :
            mod->B4SOIcgsl = value->rValue;
            mod->B4SOIcgslGiven = TRUE;
            break;
        case  B4SOI_MOD_CGDL :
            mod->B4SOIcgdl = value->rValue;
            mod->B4SOIcgdlGiven = TRUE;
            break;
        case  B4SOI_MOD_CKAPPA :
            mod->B4SOIckappa = value->rValue;
            mod->B4SOIckappaGiven = TRUE;
            break;
        case  B4SOI_MOD_CF :
            mod->B4SOIcf = value->rValue;
            mod->B4SOIcfGiven = TRUE;
            break;
        case  B4SOI_MOD_CLC :
            mod->B4SOIclc = value->rValue;
            mod->B4SOIclcGiven = TRUE;
            break;
        case  B4SOI_MOD_CLE :
            mod->B4SOIcle = value->rValue;
            mod->B4SOIcleGiven = TRUE;
            break;
        case  B4SOI_MOD_DWC :
            mod->B4SOIdwc = value->rValue;
            mod->B4SOIdwcGiven = TRUE;
            break;
        case  B4SOI_MOD_DLC :
            mod->B4SOIdlc = value->rValue;
            mod->B4SOIdlcGiven = TRUE;
            break;
        case  B4SOI_MOD_TBOX :
            mod->B4SOItbox = value->rValue;
            mod->B4SOItboxGiven = TRUE;
            break;
        case  B4SOI_MOD_TSI :
            mod->B4SOItsi = value->rValue;
            mod->B4SOItsiGiven = TRUE;
            break;
                case  B4SOI_MOD_ETSI :
            mod->B4SOIetsi = value->rValue;
            mod->B4SOIetsiGiven = TRUE;
            break;
        case  B4SOI_MOD_XJ :
            mod->B4SOIxj = value->rValue;
            mod->B4SOIxjGiven = TRUE;
            break;
        case  B4SOI_MOD_RBODY :
            mod->B4SOIrbody = value->rValue;
            mod->B4SOIrbodyGiven = TRUE;
            break;
        case  B4SOI_MOD_RBSH :
            mod->B4SOIrbsh = value->rValue;
            mod->B4SOIrbshGiven = TRUE;
            break;
        case  B4SOI_MOD_RTH0 :
            mod->B4SOIrth0 = value->rValue;
            mod->B4SOIrth0Given = TRUE;
            break;
        case  B4SOI_MOD_CTH0 :
            mod->B4SOIcth0 = value->rValue;
            mod->B4SOIcth0Given = TRUE;
            break;
        case  B4SOI_MOD_CFRCOEFF :        /* v4.4 */
            mod->B4SOIcfrcoeff = value->rValue;
            mod->B4SOIcfrcoeffGiven = TRUE;
            break;
        case  B4SOI_MOD_EGIDL :
            mod->B4SOIegidl = value->rValue;
            mod->B4SOIegidlGiven = TRUE;
            break;
        case  B4SOI_MOD_AGIDL :
            mod->B4SOIagidl = value->rValue;
            mod->B4SOIagidlGiven = TRUE;
            break;
        case  B4SOI_MOD_BGIDL :
            mod->B4SOIbgidl = value->rValue;
            mod->B4SOIbgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_CGIDL :
            mod->B4SOIcgidl = value->rValue;
            mod->B4SOIcgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_RGIDL :
            mod->B4SOIrgidl = value->rValue;
            mod->B4SOIrgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_KGIDL :
            mod->B4SOIkgidl = value->rValue;
            mod->B4SOIkgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_FGIDL :
            mod->B4SOIfgidl = value->rValue;
            mod->B4SOIfgidlGiven = TRUE;
            break;
                        
                        case  B4SOI_MOD_EGISL :
            mod->B4SOIegisl = value->rValue;
            mod->B4SOIegislGiven = TRUE;
            break;
        case  B4SOI_MOD_AGISL :
            mod->B4SOIagisl = value->rValue;
            mod->B4SOIagislGiven = TRUE;
            break;
        case  B4SOI_MOD_BGISL :
            mod->B4SOIbgisl = value->rValue;
            mod->B4SOIbgislGiven = TRUE;
            break;
        case  B4SOI_MOD_CGISL :
            mod->B4SOIcgisl = value->rValue;
            mod->B4SOIcgislGiven = TRUE;
            break;
        case  B4SOI_MOD_RGISL :
            mod->B4SOIrgisl = value->rValue;
            mod->B4SOIrgislGiven = TRUE;
            break;
        case  B4SOI_MOD_KGISL :
            mod->B4SOIkgisl = value->rValue;
            mod->B4SOIkgislGiven = TRUE;
            break;
        case  B4SOI_MOD_FGISL :
            mod->B4SOIfgisl = value->rValue;
            mod->B4SOIfgislGiven = TRUE;
            break;
        case  B4SOI_MOD_FDMOD :
           /* mod->B4SOIfdMod = value->rValue;  v4.2 */
            mod->B4SOIfdMod = value->iValue;
                        mod->B4SOIfdModGiven = TRUE;
            break; 
        case  B4SOI_MOD_VSCE :
            mod->B4SOIvsce = value->rValue;
            mod->B4SOIvsceGiven = TRUE;
            break;        
        case  B4SOI_MOD_CDSBS :
            mod->B4SOIcdsbs = value->rValue;
            mod->B4SOIcdsbsGiven = TRUE;
            break;         
        case B4SOI_MOD_MINVCV:
            mod->B4SOIminvcv = value->rValue;
            mod->B4SOIminvcvGiven = TRUE;
            break;
        case B4SOI_MOD_LMINVCV:
            mod->B4SOIlminvcv = value->rValue;
            mod->B4SOIlminvcvGiven = TRUE;
            break;
        case B4SOI_MOD_WMINVCV:
            mod->B4SOIwminvcv = value->rValue;
            mod->B4SOIwminvcvGiven = TRUE; 
            break;
        case B4SOI_MOD_PMINVCV:
            mod->B4SOIpminvcv = value->rValue;
            mod->B4SOIpminvcvGiven = TRUE;
            break;
        case B4SOI_MOD_VOFFCV:  
            mod->B4SOIvoffcv = value->rValue; 
            mod->B4SOIvoffcvGiven = TRUE; 
            break;
        case B4SOI_MOD_LVOFFCV:  
            mod->B4SOIlvoffcv = value->rValue; 
            mod->B4SOIlvoffcvGiven = TRUE; 
            break;
        case B4SOI_MOD_WVOFFCV:  
            mod->B4SOIwvoffcv = value->rValue; 
            mod->B4SOIwvoffcvGiven = TRUE; 
            break;
        case B4SOI_MOD_PVOFFCV:  
            mod->B4SOIpvoffcv = value->rValue; 
            mod->B4SOIpvoffcvGiven = TRUE; 
            break;
                        
        case  B4SOI_MOD_NDIODES : /* v4.0 */
            mod->B4SOIndiode = value->rValue;
            mod->B4SOIndiodeGiven = TRUE;
            break;
        case  B4SOI_MOD_NDIODED : /* v4.0 */
            mod->B4SOIndioded = value->rValue;
            mod->B4SOIndiodedGiven = TRUE;
            break;
        case  B4SOI_MOD_XBJT :
            mod->B4SOIxbjt = value->rValue;
            mod->B4SOIxbjtGiven = TRUE;
            break;

        case  B4SOI_MOD_XDIFS :
            mod->B4SOIxdif = value->rValue;
            mod->B4SOIxdifGiven = TRUE;
            break;
        case  B4SOI_MOD_XRECS :
            mod->B4SOIxrec = value->rValue;
            mod->B4SOIxrecGiven = TRUE;
            break;
        case  B4SOI_MOD_XTUNS :
            mod->B4SOIxtun = value->rValue;
            mod->B4SOIxtunGiven = TRUE;
            break;
        case  B4SOI_MOD_XDIFD :
            mod->B4SOIxdifd = value->rValue;
            mod->B4SOIxdifdGiven = TRUE;
            break;
        case  B4SOI_MOD_XRECD :
            mod->B4SOIxrecd = value->rValue;
            mod->B4SOIxrecdGiven = TRUE;
            break;
        case  B4SOI_MOD_XTUND :
            mod->B4SOIxtund = value->rValue;
            mod->B4SOIxtundGiven = TRUE;
            break;
        case  B4SOI_MOD_TT :
            mod->B4SOItt = value->rValue;
            mod->B4SOIttGiven = TRUE;
            break;
        case  B4SOI_MOD_VSDTH :
            mod->B4SOIvsdth = value->rValue;
            mod->B4SOIvsdthGiven = TRUE;
            break;
        case  B4SOI_MOD_VSDFB :
            mod->B4SOIvsdfb = value->rValue;
            mod->B4SOIvsdfbGiven = TRUE;
            break;
        case  B4SOI_MOD_CSDMIN :
            mod->B4SOIcsdmin = value->rValue;
            mod->B4SOIcsdminGiven = TRUE;
            break;
        case  B4SOI_MOD_ASD :
            mod->B4SOIasd = value->rValue;
            mod->B4SOIasdGiven = TRUE;
            break;


        case  B4SOI_MOD_TNOM :
            mod->B4SOItnom = value->rValue + 273.15;
            mod->B4SOItnomGiven = TRUE;
            break;
        case  B4SOI_MOD_CGSO :
            mod->B4SOIcgso = value->rValue;
            mod->B4SOIcgsoGiven = TRUE;
            break;
        case  B4SOI_MOD_CGDO :
            mod->B4SOIcgdo = value->rValue;
            mod->B4SOIcgdoGiven = TRUE;
            break;
        case  B4SOI_MOD_CGEO :
            mod->B4SOIcgeo = value->rValue;
            mod->B4SOIcgeoGiven = TRUE;
            break;


        case  B4SOI_MOD_XPART :
            mod->B4SOIxpart = value->rValue;
            mod->B4SOIxpartGiven = TRUE;
            break;
        case  B4SOI_MOD_RSH :
            mod->B4SOIsheetResistance = value->rValue;
            mod->B4SOIsheetResistanceGiven = TRUE;
            break;
        case  B4SOI_MOD_PBSWGS :        /* v4.0 */
            mod->B4SOIGatesidewallJctSPotential = value->rValue;
            mod->B4SOIGatesidewallJctSPotentialGiven = TRUE;
            break;
        case  B4SOI_MOD_PBSWGD :        /* v4.0 */
            mod->B4SOIGatesidewallJctDPotential = value->rValue;
            mod->B4SOIGatesidewallJctDPotentialGiven = TRUE;
            break;
        case  B4SOI_MOD_MJSWGS :        /* v4.0 */
            mod->B4SOIbodyJctGateSideSGradingCoeff = value->rValue;
            mod->B4SOIbodyJctGateSideSGradingCoeffGiven = TRUE;
            break;
        case  B4SOI_MOD_MJSWGD :        /* v4.0 */
            mod->B4SOIbodyJctGateSideDGradingCoeff = value->rValue;
            mod->B4SOIbodyJctGateSideDGradingCoeffGiven = TRUE;
            break;
        case  B4SOI_MOD_CJSWGS :        /* v4.0 */
            mod->B4SOIunitLengthGateSidewallJctCapS = value->rValue;
            mod->B4SOIunitLengthGateSidewallJctCapSGiven = TRUE;
            break;
        case  B4SOI_MOD_CJSWGD :        /* v4.0 */
            mod->B4SOIunitLengthGateSidewallJctCapD = value->rValue;
            mod->B4SOIunitLengthGateSidewallJctCapDGiven = TRUE;
            break;
        case  B4SOI_MOD_CSDESW :
            mod->B4SOIcsdesw = value->rValue;
            mod->B4SOIcsdeswGiven = TRUE;
            break;
        case  B4SOI_MOD_LINT :
            mod->B4SOILint = value->rValue;
            mod->B4SOILintGiven = TRUE;
            break;
        case  B4SOI_MOD_LL :
            mod->B4SOILl = value->rValue;
            mod->B4SOILlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_LLC :
            mod->B4SOILlc = value->rValue;
            mod->B4SOILlcGiven = TRUE;
            break;

        case  B4SOI_MOD_LLN :
            mod->B4SOILln = value->rValue;
            mod->B4SOILlnGiven = TRUE;
            break;
        case  B4SOI_MOD_LW :
            mod->B4SOILw = value->rValue;
            mod->B4SOILwGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_LWC :
            mod->B4SOILwc = value->rValue;
            mod->B4SOILwcGiven = TRUE;
            break;

        case  B4SOI_MOD_LWN :
            mod->B4SOILwn = value->rValue;
            mod->B4SOILwnGiven = TRUE;
            break;
        case  B4SOI_MOD_LWL :
            mod->B4SOILwl = value->rValue;
            mod->B4SOILwlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_LWLC :
            mod->B4SOILwlc = value->rValue;
            mod->B4SOILwlcGiven = TRUE;
            break;

        case  B4SOI_MOD_WINT :
            mod->B4SOIWint = value->rValue;
            mod->B4SOIWintGiven = TRUE;
            break;
        case  B4SOI_MOD_WL :
            mod->B4SOIWl = value->rValue;
            mod->B4SOIWlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_WLC :
            mod->B4SOIWlc = value->rValue;
            mod->B4SOIWlcGiven = TRUE;
            break;

        case  B4SOI_MOD_WLN :
            mod->B4SOIWln = value->rValue;
            mod->B4SOIWlnGiven = TRUE;
            break;
        case  B4SOI_MOD_WW :
            mod->B4SOIWw = value->rValue;
            mod->B4SOIWwGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_WWC :
            mod->B4SOIWwc = value->rValue;
            mod->B4SOIWwcGiven = TRUE;
            break;

        case  B4SOI_MOD_WWN :
            mod->B4SOIWwn = value->rValue;
            mod->B4SOIWwnGiven = TRUE;
            break;
        case  B4SOI_MOD_WWL :
            mod->B4SOIWwl = value->rValue;
            mod->B4SOIWwlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B4SOI_MOD_WWLC :
            mod->B4SOIWwlc = value->rValue;
            mod->B4SOIWwlcGiven = TRUE;
            break;

        case  B4SOI_MOD_NOIA :
            mod->B4SOIoxideTrapDensityA = value->rValue;
            mod->B4SOIoxideTrapDensityAGiven = TRUE;
            break;
        case  B4SOI_MOD_NOIB :
            mod->B4SOIoxideTrapDensityB = value->rValue;
            mod->B4SOIoxideTrapDensityBGiven = TRUE;
            break;
        case  B4SOI_MOD_NOIC :
            mod->B4SOIoxideTrapDensityC = value->rValue;
            mod->B4SOIoxideTrapDensityCGiven = TRUE;
            break;
        case  B4SOI_MOD_NOIF :
            mod->B4SOInoif = value->rValue;
            mod->B4SOInoifGiven = TRUE;
            break;
        case  B4SOI_MOD_EM :
            mod->B4SOIem = value->rValue;
            mod->B4SOIemGiven = TRUE;
            break;
        case  B4SOI_MOD_EF :
            mod->B4SOIef = value->rValue;
            mod->B4SOIefGiven = TRUE;
            break;
        case  B4SOI_MOD_AF :
            mod->B4SOIaf = value->rValue;
            mod->B4SOIafGiven = TRUE;
            break;
        case  B4SOI_MOD_KF :
            mod->B4SOIkf = value->rValue;
            mod->B4SOIkfGiven = TRUE;
            break;
        case  B4SOI_MOD_BF :
            mod->B4SOIbf = value->rValue;
            mod->B4SOIbfGiven = TRUE;
            break;
        case  B4SOI_MOD_W0FLK :
            mod->B4SOIw0flk = value->rValue;
            mod->B4SOIw0flkGiven = TRUE;
            break;

/* v3.0 */
        case  B4SOI_MOD_SOIMOD:
            mod->B4SOIsoiMod = value->iValue;
            mod->B4SOIsoiModGiven = TRUE;
            break; /* v3.2 bug fix */
        case  B4SOI_MOD_VBS0PD:
            mod->B4SOIvbs0pd = value->rValue;
            mod->B4SOIvbs0pdGiven = TRUE;
            break; /* v3.2 */
        case  B4SOI_MOD_VBS0FD:
            mod->B4SOIvbs0fd = value->rValue;
            mod->B4SOIvbs0fdGiven = TRUE;
            break; /* v3.2 */
        case  B4SOI_MOD_VBSA:
            mod->B4SOIvbsa = value->rValue;
            mod->B4SOIvbsaGiven = TRUE;
            break;
        case  B4SOI_MOD_NOFFFD :
            mod->B4SOInofffd = value->rValue;
            mod->B4SOInofffdGiven = TRUE;
            break;
        case  B4SOI_MOD_VOFFFD:
            mod->B4SOIvofffd = value->rValue;
            mod->B4SOIvofffdGiven = TRUE;
            break;
        case  B4SOI_MOD_K1B:
            mod->B4SOIk1b = value->rValue;
            mod->B4SOIk1bGiven = TRUE;
            break;
        case  B4SOI_MOD_K2B:
            mod->B4SOIk2b = value->rValue;
            mod->B4SOIk2bGiven = TRUE;
            break;
        case  B4SOI_MOD_DK2B:
            mod->B4SOIdk2b = value->rValue;
            mod->B4SOIdk2bGiven = TRUE;
            break;
        case  B4SOI_MOD_DVBD0:
            mod->B4SOIdvbd0 = value->rValue;
            mod->B4SOIdvbd0Given = TRUE;
            break;
        case  B4SOI_MOD_DVBD1:
            mod->B4SOIdvbd1 = value->rValue;
            mod->B4SOIdvbd1Given = TRUE;
            break;
        case  B4SOI_MOD_MOINFD:
            mod->B4SOImoinFD = value->rValue;
            mod->B4SOImoinFDGiven = TRUE;
            break;


/* v2.2 release */
        case  B4SOI_MOD_WTH0 :
            mod->B4SOIwth0 = value->rValue;
            mod->B4SOIwth0Given = TRUE;
            break;
        case  B4SOI_MOD_RHALO :
            mod->B4SOIrhalo = value->rValue;
            mod->B4SOIrhaloGiven = TRUE;
            break;
        case  B4SOI_MOD_NTOX :
            mod->B4SOIntox = value->rValue;
            mod->B4SOIntoxGiven = TRUE;
            break;
        case  B4SOI_MOD_TOXREF :
            mod->B4SOItoxref = value->rValue;
            mod->B4SOItoxrefGiven = TRUE;
            break;
        case  B4SOI_MOD_EBG :
            mod->B4SOIebg = value->rValue;
            mod->B4SOIebgGiven = TRUE;
            break;
        case  B4SOI_MOD_VEVB :
            mod->B4SOIvevb = value->rValue;
            mod->B4SOIvevbGiven = TRUE;
            break;
        case  B4SOI_MOD_ALPHAGB1 :
            mod->B4SOIalphaGB1 = value->rValue;
            mod->B4SOIalphaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_BETAGB1 :
            mod->B4SOIbetaGB1 = value->rValue;
            mod->B4SOIbetaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_VGB1 :
            mod->B4SOIvgb1 = value->rValue;
            mod->B4SOIvgb1Given = TRUE;
            break;
        case  B4SOI_MOD_VECB :
            mod->B4SOIvecb = value->rValue;
            mod->B4SOIvecbGiven = TRUE;
            break;
        case  B4SOI_MOD_ALPHAGB2 :
            mod->B4SOIalphaGB2 = value->rValue;
            mod->B4SOIalphaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_BETAGB2 :
            mod->B4SOIbetaGB2 = value->rValue;
            mod->B4SOIbetaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_VGB2 :
            mod->B4SOIvgb2 = value->rValue;
            mod->B4SOIvgb2Given = TRUE;
            break;
        case  B4SOI_MOD_AIGBCP2 :
            mod->B4SOIaigbcp2 = value->rValue;
            mod->B4SOIaigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_BIGBCP2 :
            mod->B4SOIbigbcp2 = value->rValue;
            mod->B4SOIbigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_CIGBCP2 :
            mod->B4SOIcigbcp2 = value->rValue;
            mod->B4SOIcigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_TOXQM :
            mod->B4SOItoxqm = value->rValue;
            mod->B4SOItoxqmGiven = TRUE;
            break;
        case  B4SOI_MOD_VOXH :
            mod->B4SOIvoxh = value->rValue;
            mod->B4SOIvoxhGiven = TRUE;
            break;
        case  B4SOI_MOD_DELTAVOX :
            mod->B4SOIdeltavox = value->rValue;
            mod->B4SOIdeltavoxGiven = TRUE;
            break;

/* v3.0 */
        case  B4SOI_MOD_IGBMOD :
            mod->B4SOIigbMod = value->iValue;
            mod->B4SOIigbModGiven = TRUE;
            break;
        case  B4SOI_MOD_IGCMOD :
            mod->B4SOIigcMod = value->iValue;
            mod->B4SOIigcModGiven = TRUE;
            break;
        case  B4SOI_MOD_AIGC :
            mod->B4SOIaigc = value->rValue;
            mod->B4SOIaigcGiven = TRUE;
            break;
        case  B4SOI_MOD_BIGC :
            mod->B4SOIbigc = value->rValue;
            mod->B4SOIbigcGiven = TRUE;
            break;
        case  B4SOI_MOD_CIGC :
            mod->B4SOIcigc = value->rValue;
            mod->B4SOIcigcGiven = TRUE;
            break;
        case  B4SOI_MOD_AIGSD :
            mod->B4SOIaigsd = value->rValue;
            mod->B4SOIaigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_BIGSD :
            mod->B4SOIbigsd = value->rValue;
            mod->B4SOIbigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_CIGSD :
            mod->B4SOIcigsd = value->rValue;
            mod->B4SOIcigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_NIGC :
            mod->B4SOInigc = value->rValue;
            mod->B4SOInigcGiven = TRUE;
            break;
        case  B4SOI_MOD_PIGCD :
            mod->B4SOIpigcd = value->rValue;
            mod->B4SOIpigcdGiven = TRUE;
            break;
        case  B4SOI_MOD_POXEDGE :
            mod->B4SOIpoxedge = value->rValue;
            mod->B4SOIpoxedgeGiven = TRUE;
            break;
        case  B4SOI_MOD_DLCIG :
            mod->B4SOIdlcig = value->rValue;
            mod->B4SOIdlcigGiven = TRUE;
            break;

        /* v3.1 added for RF */
        case B4SOI_MOD_RGATEMOD :
            mod->B4SOIrgateMod = value->iValue;
            mod->B4SOIrgateModGiven = TRUE;
            break;
        case B4SOI_MOD_XRCRG1 :
            mod->B4SOIxrcrg1 = value->rValue;
            mod->B4SOIxrcrg1Given = TRUE;
            break;
        case B4SOI_MOD_XRCRG2 :
            mod->B4SOIxrcrg2 = value->rValue;
            mod->B4SOIxrcrg2Given = TRUE;
            break;
        case B4SOI_MOD_RSHG :
            mod->B4SOIrshg = value->rValue;
            mod->B4SOIrshgGiven = TRUE;
            break;
        case B4SOI_MOD_NGCON :
            mod->B4SOIngcon = value->rValue;
            mod->B4SOIngconGiven = TRUE;
            break;
        case  B4SOI_MOD_XGW :
            mod->B4SOIxgw = value->rValue;
            mod->B4SOIxgwGiven = TRUE;
            break;
        case  B4SOI_MOD_XGL :
            mod->B4SOIxgl = value->rValue;
            mod->B4SOIxglGiven = TRUE;
            break;
        /* v3.1 added RF end */

        /* v4.0 */
        case  B4SOI_MOD_RDSMOD :
            mod->B4SOIrdsMod = value->iValue;
            mod->B4SOIrdsModGiven = TRUE;
            break;
        case  B4SOI_MOD_GBMIN :
            mod->B4SOIgbmin = value->rValue;
            mod->B4SOIgbminGiven = TRUE;
            break;
        case  B4SOI_MOD_RBODYMOD :
            mod->B4SOIrbodyMod = value->iValue;
            mod->B4SOIrbodyModGiven = TRUE;
            break;
        case  B4SOI_MOD_RBDB :
            mod->B4SOIrbdb = value->rValue;
            mod->B4SOIrbdbGiven = TRUE;
            break; /* Bug fix # 31 Jul 09 */                 
         case  B4SOI_MOD_RBSB :
            mod->B4SOIrbsb = value->rValue;
            mod->B4SOIrbsbGiven = TRUE;
            break;
        case  B4SOI_MOD_FRBODY :
            mod->B4SOIfrbody = value->rValue;
            mod->B4SOIfrbodyGiven = TRUE;
            break;
        case  B4SOI_MOD_DVTP0:
            mod->B4SOIdvtp0 = value->rValue;
            mod->B4SOIdvtp0Given = TRUE;
            break;
        case  B4SOI_MOD_DVTP1:
            mod->B4SOIdvtp1 = value->rValue;
            mod->B4SOIdvtp1Given = TRUE;
            break;
        case  B4SOI_MOD_DVTP2:
            mod->B4SOIdvtp2 = value->rValue;
            mod->B4SOIdvtp2Given = TRUE;
            break;
        case  B4SOI_MOD_DVTP3:
            mod->B4SOIdvtp3 = value->rValue;
            mod->B4SOIdvtp3Given = TRUE;
            break;
        case  B4SOI_MOD_DVTP4:
            mod->B4SOIdvtp4 = value->rValue;
            mod->B4SOIdvtp4Given = TRUE;
            break;
        case  B4SOI_MOD_LDVTP0:
            mod->B4SOIldvtp0 = value->rValue;
            mod->B4SOIldvtp0Given = TRUE;
            break;
        case  B4SOI_MOD_LDVTP1:
            mod->B4SOIldvtp1 = value->rValue;
            mod->B4SOIldvtp1Given = TRUE;
            break;
        case  B4SOI_MOD_LDVTP2:
            mod->B4SOIldvtp2 = value->rValue;
            mod->B4SOIldvtp2Given = TRUE;
            break;
        case  B4SOI_MOD_LDVTP3:
            mod->B4SOIldvtp3 = value->rValue;
            mod->B4SOIldvtp3Given = TRUE;
            break;
        case  B4SOI_MOD_LDVTP4:
            mod->B4SOIldvtp4 = value->rValue;
            mod->B4SOIldvtp4Given = TRUE;
            break;
        case  B4SOI_MOD_WDVTP0:
            mod->B4SOIwdvtp0 = value->rValue;
            mod->B4SOIwdvtp0Given = TRUE;
            break;
        case  B4SOI_MOD_WDVTP1:
            mod->B4SOIwdvtp1 = value->rValue;
            mod->B4SOIwdvtp1Given = TRUE;
            break;
        case  B4SOI_MOD_WDVTP2:
            mod->B4SOIwdvtp2 = value->rValue;
            mod->B4SOIwdvtp2Given = TRUE;
            break;
        case  B4SOI_MOD_WDVTP3:
            mod->B4SOIwdvtp3 = value->rValue;
            mod->B4SOIwdvtp3Given = TRUE;
            break;
        case  B4SOI_MOD_WDVTP4:
            mod->B4SOIwdvtp4 = value->rValue;
            mod->B4SOIwdvtp4Given = TRUE;
            break;
        case  B4SOI_MOD_PDVTP0:
            mod->B4SOIpdvtp0 = value->rValue;
            mod->B4SOIpdvtp0Given = TRUE;
            break;
        case  B4SOI_MOD_PDVTP1:
            mod->B4SOIpdvtp1 = value->rValue;
            mod->B4SOIpdvtp1Given = TRUE;
            break;
        case  B4SOI_MOD_PDVTP2:
            mod->B4SOIpdvtp2 = value->rValue;
            mod->B4SOIpdvtp2Given = TRUE;
            break;
        case  B4SOI_MOD_PDVTP3:
            mod->B4SOIpdvtp3 = value->rValue;
            mod->B4SOIpdvtp3Given = TRUE;
            break;
        case  B4SOI_MOD_PDVTP4:
            mod->B4SOIpdvtp4 = value->rValue;
            mod->B4SOIpdvtp4Given = TRUE;
            break;
        case B4SOI_MOD_MINV:
            mod->B4SOIminv = value->rValue;
            mod->B4SOIminvGiven = TRUE;
            break;
        case B4SOI_MOD_LMINV:
            mod->B4SOIlminv = value->rValue;
            mod->B4SOIlminvGiven = TRUE;
            break;
        case B4SOI_MOD_WMINV:
            mod->B4SOIwminv = value->rValue;
            mod->B4SOIwminvGiven = TRUE;
            break;
        case B4SOI_MOD_PMINV:
            mod->B4SOIpminv = value->rValue;
            mod->B4SOIpminvGiven = TRUE;
            break;
        case B4SOI_MOD_FPROUT:
            mod->B4SOIfprout = value->rValue;
            mod->B4SOIfproutGiven = TRUE;
            break;
        case B4SOI_MOD_PDITS:
            mod->B4SOIpdits = value->rValue;
            mod->B4SOIpditsGiven = TRUE;
            break;
        case B4SOI_MOD_PDITSD:
            mod->B4SOIpditsd = value->rValue;
            mod->B4SOIpditsdGiven = TRUE;
            break;
        case B4SOI_MOD_PDITSL:
            mod->B4SOIpditsl = value->rValue;
            mod->B4SOIpditslGiven = TRUE;
            break;
        case B4SOI_MOD_LFPROUT:
            mod->B4SOIlfprout = value->rValue;
            mod->B4SOIlfproutGiven = TRUE;
            break;
        case B4SOI_MOD_LPDITS:
            mod->B4SOIlpdits = value->rValue;
            mod->B4SOIlpditsGiven = TRUE;
            break;
        case B4SOI_MOD_LPDITSD:
            mod->B4SOIlpditsd = value->rValue;
            mod->B4SOIlpditsdGiven = TRUE;
            break;
        case B4SOI_MOD_WFPROUT:
            mod->B4SOIwfprout = value->rValue;
            mod->B4SOIwfproutGiven = TRUE;
            break;
        case B4SOI_MOD_WPDITS:
            mod->B4SOIwpdits = value->rValue;
            mod->B4SOIwpditsGiven = TRUE;
            break;
        case B4SOI_MOD_WPDITSD:
            mod->B4SOIwpditsd = value->rValue;
            mod->B4SOIwpditsdGiven = TRUE;
            break;
        case B4SOI_MOD_PFPROUT:
            mod->B4SOIpfprout = value->rValue;
            mod->B4SOIpfproutGiven = TRUE;
            break;
        case B4SOI_MOD_PPDITS:
            mod->B4SOIppdits = value->rValue;
            mod->B4SOIppditsGiven = TRUE;
            break;
        case B4SOI_MOD_PPDITSD:
            mod->B4SOIppditsd = value->rValue;
            mod->B4SOIppditsdGiven = TRUE;
            break;

        /* v4.0 */

        /* v4.0 stress effect */
        case  B4SOI_MOD_SAREF :
            mod->B4SOIsaref = value->rValue;
            mod->B4SOIsarefGiven = TRUE;
            break;
        case  B4SOI_MOD_SBREF :
            mod->B4SOIsbref = value->rValue;
            mod->B4SOIsbrefGiven = TRUE;
            break;
        case  B4SOI_MOD_WLOD :
            mod->B4SOIwlod = value->rValue;
            mod->B4SOIwlodGiven = TRUE;
            break;
        case  B4SOI_MOD_KU0 :
            mod->B4SOIku0 = value->rValue;
            mod->B4SOIku0Given = TRUE;
            break;
        case  B4SOI_MOD_KVSAT :
            mod->B4SOIkvsat = value->rValue;
            mod->B4SOIkvsatGiven = TRUE;
            break;
        case  B4SOI_MOD_KVTH0 :
            mod->B4SOIkvth0 = value->rValue;
            mod->B4SOIkvth0Given = TRUE;
            break;
        case  B4SOI_MOD_TKU0 :
            mod->B4SOItku0 = value->rValue;
            mod->B4SOItku0Given = TRUE;
            break;
        case  B4SOI_MOD_LLODKU0 :
            mod->B4SOIllodku0 = value->rValue;
            mod->B4SOIllodku0Given = TRUE;
            break;
        case  B4SOI_MOD_WLODKU0 :
            mod->B4SOIwlodku0 = value->rValue;
            mod->B4SOIwlodku0Given = TRUE;
            break;
        case  B4SOI_MOD_LLODVTH :
            mod->B4SOIllodvth = value->rValue;
            mod->B4SOIllodvthGiven = TRUE;
            break;
        case  B4SOI_MOD_WLODVTH :
            mod->B4SOIwlodvth = value->rValue;
            mod->B4SOIwlodvthGiven = TRUE;
            break;
        case  B4SOI_MOD_LKU0 :
            mod->B4SOIlku0 = value->rValue;
            mod->B4SOIlku0Given = TRUE;
            break;
        case  B4SOI_MOD_WKU0 :
            mod->B4SOIwku0 = value->rValue;
            mod->B4SOIwku0Given = TRUE;
            break;
        case  B4SOI_MOD_PKU0 :
            mod->B4SOIpku0 = value->rValue;
            mod->B4SOIpku0Given = TRUE;
            break;
        case  B4SOI_MOD_LKVTH0 :
            mod->B4SOIlkvth0 = value->rValue;
            mod->B4SOIlkvth0Given = TRUE;
            break;
        case  B4SOI_MOD_WKVTH0 :
            mod->B4SOIwkvth0 = value->rValue;
            mod->B4SOIwkvth0Given = TRUE;
            break;
        case  B4SOI_MOD_PKVTH0 :
            mod->B4SOIpkvth0 = value->rValue;
            mod->B4SOIpkvth0Given = TRUE;
            break;
        case  B4SOI_MOD_STK2 :
            mod->B4SOIstk2 = value->rValue;
            mod->B4SOIstk2Given = TRUE;
            break;
        case  B4SOI_MOD_LODK2 :
            mod->B4SOIlodk2 = value->rValue;
            mod->B4SOIlodk2Given = TRUE;
                       break; /* Bug fix # 31 Jul 09*/ 
        case  B4SOI_MOD_STETA0 :
            mod->B4SOIsteta0 = value->rValue;
            mod->B4SOIsteta0Given = TRUE;
            break;
        case  B4SOI_MOD_LODETA0 :
            mod->B4SOIlodeta0 = value->rValue;
            mod->B4SOIlodeta0Given = TRUE;
            break;

        /* v4.0 stress effect end */

        /* v3.2 */
        case B4SOI_MOD_FNOIMOD :
            mod->B4SOIfnoiMod = value->iValue;
            mod->B4SOIfnoiModGiven = TRUE;
            break;
        case B4SOI_MOD_TNOIMOD :
            mod->B4SOItnoiMod = value->iValue;
            mod->B4SOItnoiModGiven = TRUE;
            break;
        case  B4SOI_MOD_TNOIA :
            mod->B4SOItnoia = value->rValue;
            mod->B4SOItnoiaGiven = TRUE;
            break;
        case  B4SOI_MOD_TNOIB :
            mod->B4SOItnoib = value->rValue;
            mod->B4SOItnoibGiven = TRUE;
            break;
        case  B4SOI_MOD_RNOIA :
            mod->B4SOIrnoia = value->rValue;
            mod->B4SOIrnoiaGiven = TRUE;
            break;
        case  B4SOI_MOD_RNOIB :
            mod->B4SOIrnoib = value->rValue;
            mod->B4SOIrnoibGiven = TRUE;
            break;
        case  B4SOI_MOD_NTNOI :
            mod->B4SOIntnoi = value->rValue;
            mod->B4SOIntnoiGiven = TRUE;
            break;

        /* v3.2 end */

/* v2.0 release */
        case  B4SOI_MOD_K1W1 :         
            mod->B4SOIk1w1 = value->rValue;
            mod->B4SOIk1w1Given = TRUE;
            break;
        case  B4SOI_MOD_K1W2 :
            mod->B4SOIk1w2 = value->rValue;
            mod->B4SOIk1w2Given = TRUE;
            break;
        case  B4SOI_MOD_KETAS :
            mod->B4SOIketas = value->rValue;
            mod->B4SOIketasGiven = TRUE;
            break;
        case  B4SOI_MOD_DWBC :
            mod->B4SOIdwbc = value->rValue;
            mod->B4SOIdwbcGiven = TRUE;
            break;
        case  B4SOI_MOD_BETA0 :
            mod->B4SOIbeta0 = value->rValue;
            mod->B4SOIbeta0Given = TRUE;
            break;
        case  B4SOI_MOD_BETA1 :
            mod->B4SOIbeta1 = value->rValue;
            mod->B4SOIbeta1Given = TRUE;
            break;
        case  B4SOI_MOD_BETA2 :
            mod->B4SOIbeta2 = value->rValue;
            mod->B4SOIbeta2Given = TRUE;
            break;
        case  B4SOI_MOD_VDSATII0 :
            mod->B4SOIvdsatii0 = value->rValue;
            mod->B4SOIvdsatii0Given = TRUE;
            break;
        case  B4SOI_MOD_TII :
            mod->B4SOItii = value->rValue;
            mod->B4SOItiiGiven = TRUE;
            break;
        case  B4SOI_MOD_TVBCI :
            mod->B4SOItvbci = value->rValue;
            mod->B4SOItvbciGiven = TRUE;
            break;
        case  B4SOI_MOD_LII :
            mod->B4SOIlii = value->rValue;
            mod->B4SOIliiGiven = TRUE;
            break;
        case  B4SOI_MOD_SII0 :
            mod->B4SOIsii0 = value->rValue;
            mod->B4SOIsii0Given = TRUE;
            break;
        case  B4SOI_MOD_SII1 :
            mod->B4SOIsii1 = value->rValue;
            mod->B4SOIsii1Given = TRUE;
            break;
        case  B4SOI_MOD_SII2 :
            mod->B4SOIsii2 = value->rValue;
            mod->B4SOIsii2Given = TRUE;
            break;
        case  B4SOI_MOD_SIID :
            mod->B4SOIsiid = value->rValue;
            mod->B4SOIsiidGiven = TRUE;
            break;
        case  B4SOI_MOD_FBJTII :
            mod->B4SOIfbjtii = value->rValue;
            mod->B4SOIfbjtiiGiven = TRUE;
            break;
        /*4.1 Iii model*/
           case  B4SOI_MOD_EBJTII :
            mod->B4SOIebjtii = value->rValue;
            mod->B4SOIebjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_CBJTII :
            mod->B4SOIcbjtii = value->rValue;
            mod->B4SOIcbjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_VBCI :
            mod->B4SOIvbci = value->rValue;
            mod->B4SOIvbciGiven = TRUE;
            break;
        case  B4SOI_MOD_ABJTII :
            mod->B4SOIabjtii = value->rValue;
            mod->B4SOIabjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_MBJTII :
            mod->B4SOImbjtii = value->rValue;
            mod->B4SOImbjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_ESATII :
            mod->B4SOIesatii = value->rValue;
            mod->B4SOIesatiiGiven = TRUE;
            break;
        case  B4SOI_MOD_NTUNS :                /* v4.0 */
            mod->B4SOIntun = value->rValue;
            mod->B4SOIntunGiven = TRUE;
            break;
        case  B4SOI_MOD_NTUND :                /* v4.0 */
            mod->B4SOIntund = value->rValue;
            mod->B4SOIntundGiven = TRUE;
            break;
        case  B4SOI_MOD_NRECF0S :        /* v4.0 */
            mod->B4SOInrecf0 = value->rValue;
            mod->B4SOInrecf0Given = TRUE;
            break;
        case  B4SOI_MOD_NRECF0D :        /* v4.0 */
            mod->B4SOInrecf0d = value->rValue;
            mod->B4SOInrecf0dGiven = TRUE;
            break;
        case  B4SOI_MOD_NRECR0S :        /* v4.0 */
            mod->B4SOInrecr0 = value->rValue;
            mod->B4SOInrecr0Given = TRUE;
            break;
        case  B4SOI_MOD_NRECR0D :        /* v4.0 */
            mod->B4SOInrecr0d = value->rValue;
            mod->B4SOInrecr0dGiven = TRUE;
            break;
        case  B4SOI_MOD_ISBJT :
            mod->B4SOIisbjt = value->rValue;
            mod->B4SOIisbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_IDBJT :        /* v4.0 */
            mod->B4SOIidbjt = value->rValue;
            mod->B4SOIidbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_ISDIF :
            mod->B4SOIisdif = value->rValue;
            mod->B4SOIisdifGiven = TRUE;
            break;
        case  B4SOI_MOD_IDDIF :                /* v4.0 */
            mod->B4SOIiddif = value->rValue;
            mod->B4SOIiddifGiven = TRUE;
            break;
        case  B4SOI_MOD_ISREC :
            mod->B4SOIisrec = value->rValue;
            mod->B4SOIisrecGiven = TRUE;
            break;
        case  B4SOI_MOD_IDREC :                /* v4.0 */
            mod->B4SOIidrec = value->rValue;
            mod->B4SOIidrecGiven = TRUE;
            break;
        case  B4SOI_MOD_ISTUN :
            mod->B4SOIistun = value->rValue;
            mod->B4SOIistunGiven = TRUE;
            break;
        case  B4SOI_MOD_IDTUN :                /* v4.0 */
            mod->B4SOIidtun = value->rValue;
            mod->B4SOIidtunGiven = TRUE;
            break;
        case  B4SOI_MOD_LN :
            mod->B4SOIln = value->rValue;
            mod->B4SOIlnGiven = TRUE;
            break;
        case  B4SOI_MOD_VREC0S :        /* v4.0 */
            mod->B4SOIvrec0 = value->rValue;
            mod->B4SOIvrec0Given = TRUE;
            break;
        case  B4SOI_MOD_VREC0D :        /* v4.0 */
            mod->B4SOIvrec0d = value->rValue;
            mod->B4SOIvrec0dGiven = TRUE;
            break;
        case  B4SOI_MOD_VTUN0S :        /* v4.0 */
            mod->B4SOIvtun0 = value->rValue;
            mod->B4SOIvtun0Given = TRUE;
            break;
        case  B4SOI_MOD_VTUN0D :        /* v4.0 */
            mod->B4SOIvtun0d = value->rValue;
            mod->B4SOIvtun0dGiven = TRUE;
            break;
        case  B4SOI_MOD_NBJT :
            mod->B4SOInbjt = value->rValue;
            mod->B4SOInbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_LBJT0 :
            mod->B4SOIlbjt0 = value->rValue;
            mod->B4SOIlbjt0Given = TRUE;
            break;
        case  B4SOI_MOD_LDIF0 :
            mod->B4SOIldif0 = value->rValue;
            mod->B4SOIldif0Given = TRUE;
            break;
        case  B4SOI_MOD_VABJT :
            mod->B4SOIvabjt = value->rValue;
            mod->B4SOIvabjtGiven = TRUE;
            break;
        case  B4SOI_MOD_AELY :
            mod->B4SOIaely = value->rValue;
            mod->B4SOIaelyGiven = TRUE;
            break;
        case  B4SOI_MOD_AHLIS :        /* v4.0 */
            mod->B4SOIahli = value->rValue;
            mod->B4SOIahliGiven = TRUE;
            break;
        case  B4SOI_MOD_AHLID :        /* v4.0 */
            mod->B4SOIahlid = value->rValue;
            mod->B4SOIahlidGiven = TRUE;
            break;
        case  B4SOI_MOD_NDIF :
            mod->B4SOIndif = value->rValue;
            mod->B4SOIndifGiven = TRUE;
            break;
        case  B4SOI_MOD_NTRECF :
            mod->B4SOIntrecf = value->rValue;
            mod->B4SOIntrecfGiven = TRUE;
            break;
        case  B4SOI_MOD_NTRECR :
            mod->B4SOIntrecr = value->rValue;
            mod->B4SOIntrecrGiven = TRUE;
            break;
        case  B4SOI_MOD_DLCB :
            mod->B4SOIdlcb = value->rValue;
            mod->B4SOIdlcbGiven = TRUE;
            break;
        case  B4SOI_MOD_FBODY :
            mod->B4SOIfbody = value->rValue;
            mod->B4SOIfbodyGiven = TRUE;
            break;
        case  B4SOI_MOD_TCJSWGS :
            mod->B4SOItcjswg = value->rValue;
            mod->B4SOItcjswgGiven = TRUE;
            break;
        case  B4SOI_MOD_TPBSWGS :
            mod->B4SOItpbswg = value->rValue;
            mod->B4SOItpbswgGiven = TRUE;
            break;
        case  B4SOI_MOD_TCJSWGD :
            mod->B4SOItcjswgd = value->rValue;
            mod->B4SOItcjswgdGiven = TRUE;
            break;
        case  B4SOI_MOD_TPBSWGD :
            mod->B4SOItpbswgd = value->rValue;
            mod->B4SOItpbswgdGiven = TRUE;
            break;


        case  B4SOI_MOD_ACDE :
            mod->B4SOIacde = value->rValue;
            mod->B4SOIacdeGiven = TRUE;
            break;
        case  B4SOI_MOD_MOIN :
            mod->B4SOImoin = value->rValue;
            mod->B4SOImoinGiven = TRUE;
            break;
        case  B4SOI_MOD_NOFF :
            mod->B4SOInoff = value->rValue;
            mod->B4SOInoffGiven = TRUE;
            break; /* v3.2 */
        case  B4SOI_MOD_DELVT :
            mod->B4SOIdelvt = value->rValue;
            mod->B4SOIdelvtGiven = TRUE;
            break;
        case  B4SOI_MOD_KB1 :
            mod->B4SOIkb1 = value->rValue;
            mod->B4SOIkb1Given = TRUE;
            break;
        case  B4SOI_MOD_DLBG :
            mod->B4SOIdlbg = value->rValue;
            mod->B4SOIdlbgGiven = TRUE;
            break;

/* Added for binning - START */
        /* Length Dependence */
/* v3.1 */
        case  B4SOI_MOD_LXJ :
            mod->B4SOIlxj = value->rValue;
            mod->B4SOIlxjGiven = TRUE;
            break;
        case  B4SOI_MOD_LALPHAGB1 :
            mod->B4SOIlalphaGB1 = value->rValue;
            mod->B4SOIlalphaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_LALPHAGB2 :
            mod->B4SOIlalphaGB2 = value->rValue;
            mod->B4SOIlalphaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_LBETAGB1 :
            mod->B4SOIlbetaGB1 = value->rValue;
            mod->B4SOIlbetaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_LBETAGB2 :
            mod->B4SOIlbetaGB2 = value->rValue;
            mod->B4SOIlbetaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_LAIGBCP2 :
            mod->B4SOIlaigbcp2 = value->rValue;
            mod->B4SOIlaigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_LBIGBCP2 :
            mod->B4SOIlbigbcp2 = value->rValue;
            mod->B4SOIlbigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_LCIGBCP2 :
            mod->B4SOIlcigbcp2 = value->rValue;
            mod->B4SOIlcigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_LNDIF :
            mod->B4SOIlndif = value->rValue;
            mod->B4SOIlndifGiven = TRUE;
            break;
        case  B4SOI_MOD_LNTRECF :
            mod->B4SOIlntrecf = value->rValue;
            mod->B4SOIlntrecfGiven = TRUE;
            break;
        case  B4SOI_MOD_LNTRECR :
            mod->B4SOIlntrecr = value->rValue;
            mod->B4SOIlntrecrGiven = TRUE;
            break;
        case  B4SOI_MOD_LXBJT :
            mod->B4SOIlxbjt = value->rValue;
            mod->B4SOIlxbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_LXDIFS :
            mod->B4SOIlxdif = value->rValue;
            mod->B4SOIlxdifGiven = TRUE;
            break;
        case  B4SOI_MOD_LXRECS :
            mod->B4SOIlxrec = value->rValue;
            mod->B4SOIlxrecGiven = TRUE;
            break;
        case  B4SOI_MOD_LXTUNS :
            mod->B4SOIlxtun = value->rValue;
            mod->B4SOIlxtunGiven = TRUE;
            break;
        case  B4SOI_MOD_LXDIFD :
            mod->B4SOIlxdifd = value->rValue;
            mod->B4SOIlxdifdGiven = TRUE;
            break;
        case  B4SOI_MOD_LXRECD :
            mod->B4SOIlxrecd = value->rValue;
            mod->B4SOIlxrecdGiven = TRUE;
            break;
        case  B4SOI_MOD_LXTUND :
            mod->B4SOIlxtund = value->rValue;
            mod->B4SOIlxtundGiven = TRUE;
            break;
        case  B4SOI_MOD_LCGDL :
            mod->B4SOIlcgdl = value->rValue;
            mod->B4SOIlcgdlGiven = TRUE;
            break;
        case  B4SOI_MOD_LCGSL :
            mod->B4SOIlcgsl = value->rValue;
            mod->B4SOIlcgslGiven = TRUE;
            break;
        case  B4SOI_MOD_LCKAPPA :
            mod->B4SOIlckappa = value->rValue;
            mod->B4SOIlckappaGiven = TRUE;
            break;
        case  B4SOI_MOD_LUTE :
            mod->B4SOIlute = value->rValue;
            mod->B4SOIluteGiven = TRUE;
            break;
        case  B4SOI_MOD_LKT1 :
            mod->B4SOIlkt1 = value->rValue;
            mod->B4SOIlkt1Given = TRUE;
            break;
        case  B4SOI_MOD_LKT2 :
            mod->B4SOIlkt2 = value->rValue;
            mod->B4SOIlkt2Given = TRUE;
            break;
        case  B4SOI_MOD_LKT1L :
            mod->B4SOIlkt1l = value->rValue;
            mod->B4SOIlkt1lGiven = TRUE;
            break;
        case  B4SOI_MOD_LUA1 :
            mod->B4SOIlua1 = value->rValue;
            mod->B4SOIlua1Given = TRUE;
            break;
        case  B4SOI_MOD_LUB1 :
            mod->B4SOIlub1 = value->rValue;
            mod->B4SOIlub1Given = TRUE;
            break;
        case  B4SOI_MOD_LUC1 :
            mod->B4SOIluc1 = value->rValue;
            mod->B4SOIluc1Given = TRUE;
            break;
        case  B4SOI_MOD_LAT :
            mod->B4SOIlat = value->rValue;
            mod->B4SOIlatGiven = TRUE;
            break;
        case  B4SOI_MOD_LPRT :
            mod->B4SOIlprt = value->rValue;
            mod->B4SOIlprtGiven = TRUE;
            break;


/* v3.0 */
        case  B4SOI_MOD_LAIGC :
            mod->B4SOIlaigc = value->rValue;
            mod->B4SOIlaigcGiven = TRUE;
            break;
        case  B4SOI_MOD_LBIGC :
            mod->B4SOIlbigc = value->rValue;
            mod->B4SOIlbigcGiven = TRUE;
            break;
        case  B4SOI_MOD_LCIGC :
            mod->B4SOIlcigc = value->rValue;
            mod->B4SOIlcigcGiven = TRUE;
            break;
        case  B4SOI_MOD_LAIGSD :
            mod->B4SOIlaigsd = value->rValue;
            mod->B4SOIlaigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_LBIGSD :
            mod->B4SOIlbigsd = value->rValue;
            mod->B4SOIlbigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_LCIGSD :
            mod->B4SOIlcigsd = value->rValue;
            mod->B4SOIlcigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_LNIGC :
            mod->B4SOIlnigc = value->rValue;
            mod->B4SOIlnigcGiven = TRUE;
            break;
        case  B4SOI_MOD_LPIGCD :
            mod->B4SOIlpigcd = value->rValue;
            mod->B4SOIlpigcdGiven = TRUE;
            break;
        case  B4SOI_MOD_LPOXEDGE :
            mod->B4SOIlpoxedge = value->rValue;
            mod->B4SOIlpoxedgeGiven = TRUE;
            break;

        case B4SOI_MOD_LNPEAK:
            mod->B4SOIlnpeak = value->rValue;
            mod->B4SOIlnpeakGiven = TRUE;
            break;
        case B4SOI_MOD_LNSUB:
            mod->B4SOIlnsub = value->rValue;
            mod->B4SOIlnsubGiven = TRUE;
            break;
        case B4SOI_MOD_LNGATE:
            mod->B4SOIlngate = value->rValue;
            mod->B4SOIlngateGiven = TRUE;
            break;
        case B4SOI_MOD_LNSD:
            mod->B4SOIlnsd = value->rValue;
            mod->B4SOIlnsdGiven = TRUE;
            break;
        case B4SOI_MOD_LVTH0:
            mod->B4SOIlvth0 = value->rValue;
            mod->B4SOIlvth0Given = TRUE;
            break;
        case B4SOI_MOD_LVFB:
            mod->B4SOIlvfb = value->rValue;   
            mod->B4SOIlvfbGiven = TRUE;   
            break; /* v4.1 */ 
        case  B4SOI_MOD_LK1:
            mod->B4SOIlk1 = value->rValue;
            mod->B4SOIlk1Given = TRUE;
            break;
        case  B4SOI_MOD_LK1W1:
            mod->B4SOIlk1w1 = value->rValue;
            mod->B4SOIlk1w1Given = TRUE;
            break;
        case  B4SOI_MOD_LK1W2:
            mod->B4SOIlk1w2 = value->rValue;
            mod->B4SOIlk1w2Given = TRUE;
            break;
        case  B4SOI_MOD_LK2:
            mod->B4SOIlk2 = value->rValue;
            mod->B4SOIlk2Given = TRUE;
            break;
        case  B4SOI_MOD_LK3:
            mod->B4SOIlk3 = value->rValue;
            mod->B4SOIlk3Given = TRUE;
            break;
        case  B4SOI_MOD_LK3B:
            mod->B4SOIlk3b = value->rValue;
            mod->B4SOIlk3bGiven = TRUE;
            break;
        case  B4SOI_MOD_LKB1 :
            mod->B4SOIlkb1 = value->rValue;
            mod->B4SOIlkb1Given = TRUE;
            break;
        case  B4SOI_MOD_LW0:
            mod->B4SOIlw0 = value->rValue;
            mod->B4SOIlw0Given = TRUE;
            break;
        case  B4SOI_MOD_LLPE0:
            mod->B4SOIllpe0 = value->rValue;
            mod->B4SOIllpe0Given = TRUE;
            break;
        case  B4SOI_MOD_LLPEB:        /* v4.0 for Vth */
            mod->B4SOIllpeb = value->rValue;
            mod->B4SOIllpebGiven = TRUE;
            break;
        case  B4SOI_MOD_LDVT0:               
            mod->B4SOIldvt0 = value->rValue;
            mod->B4SOIldvt0Given = TRUE;
            break;
        case  B4SOI_MOD_LDVT1:             
            mod->B4SOIldvt1 = value->rValue;
            mod->B4SOIldvt1Given = TRUE;
            break;
        case  B4SOI_MOD_LDVT2:             
            mod->B4SOIldvt2 = value->rValue;
            mod->B4SOIldvt2Given = TRUE;
            break;
        case  B4SOI_MOD_LDVT0W:               
            mod->B4SOIldvt0w = value->rValue;
            mod->B4SOIldvt0wGiven = TRUE;
            break;
        case  B4SOI_MOD_LDVT1W:             
            mod->B4SOIldvt1w = value->rValue;
            mod->B4SOIldvt1wGiven = TRUE;
            break;
        case  B4SOI_MOD_LDVT2W:             
            mod->B4SOIldvt2w = value->rValue;
            mod->B4SOIldvt2wGiven = TRUE;
            break;
        case  B4SOI_MOD_LU0 :
            mod->B4SOIlu0 = value->rValue;
            mod->B4SOIlu0Given = TRUE;
            break;
        case B4SOI_MOD_LUA:
            mod->B4SOIlua = value->rValue;
            mod->B4SOIluaGiven = TRUE;
            break;
        case B4SOI_MOD_LUB:
            mod->B4SOIlub = value->rValue;
            mod->B4SOIlubGiven = TRUE;
            break;
        case B4SOI_MOD_LUC:
            mod->B4SOIluc = value->rValue;
            mod->B4SOIlucGiven = TRUE;
            break;
        case B4SOI_MOD_LVSAT:
            mod->B4SOIlvsat = value->rValue;
            mod->B4SOIlvsatGiven = TRUE;
            break;
        case B4SOI_MOD_LA0:
            mod->B4SOIla0 = value->rValue;
            mod->B4SOIla0Given = TRUE;
            break;
        case B4SOI_MOD_LAGS:
            mod->B4SOIlags= value->rValue;
            mod->B4SOIlagsGiven = TRUE;
            break;
        case  B4SOI_MOD_LB0 :
            mod->B4SOIlb0 = value->rValue;
            mod->B4SOIlb0Given = TRUE;
            break;
        case  B4SOI_MOD_LB1 :
            mod->B4SOIlb1 = value->rValue;
            mod->B4SOIlb1Given = TRUE;
            break;
        case B4SOI_MOD_LKETA:
            mod->B4SOIlketa = value->rValue;
            mod->B4SOIlketaGiven = TRUE;
            break;    
        case B4SOI_MOD_LKETAS:
            mod->B4SOIlketas = value->rValue;
            mod->B4SOIlketasGiven = TRUE;
            break;    
        case B4SOI_MOD_LA1:
            mod->B4SOIla1 = value->rValue;
            mod->B4SOIla1Given = TRUE;
            break;
        case B4SOI_MOD_LA2:
            mod->B4SOIla2 = value->rValue;
            mod->B4SOIla2Given = TRUE;
            break;
        case B4SOI_MOD_LRDSW:
            mod->B4SOIlrdsw = value->rValue;
            mod->B4SOIlrdswGiven = TRUE;
            break;                     
        case B4SOI_MOD_LRSW:
            mod->B4SOIlrsw = value->rValue;
            mod->B4SOIlrswGiven = TRUE;
            break;                     
        case B4SOI_MOD_LRDW:
            mod->B4SOIlrdw = value->rValue;
            mod->B4SOIlrdwGiven = TRUE;
            break;                     
        case B4SOI_MOD_LPRWB:
            mod->B4SOIlprwb = value->rValue;
            mod->B4SOIlprwbGiven = TRUE;
            break;                     
        case B4SOI_MOD_LPRWG:
            mod->B4SOIlprwg = value->rValue;
            mod->B4SOIlprwgGiven = TRUE;
            break;                     
        case  B4SOI_MOD_LWR :
            mod->B4SOIlwr = value->rValue;
            mod->B4SOIlwrGiven = TRUE;
            break;
        case  B4SOI_MOD_LNFACTOR :
            mod->B4SOIlnfactor = value->rValue;
            mod->B4SOIlnfactorGiven = TRUE;
            break;
        case  B4SOI_MOD_LDWG :
            mod->B4SOIldwg = value->rValue;
            mod->B4SOIldwgGiven = TRUE;
            break;
        case  B4SOI_MOD_LDWB :
            mod->B4SOIldwb = value->rValue;
            mod->B4SOIldwbGiven = TRUE;
            break;
        case B4SOI_MOD_LVOFF:
            mod->B4SOIlvoff = value->rValue;
            mod->B4SOIlvoffGiven = TRUE;
            break;
        case B4SOI_MOD_LETA0:
            mod->B4SOIleta0 = value->rValue;
            mod->B4SOIleta0Given = TRUE;
            break;                 
        case B4SOI_MOD_LETAB:
            mod->B4SOIletab = value->rValue;
            mod->B4SOIletabGiven = TRUE;
            break;                 
        case  B4SOI_MOD_LDSUB:             
            mod->B4SOIldsub = value->rValue;
            mod->B4SOIldsubGiven = TRUE;
            break;
        case  B4SOI_MOD_LCIT :
            mod->B4SOIlcit = value->rValue;
            mod->B4SOIlcitGiven = TRUE;
            break;
        case  B4SOI_MOD_LCDSC :
            mod->B4SOIlcdsc = value->rValue;
            mod->B4SOIlcdscGiven = TRUE;
            break;
        case  B4SOI_MOD_LCDSCB :
            mod->B4SOIlcdscb = value->rValue;
            mod->B4SOIlcdscbGiven = TRUE;
            break;
        case  B4SOI_MOD_LCDSCD :
            mod->B4SOIlcdscd = value->rValue;
            mod->B4SOIlcdscdGiven = TRUE;
            break;
        case B4SOI_MOD_LPCLM:
            mod->B4SOIlpclm = value->rValue;
            mod->B4SOIlpclmGiven = TRUE;
            break;                 
        case B4SOI_MOD_LPDIBL1:
            mod->B4SOIlpdibl1 = value->rValue;
            mod->B4SOIlpdibl1Given = TRUE;
            break;                 
        case B4SOI_MOD_LPDIBL2:
            mod->B4SOIlpdibl2 = value->rValue;
            mod->B4SOIlpdibl2Given = TRUE;
            break;                 
        case B4SOI_MOD_LPDIBLB:
            mod->B4SOIlpdiblb = value->rValue;
            mod->B4SOIlpdiblbGiven = TRUE;
            break;                 
        case  B4SOI_MOD_LDROUT:             
            mod->B4SOIldrout = value->rValue;
            mod->B4SOIldroutGiven = TRUE;
            break;
        case B4SOI_MOD_LPVAG:
            mod->B4SOIlpvag = value->rValue;
            mod->B4SOIlpvagGiven = TRUE;
            break;                 
        case  B4SOI_MOD_LDELTA :
            mod->B4SOIldelta = value->rValue;
            mod->B4SOIldeltaGiven = TRUE;
            break;
        case  B4SOI_MOD_LALPHA0 :
            mod->B4SOIlalpha0 = value->rValue;
            mod->B4SOIlalpha0Given = TRUE;
            break;
        case  B4SOI_MOD_LFBJTII :
            mod->B4SOIlfbjtii = value->rValue;
            mod->B4SOIlfbjtiiGiven = TRUE;
            break;
        /*4.1 Iii model*/
        case  B4SOI_MOD_LEBJTII :
            mod->B4SOIlebjtii = value->rValue;
            mod->B4SOIlebjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_LCBJTII :
            mod->B4SOIlcbjtii = value->rValue;
            mod->B4SOIlcbjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_LVBCI :
            mod->B4SOIlvbci = value->rValue;
            mod->B4SOIlvbciGiven = TRUE;
            break;
        case  B4SOI_MOD_LABJTII :
            mod->B4SOIlabjtii = value->rValue;
            mod->B4SOIlabjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_LMBJTII :
            mod->B4SOIlmbjtii = value->rValue;
            mod->B4SOIlmbjtiiGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_LBETA0 :
            mod->B4SOIlbeta0 = value->rValue;
            mod->B4SOIlbeta0Given = TRUE;
            break;
        case  B4SOI_MOD_LBETA1 :
            mod->B4SOIlbeta1 = value->rValue;
            mod->B4SOIlbeta1Given = TRUE;
            break;
        case  B4SOI_MOD_LBETA2 :
            mod->B4SOIlbeta2 = value->rValue;
            mod->B4SOIlbeta2Given = TRUE;
            break;
        case  B4SOI_MOD_LVDSATII0 :
            mod->B4SOIlvdsatii0 = value->rValue;
            mod->B4SOIlvdsatii0Given = TRUE;
            break;
        case  B4SOI_MOD_LLII :
            mod->B4SOIllii = value->rValue;
            mod->B4SOIlliiGiven = TRUE;
            break;
        case  B4SOI_MOD_LESATII :
            mod->B4SOIlesatii = value->rValue;
            mod->B4SOIlesatiiGiven = TRUE;
            break;
        case  B4SOI_MOD_LSII0 :
            mod->B4SOIlsii0 = value->rValue;
            mod->B4SOIlsii0Given = TRUE;
            break;
        case  B4SOI_MOD_LSII1 :
            mod->B4SOIlsii1 = value->rValue;
            mod->B4SOIlsii1Given = TRUE;
            break;
        case  B4SOI_MOD_LSII2 :
            mod->B4SOIlsii2 = value->rValue;
            mod->B4SOIlsii2Given = TRUE;
            break;
        case  B4SOI_MOD_LSIID :
            mod->B4SOIlsiid = value->rValue;
            mod->B4SOIlsiidGiven = TRUE;
            break;
        case  B4SOI_MOD_LAGIDL :
            mod->B4SOIlagidl = value->rValue;
            mod->B4SOIlagidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LBGIDL :
            mod->B4SOIlbgidl = value->rValue;
            mod->B4SOIlbgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LCGIDL :
            mod->B4SOIlcgidl = value->rValue;
            mod->B4SOIlcgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LEGIDL :
            mod->B4SOIlegidl = value->rValue;
            mod->B4SOIlegidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LRGIDL :
            mod->B4SOIlrgidl = value->rValue;
            mod->B4SOIlrgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LKGIDL :
            mod->B4SOIlkgidl = value->rValue;
            mod->B4SOIlkgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LFGIDL :
            mod->B4SOIlfgidl = value->rValue;
            mod->B4SOIlfgidlGiven = TRUE;
            break;
                        
                        case  B4SOI_MOD_LAGISL :
            mod->B4SOIlagisl = value->rValue;
            mod->B4SOIlagislGiven = TRUE;
            break;
        case  B4SOI_MOD_LBGISL :
            mod->B4SOIlbgisl = value->rValue;
            mod->B4SOIlbgislGiven = TRUE;
            break;
        case  B4SOI_MOD_LCGISL :
            mod->B4SOIlcgisl = value->rValue;
            mod->B4SOIlcgislGiven = TRUE;
            break;
        case  B4SOI_MOD_LEGISL :
            mod->B4SOIlegisl = value->rValue;
            mod->B4SOIlegislGiven = TRUE;
            break;
        case  B4SOI_MOD_LRGISL :
            mod->B4SOIlrgisl = value->rValue;
            mod->B4SOIlrgislGiven = TRUE;
            break;
        case  B4SOI_MOD_LKGISL :
            mod->B4SOIlkgisl = value->rValue;
            mod->B4SOIlkgislGiven = TRUE;
            break;
        case  B4SOI_MOD_LFGISL :
            mod->B4SOIlfgisl = value->rValue;
            mod->B4SOIlfgislGiven = TRUE;
            break;
        case  B4SOI_MOD_LNTUNS :        /* v4.0 */
            mod->B4SOIlntun = value->rValue;
            mod->B4SOIlntunGiven = TRUE;
            break;
        case  B4SOI_MOD_LNTUND :        /* v4.0 */
            mod->B4SOIlntund = value->rValue;
            mod->B4SOIlntundGiven = TRUE;
            break;
        case  B4SOI_MOD_LNDIODES :        /* v4.0 */
            mod->B4SOIlndiode = value->rValue;
            mod->B4SOIlndiodeGiven = TRUE;
            break;
        case  B4SOI_MOD_LNDIODED :        /* v4.0 */
            mod->B4SOIlndioded = value->rValue;
            mod->B4SOIlndiodedGiven = TRUE;
            break;
        case  B4SOI_MOD_LNRECF0S :        /* v4.0 */
            mod->B4SOIlnrecf0 = value->rValue;
            mod->B4SOIlnrecf0Given = TRUE;
            break;
        case  B4SOI_MOD_LNRECF0D :        /* v4.0 */
            mod->B4SOIlnrecf0d = value->rValue;
            mod->B4SOIlnrecf0dGiven = TRUE;
            break;
        case  B4SOI_MOD_LNRECR0S :        /* v4.0 */
            mod->B4SOIlnrecr0 = value->rValue;
            mod->B4SOIlnrecr0Given = TRUE;
            break;
        case  B4SOI_MOD_LNRECR0D :        /* v4.0 */
            mod->B4SOIlnrecr0d = value->rValue;
            mod->B4SOIlnrecr0dGiven = TRUE;
            break;
        case  B4SOI_MOD_LISBJT :
            mod->B4SOIlisbjt = value->rValue;
            mod->B4SOIlisbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_LIDBJT :        /* v4.0 */
            mod->B4SOIlidbjt = value->rValue;
            mod->B4SOIlidbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_LISDIF :
            mod->B4SOIlisdif = value->rValue;
            mod->B4SOIlisdifGiven = TRUE;
            break;
        case  B4SOI_MOD_LIDDIF :        /* v4.0 */
            mod->B4SOIliddif = value->rValue;
            mod->B4SOIliddifGiven = TRUE;
            break;
        case  B4SOI_MOD_LISREC :
            mod->B4SOIlisrec = value->rValue;
            mod->B4SOIlisrecGiven = TRUE;
            break;
        case  B4SOI_MOD_LIDREC :        /* v4.0 */
            mod->B4SOIlidrec = value->rValue;
            mod->B4SOIlidrecGiven = TRUE;
            break;
        case  B4SOI_MOD_LISTUN :
            mod->B4SOIlistun = value->rValue;
            mod->B4SOIlistunGiven = TRUE;
            break;
        case  B4SOI_MOD_LIDTUN :        /* v4.0 */
            mod->B4SOIlidtun = value->rValue;
            mod->B4SOIlidtunGiven = TRUE;
            break;
        case  B4SOI_MOD_LVREC0S :        /* v4.0 */
            mod->B4SOIlvrec0 = value->rValue;
            mod->B4SOIlvrec0Given = TRUE;
            break;
        case  B4SOI_MOD_LVREC0D :        /* v4.0 */
            mod->B4SOIlvrec0d = value->rValue;
            mod->B4SOIlvrec0dGiven = TRUE;
            break;
        case  B4SOI_MOD_LVTUN0S :        /* v4.0 */
            mod->B4SOIlvtun0 = value->rValue;
            mod->B4SOIlvtun0Given = TRUE;
            break;
        case  B4SOI_MOD_LVTUN0D :        /* v4.0 */
            mod->B4SOIlvtun0d = value->rValue;
            mod->B4SOIlvtun0dGiven = TRUE;
            break;
        case  B4SOI_MOD_LNBJT :
            mod->B4SOIlnbjt = value->rValue;
            mod->B4SOIlnbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_LLBJT0 :
            mod->B4SOIllbjt0 = value->rValue;
            mod->B4SOIllbjt0Given = TRUE;
            break;
        case  B4SOI_MOD_LVABJT :
            mod->B4SOIlvabjt = value->rValue;
            mod->B4SOIlvabjtGiven = TRUE;
            break;
        case  B4SOI_MOD_LAELY :
            mod->B4SOIlaely = value->rValue;
            mod->B4SOIlaelyGiven = TRUE;
            break;
        case  B4SOI_MOD_LAHLIS :        /* v4.0 */
            mod->B4SOIlahli = value->rValue;
            mod->B4SOIlahliGiven = TRUE;
            break;
        case  B4SOI_MOD_LAHLID :        /* v4.0 */
            mod->B4SOIlahlid = value->rValue;
            mod->B4SOIlahlidGiven = TRUE;
            break;

/* v3.1 for RF */
        case B4SOI_MOD_LXRCRG1 :
            mod->B4SOIlxrcrg1 = value->rValue;
            mod->B4SOIlxrcrg1Given = TRUE;
            break;
        case B4SOI_MOD_LXRCRG2 :
            mod->B4SOIlxrcrg2 = value->rValue;
            mod->B4SOIlxrcrg2Given = TRUE;
            break;
/* v3.1 for RF end */

        /* CV Model */
        case  B4SOI_MOD_LVSDFB :
            mod->B4SOIlvsdfb = value->rValue;
            mod->B4SOIlvsdfbGiven = TRUE;
            break;
        case  B4SOI_MOD_LVSDTH :
            mod->B4SOIlvsdth = value->rValue;
            mod->B4SOIlvsdthGiven = TRUE;
            break;
        case  B4SOI_MOD_LDELVT :
            mod->B4SOIldelvt = value->rValue;
            mod->B4SOIldelvtGiven = TRUE;
            break;
        case  B4SOI_MOD_LACDE :
            mod->B4SOIlacde = value->rValue;
            mod->B4SOIlacdeGiven = TRUE;
            break;
        case  B4SOI_MOD_LMOIN :
            mod->B4SOIlmoin = value->rValue;
            mod->B4SOIlmoinGiven = TRUE;
            break;
        case  B4SOI_MOD_LNOFF :
            mod->B4SOIlnoff = value->rValue;
            mod->B4SOIlnoffGiven = TRUE;
            break; /* v3.2 */

        /* Width Dependence */
/* v3.1 */
        case  B4SOI_MOD_WXJ :
            mod->B4SOIwxj = value->rValue;
            mod->B4SOIwxjGiven = TRUE;
            break;
        case  B4SOI_MOD_WALPHAGB1 :
            mod->B4SOIwalphaGB1 = value->rValue;
            mod->B4SOIwalphaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_WALPHAGB2 :
            mod->B4SOIwalphaGB2 = value->rValue;
            mod->B4SOIwalphaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_WBETAGB1 :
            mod->B4SOIwbetaGB1 = value->rValue;
            mod->B4SOIwbetaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_WBETAGB2 :
            mod->B4SOIwbetaGB2 = value->rValue;
            mod->B4SOIwbetaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_WAIGBCP2 :
            mod->B4SOIwaigbcp2 = value->rValue;
            mod->B4SOIwaigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_WBIGBCP2 :
            mod->B4SOIwbigbcp2 = value->rValue;
            mod->B4SOIwbigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_WCIGBCP2 :
            mod->B4SOIwcigbcp2 = value->rValue;
            mod->B4SOIwcigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_WNDIF :
            mod->B4SOIwndif = value->rValue;
            mod->B4SOIwndifGiven = TRUE;
            break;
        case  B4SOI_MOD_WNTRECF :
            mod->B4SOIwntrecf = value->rValue;
            mod->B4SOIwntrecfGiven = TRUE;
            break;
        case  B4SOI_MOD_WNTRECR :
            mod->B4SOIwntrecr = value->rValue;
            mod->B4SOIwntrecrGiven = TRUE;
            break;
        case  B4SOI_MOD_WXBJT :
            mod->B4SOIwxbjt = value->rValue;
            mod->B4SOIwxbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_WXDIFS :
            mod->B4SOIwxdif = value->rValue;
            mod->B4SOIwxdifGiven = TRUE;
            break;
        case  B4SOI_MOD_WXRECS :
            mod->B4SOIwxrec = value->rValue;
            mod->B4SOIwxrecGiven = TRUE;
            break;
        case  B4SOI_MOD_WXTUNS :
            mod->B4SOIwxtun = value->rValue;
            mod->B4SOIwxtunGiven = TRUE;
            break;
        case  B4SOI_MOD_WXDIFD :
            mod->B4SOIwxdifd = value->rValue;
            mod->B4SOIwxdifdGiven = TRUE;
            break;
        case  B4SOI_MOD_WXRECD :
            mod->B4SOIwxrecd = value->rValue;
            mod->B4SOIwxrecdGiven = TRUE;
            break;
        case  B4SOI_MOD_WXTUND :
            mod->B4SOIwxtund = value->rValue;
            mod->B4SOIwxtundGiven = TRUE;
            break;
        case  B4SOI_MOD_WCGDL :
            mod->B4SOIwcgdl = value->rValue;
            mod->B4SOIwcgdlGiven = TRUE;
            break;
        case  B4SOI_MOD_WCGSL :
            mod->B4SOIwcgsl = value->rValue;
            mod->B4SOIwcgslGiven = TRUE;
            break;
        case  B4SOI_MOD_WCKAPPA :
            mod->B4SOIwckappa = value->rValue;
            mod->B4SOIwckappaGiven = TRUE;
            break;
        case  B4SOI_MOD_WUTE :
            mod->B4SOIwute = value->rValue;
            mod->B4SOIwuteGiven = TRUE;
            break;
        case  B4SOI_MOD_WKT1 :
            mod->B4SOIwkt1 = value->rValue;
            mod->B4SOIwkt1Given = TRUE;
            break;
        case  B4SOI_MOD_WKT2 :
            mod->B4SOIwkt2 = value->rValue;
            mod->B4SOIwkt2Given = TRUE;
            break;
        case  B4SOI_MOD_WKT1L :
            mod->B4SOIwkt1l = value->rValue;
            mod->B4SOIwkt1lGiven = TRUE;
            break;
        case  B4SOI_MOD_WUA1 :
            mod->B4SOIwua1 = value->rValue;
            mod->B4SOIwua1Given = TRUE;
            break;
        case  B4SOI_MOD_WUB1 :
            mod->B4SOIwub1 = value->rValue;
            mod->B4SOIwub1Given = TRUE;
            break;
        case  B4SOI_MOD_WUC1 :
            mod->B4SOIwuc1 = value->rValue;
            mod->B4SOIwuc1Given = TRUE;
            break;
        case  B4SOI_MOD_WAT :
            mod->B4SOIwat = value->rValue;
            mod->B4SOIwatGiven = TRUE;
            break;
        case  B4SOI_MOD_WPRT :
            mod->B4SOIwprt = value->rValue;
            mod->B4SOIwprtGiven = TRUE;
            break;

/* v3.0 */
        case  B4SOI_MOD_WAIGC :
            mod->B4SOIwaigc = value->rValue;
            mod->B4SOIwaigcGiven = TRUE;
            break;
        case  B4SOI_MOD_WBIGC :
            mod->B4SOIwbigc = value->rValue;
            mod->B4SOIwbigcGiven = TRUE;
            break;
        case  B4SOI_MOD_WCIGC :
            mod->B4SOIwcigc = value->rValue;
            mod->B4SOIwcigcGiven = TRUE;
            break;
        case  B4SOI_MOD_WAIGSD :
            mod->B4SOIwaigsd = value->rValue;
            mod->B4SOIwaigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_WBIGSD :
            mod->B4SOIwbigsd = value->rValue;
            mod->B4SOIwbigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_WCIGSD :
            mod->B4SOIwcigsd = value->rValue;
            mod->B4SOIwcigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_WNIGC :
            mod->B4SOIwnigc = value->rValue;
            mod->B4SOIwnigcGiven = TRUE;
            break;
        case  B4SOI_MOD_WPIGCD :
            mod->B4SOIwpigcd = value->rValue;
            mod->B4SOIwpigcdGiven = TRUE;
            break;
        case  B4SOI_MOD_WPOXEDGE :
            mod->B4SOIwpoxedge = value->rValue;
            mod->B4SOIwpoxedgeGiven = TRUE;
            break;

        case B4SOI_MOD_WNPEAK:
            mod->B4SOIwnpeak = value->rValue;
            mod->B4SOIwnpeakGiven = TRUE;
            break;
        case B4SOI_MOD_WNSUB:
            mod->B4SOIwnsub = value->rValue;
            mod->B4SOIwnsubGiven = TRUE;
            break;
        case B4SOI_MOD_WNGATE:
            mod->B4SOIwngate = value->rValue;
            mod->B4SOIwngateGiven = TRUE;
            break;
        case B4SOI_MOD_WNSD:
            mod->B4SOIwnsd = value->rValue;
            mod->B4SOIwnsdGiven = TRUE;
            break;
        case B4SOI_MOD_WVTH0:
            mod->B4SOIwvth0 = value->rValue;
            mod->B4SOIwvth0Given = TRUE;
            break;
        case B4SOI_MOD_WVFB:
            mod->B4SOIwvfb = value->rValue;   
            mod->B4SOIwvfbGiven = TRUE;   
            break; /* v4.1 */ 
        case  B4SOI_MOD_WK1:
            mod->B4SOIwk1 = value->rValue;
            mod->B4SOIwk1Given = TRUE;
            break;
        case  B4SOI_MOD_WK1W1:
            mod->B4SOIwk1w1 = value->rValue;
            mod->B4SOIwk1w1Given = TRUE;
            break;
        case  B4SOI_MOD_WK1W2:
            mod->B4SOIwk1w2 = value->rValue;
            mod->B4SOIwk1w2Given = TRUE;
            break;
        case  B4SOI_MOD_WK2:
            mod->B4SOIwk2 = value->rValue;
            mod->B4SOIwk2Given = TRUE;
            break;
        case  B4SOI_MOD_WK3:
            mod->B4SOIwk3 = value->rValue;
            mod->B4SOIwk3Given = TRUE;
            break;
        case  B4SOI_MOD_WK3B:
            mod->B4SOIwk3b = value->rValue;
            mod->B4SOIwk3bGiven = TRUE;
            break;
        case  B4SOI_MOD_WKB1 :
            mod->B4SOIwkb1 = value->rValue;
            mod->B4SOIwkb1Given = TRUE;
            break;
        case  B4SOI_MOD_WW0:
            mod->B4SOIww0 = value->rValue;
            mod->B4SOIww0Given = TRUE;
            break;
        case  B4SOI_MOD_WLPE0:
            mod->B4SOIwlpe0 = value->rValue;
            mod->B4SOIwlpe0Given = TRUE;
            break;
        case  B4SOI_MOD_WLPEB:        /* v4.0 for Vth */
            mod->B4SOIwlpeb = value->rValue;
            mod->B4SOIwlpebGiven = TRUE;
            break;
        case  B4SOI_MOD_WDVT0:               
            mod->B4SOIwdvt0 = value->rValue;
            mod->B4SOIwdvt0Given = TRUE;
            break;
        case  B4SOI_MOD_WDVT1:             
            mod->B4SOIwdvt1 = value->rValue;
            mod->B4SOIwdvt1Given = TRUE;
            break;
        case  B4SOI_MOD_WDVT2:             
            mod->B4SOIwdvt2 = value->rValue;
            mod->B4SOIwdvt2Given = TRUE;
            break;
        case  B4SOI_MOD_WDVT0W:               
            mod->B4SOIwdvt0w = value->rValue;
            mod->B4SOIwdvt0wGiven = TRUE;
            break;
        case  B4SOI_MOD_WDVT1W:             
            mod->B4SOIwdvt1w = value->rValue;
            mod->B4SOIwdvt1wGiven = TRUE;
            break;
        case  B4SOI_MOD_WDVT2W:             
            mod->B4SOIwdvt2w = value->rValue;
            mod->B4SOIwdvt2wGiven = TRUE;
            break;
        case  B4SOI_MOD_WU0 :
            mod->B4SOIwu0 = value->rValue;
            mod->B4SOIwu0Given = TRUE;
            break;
        case B4SOI_MOD_WUA:
            mod->B4SOIwua = value->rValue;
            mod->B4SOIwuaGiven = TRUE;
            break;
        case B4SOI_MOD_WUB:
            mod->B4SOIwub = value->rValue;
            mod->B4SOIwubGiven = TRUE;
            break;
        case B4SOI_MOD_WUC:
            mod->B4SOIwuc = value->rValue;
            mod->B4SOIwucGiven = TRUE;
            break;
        case B4SOI_MOD_WVSAT:
            mod->B4SOIwvsat = value->rValue;
            mod->B4SOIwvsatGiven = TRUE;
            break;
        case B4SOI_MOD_WA0:
            mod->B4SOIwa0 = value->rValue;
            mod->B4SOIwa0Given = TRUE;
            break;
        case B4SOI_MOD_WAGS:
            mod->B4SOIwags= value->rValue;
            mod->B4SOIwagsGiven = TRUE;
            break;
        case  B4SOI_MOD_WB0 :
            mod->B4SOIwb0 = value->rValue;
            mod->B4SOIwb0Given = TRUE;
            break;
        case  B4SOI_MOD_WB1 :
            mod->B4SOIwb1 = value->rValue;
            mod->B4SOIwb1Given = TRUE;
            break;
        case B4SOI_MOD_WKETA:
            mod->B4SOIwketa = value->rValue;
            mod->B4SOIwketaGiven = TRUE;
            break;    
        case B4SOI_MOD_WKETAS:
            mod->B4SOIwketas = value->rValue;
            mod->B4SOIwketasGiven = TRUE;
            break;    
        case B4SOI_MOD_WA1:
            mod->B4SOIwa1 = value->rValue;
            mod->B4SOIwa1Given = TRUE;
            break;
        case B4SOI_MOD_WA2:
            mod->B4SOIwa2 = value->rValue;
            mod->B4SOIwa2Given = TRUE;
            break;
        case B4SOI_MOD_WRDSW:
            mod->B4SOIwrdsw = value->rValue;
            mod->B4SOIwrdswGiven = TRUE;
            break;                     
        case B4SOI_MOD_WRSW:
            mod->B4SOIwrsw = value->rValue;
            mod->B4SOIwrswGiven = TRUE;
            break;                     
        case B4SOI_MOD_WRDW:
            mod->B4SOIwrdw = value->rValue;
            mod->B4SOIwrdwGiven = TRUE;
            break;                     
        case B4SOI_MOD_WPRWB:
            mod->B4SOIwprwb = value->rValue;
            mod->B4SOIwprwbGiven = TRUE;
            break;                     
        case B4SOI_MOD_WPRWG:
            mod->B4SOIwprwg = value->rValue;
            mod->B4SOIwprwgGiven = TRUE;
            break;                     
        case  B4SOI_MOD_WWR :
            mod->B4SOIwwr = value->rValue;
            mod->B4SOIwwrGiven = TRUE;
            break;
        case  B4SOI_MOD_WNFACTOR :
            mod->B4SOIwnfactor = value->rValue;
            mod->B4SOIwnfactorGiven = TRUE;
            break;
        case  B4SOI_MOD_WDWG :
            mod->B4SOIwdwg = value->rValue;
            mod->B4SOIwdwgGiven = TRUE;
            break;
        case  B4SOI_MOD_WDWB :
            mod->B4SOIwdwb = value->rValue;
            mod->B4SOIwdwbGiven = TRUE;
            break;
        case B4SOI_MOD_WVOFF:
            mod->B4SOIwvoff = value->rValue;
            mod->B4SOIwvoffGiven = TRUE;
            break;
        case B4SOI_MOD_WETA0:
            mod->B4SOIweta0 = value->rValue;
            mod->B4SOIweta0Given = TRUE;
            break;                 
        case B4SOI_MOD_WETAB:
            mod->B4SOIwetab = value->rValue;
            mod->B4SOIwetabGiven = TRUE;
            break;                 
        case  B4SOI_MOD_WDSUB:             
            mod->B4SOIwdsub = value->rValue;
            mod->B4SOIwdsubGiven = TRUE;
            break;
        case  B4SOI_MOD_WCIT :
            mod->B4SOIwcit = value->rValue;
            mod->B4SOIwcitGiven = TRUE;
            break;
        case  B4SOI_MOD_WCDSC :
            mod->B4SOIwcdsc = value->rValue;
            mod->B4SOIwcdscGiven = TRUE;
            break;
        case  B4SOI_MOD_WCDSCB :
            mod->B4SOIwcdscb = value->rValue;
            mod->B4SOIwcdscbGiven = TRUE;
            break;
        case  B4SOI_MOD_WCDSCD :
            mod->B4SOIwcdscd = value->rValue;
            mod->B4SOIwcdscdGiven = TRUE;
            break;
        case B4SOI_MOD_WPCLM:
            mod->B4SOIwpclm = value->rValue;
            mod->B4SOIwpclmGiven = TRUE;
            break;                 
        case B4SOI_MOD_WPDIBL1:
            mod->B4SOIwpdibl1 = value->rValue;
            mod->B4SOIwpdibl1Given = TRUE;
            break;                 
        case B4SOI_MOD_WPDIBL2:
            mod->B4SOIwpdibl2 = value->rValue;
            mod->B4SOIwpdibl2Given = TRUE;
            break;                 
        case B4SOI_MOD_WPDIBLB:
            mod->B4SOIwpdiblb = value->rValue;
            mod->B4SOIwpdiblbGiven = TRUE;
            break;                 
        case  B4SOI_MOD_WDROUT:             
            mod->B4SOIwdrout = value->rValue;
            mod->B4SOIwdroutGiven = TRUE;
            break;
        case B4SOI_MOD_WPVAG:
            mod->B4SOIwpvag = value->rValue;
            mod->B4SOIwpvagGiven = TRUE;
            break;                 
        case  B4SOI_MOD_WDELTA :
            mod->B4SOIwdelta = value->rValue;
            mod->B4SOIwdeltaGiven = TRUE;
            break;
        case  B4SOI_MOD_WALPHA0 :
            mod->B4SOIwalpha0 = value->rValue;
            mod->B4SOIwalpha0Given = TRUE;
            break;
        case  B4SOI_MOD_WFBJTII :
            mod->B4SOIwfbjtii = value->rValue;
            mod->B4SOIwfbjtiiGiven = TRUE;
            break;
                /*4.1 Iii model*/
                case  B4SOI_MOD_WEBJTII :
            mod->B4SOIwebjtii = value->rValue;
            mod->B4SOIwebjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_WCBJTII :
            mod->B4SOIwcbjtii = value->rValue;
            mod->B4SOIwcbjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_WVBCI :
            mod->B4SOIwvbci = value->rValue;
            mod->B4SOIwvbciGiven = TRUE;
            break;
        case  B4SOI_MOD_WABJTII :
            mod->B4SOIwabjtii = value->rValue;
            mod->B4SOIwabjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_WMBJTII :
            mod->B4SOIwmbjtii = value->rValue;
            mod->B4SOIwmbjtiiGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WBETA0 :
            mod->B4SOIwbeta0 = value->rValue;
            mod->B4SOIwbeta0Given = TRUE;
            break;
        case  B4SOI_MOD_WBETA1 :
            mod->B4SOIwbeta1 = value->rValue;
            mod->B4SOIwbeta1Given = TRUE;
            break;
        case  B4SOI_MOD_WBETA2 :
            mod->B4SOIwbeta2 = value->rValue;
            mod->B4SOIwbeta2Given = TRUE;
            break;
        case  B4SOI_MOD_WVDSATII0 :
            mod->B4SOIwvdsatii0 = value->rValue;
            mod->B4SOIwvdsatii0Given = TRUE;
            break;
        case  B4SOI_MOD_WLII :
            mod->B4SOIwlii = value->rValue;
            mod->B4SOIwliiGiven = TRUE;
            break;
        case  B4SOI_MOD_WESATII :
            mod->B4SOIwesatii = value->rValue;
            mod->B4SOIwesatiiGiven = TRUE;
            break;
        case  B4SOI_MOD_WSII0 :
            mod->B4SOIwsii0 = value->rValue;
            mod->B4SOIwsii0Given = TRUE;
            break;
        case  B4SOI_MOD_WSII1 :
            mod->B4SOIwsii1 = value->rValue;
            mod->B4SOIwsii1Given = TRUE;
            break;
        case  B4SOI_MOD_WSII2 :
            mod->B4SOIwsii2 = value->rValue;
            mod->B4SOIwsii2Given = TRUE;
            break;
        case  B4SOI_MOD_WSIID :
            mod->B4SOIwsiid = value->rValue;
            mod->B4SOIwsiidGiven = TRUE;
            break;
        case  B4SOI_MOD_WAGIDL :
            mod->B4SOIwagidl = value->rValue;
            mod->B4SOIwagidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WBGIDL :
            mod->B4SOIwbgidl = value->rValue;
            mod->B4SOIwbgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WCGIDL :
            mod->B4SOIwcgidl = value->rValue;
            mod->B4SOIwcgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WEGIDL :
            mod->B4SOIwegidl = value->rValue;
            mod->B4SOIwegidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WRGIDL :
            mod->B4SOIwrgidl = value->rValue;
            mod->B4SOIwrgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WKGIDL :
            mod->B4SOIwkgidl = value->rValue;
            mod->B4SOIwkgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WFGIDL :
            mod->B4SOIwfgidl = value->rValue;
            mod->B4SOIwfgidlGiven = TRUE;
            break;
                        
                        
        case  B4SOI_MOD_WAGISL :
            mod->B4SOIwagisl = value->rValue;
            mod->B4SOIwagislGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WBGISL :
            mod->B4SOIwbgisl = value->rValue;
            mod->B4SOIwbgislGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WCGISL :
            mod->B4SOIwcgisl = value->rValue;
            mod->B4SOIwcgislGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WEGISL :
            mod->B4SOIwegisl = value->rValue;
            mod->B4SOIwegislGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WRGISL :
            mod->B4SOIwrgisl = value->rValue;
            mod->B4SOIwrgislGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WKGISL :
            mod->B4SOIwkgisl = value->rValue;
            mod->B4SOIwkgislGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_WFGISL :
            mod->B4SOIwfgisl = value->rValue;
            mod->B4SOIwfgislGiven = TRUE;
            break;
                        
                        
        case  B4SOI_MOD_WNTUNS :   /* v4.0 */
            mod->B4SOIwntun = value->rValue;
            mod->B4SOIwntunGiven = TRUE;
            break;
        case  B4SOI_MOD_WNTUND :   /* v4.0 */
            mod->B4SOIwntund = value->rValue;
            mod->B4SOIwntundGiven = TRUE;
            break;
        case  B4SOI_MOD_WNDIODES : /* v4.0 */
            mod->B4SOIwndiode = value->rValue;
            mod->B4SOIwndiodeGiven = TRUE;
            break;
        case  B4SOI_MOD_WNDIODED : /* v4.0 */
            mod->B4SOIwndioded = value->rValue;
            mod->B4SOIwndiodedGiven = TRUE;
            break;
        case  B4SOI_MOD_WNRECF0S :        /* v4.0 */
            mod->B4SOIwnrecf0 = value->rValue;
            mod->B4SOIwnrecf0Given = TRUE;
            break;
        case  B4SOI_MOD_WNRECF0D :        /* v4.0 */
            mod->B4SOIwnrecf0d = value->rValue;
            mod->B4SOIwnrecf0dGiven = TRUE;
            break;
        case  B4SOI_MOD_WNRECR0S :        /* v4.0 */
            mod->B4SOIwnrecr0 = value->rValue;
            mod->B4SOIwnrecr0Given = TRUE;
            break;
        case  B4SOI_MOD_WNRECR0D :        /* v4.0 */
            mod->B4SOIwnrecr0d = value->rValue;
            mod->B4SOIwnrecr0dGiven = TRUE;
            break;
        case  B4SOI_MOD_WISBJT :
            mod->B4SOIwisbjt = value->rValue;
            mod->B4SOIwisbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_WIDBJT :        /* v4.0 */
            mod->B4SOIwidbjt = value->rValue;
            mod->B4SOIwidbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_WISDIF :
            mod->B4SOIwisdif = value->rValue;
            mod->B4SOIwisdifGiven = TRUE;
            break;
        case  B4SOI_MOD_WIDDIF :        /* v4.0 */
            mod->B4SOIwiddif = value->rValue;
            mod->B4SOIwiddifGiven = TRUE;
            break;
        case  B4SOI_MOD_WISREC :
            mod->B4SOIwisrec = value->rValue;
            mod->B4SOIwisrecGiven = TRUE;
            break;
        case  B4SOI_MOD_WIDREC :        /* v4.0 */
            mod->B4SOIwidrec = value->rValue;
            mod->B4SOIwidrecGiven = TRUE;
            break;
        case  B4SOI_MOD_WISTUN :
            mod->B4SOIwistun = value->rValue;
            mod->B4SOIwistunGiven = TRUE;
            break;
        case  B4SOI_MOD_WIDTUN :        /* v4.0 */
            mod->B4SOIwidtun = value->rValue;
            mod->B4SOIwidtunGiven = TRUE;
            break;
        case  B4SOI_MOD_WVREC0S :        /* v4.0 */
            mod->B4SOIwvrec0 = value->rValue;
            mod->B4SOIwvrec0Given = TRUE;
            break;
        case  B4SOI_MOD_WVREC0D :        /* v4.0 */
            mod->B4SOIwvrec0d = value->rValue;
            mod->B4SOIwvrec0dGiven = TRUE;
            break;
        case  B4SOI_MOD_WVTUN0S :        /* v4.0 */
            mod->B4SOIwvtun0 = value->rValue;
            mod->B4SOIwvtun0Given = TRUE;
            break;
        case  B4SOI_MOD_WVTUN0D :        /* v4.0 */
            mod->B4SOIwvtun0d = value->rValue;
            mod->B4SOIwvtun0dGiven = TRUE;
            break;
        case  B4SOI_MOD_WNBJT :
            mod->B4SOIwnbjt = value->rValue;
            mod->B4SOIwnbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_WLBJT0 :
            mod->B4SOIwlbjt0 = value->rValue;
            mod->B4SOIwlbjt0Given = TRUE;
            break;
        case  B4SOI_MOD_WVABJT :
            mod->B4SOIwvabjt = value->rValue;
            mod->B4SOIwvabjtGiven = TRUE;
            break;
        case  B4SOI_MOD_WAELY :
            mod->B4SOIwaely = value->rValue;
            mod->B4SOIwaelyGiven = TRUE;
            break;
        case  B4SOI_MOD_WAHLIS :        /* v4.0 */
            mod->B4SOIwahli = value->rValue;
            mod->B4SOIwahliGiven = TRUE;
            break;
        case  B4SOI_MOD_WAHLID :        /* v4.0 */
            mod->B4SOIwahlid = value->rValue;
            mod->B4SOIwahlidGiven = TRUE;
            break;

/* v3.1 for RF */
        case B4SOI_MOD_WXRCRG1 :
            mod->B4SOIwxrcrg1 = value->rValue;
            mod->B4SOIwxrcrg1Given = TRUE;
            break;
        case B4SOI_MOD_WXRCRG2 :
            mod->B4SOIwxrcrg2 = value->rValue;
            mod->B4SOIwxrcrg2Given = TRUE;
            break;
/* v3.1 for RF end */

        /* CV Model */
        case  B4SOI_MOD_WVSDFB :
            mod->B4SOIwvsdfb = value->rValue;
            mod->B4SOIwvsdfbGiven = TRUE;
            break;
        case  B4SOI_MOD_WVSDTH :
            mod->B4SOIwvsdth = value->rValue;
            mod->B4SOIwvsdthGiven = TRUE;
            break;
        case  B4SOI_MOD_WDELVT :
            mod->B4SOIwdelvt = value->rValue;
            mod->B4SOIwdelvtGiven = TRUE;
            break;
        case  B4SOI_MOD_WACDE :
            mod->B4SOIwacde = value->rValue;
            mod->B4SOIwacdeGiven = TRUE;
            break;
        case  B4SOI_MOD_WMOIN :
            mod->B4SOIwmoin = value->rValue;
            mod->B4SOIwmoinGiven = TRUE;
            break;
        case  B4SOI_MOD_WNOFF :
            mod->B4SOIwnoff = value->rValue;
            mod->B4SOIwnoffGiven = TRUE;
            break; /* v3.2 */

        /* Cross-term Dependence */
/* v3.1 */
        case  B4SOI_MOD_PXJ :
            mod->B4SOIpxj = value->rValue;
            mod->B4SOIpxjGiven = TRUE;
            break;
        case  B4SOI_MOD_PALPHAGB1 :
            mod->B4SOIpalphaGB1 = value->rValue;
            mod->B4SOIpalphaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_PALPHAGB2 :
            mod->B4SOIpalphaGB2 = value->rValue;
            mod->B4SOIpalphaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_PBETAGB1 :
            mod->B4SOIpbetaGB1 = value->rValue;
            mod->B4SOIpbetaGB1Given = TRUE;
            break;
        case  B4SOI_MOD_PBETAGB2 :
            mod->B4SOIpbetaGB2 = value->rValue;
            mod->B4SOIpbetaGB2Given = TRUE;
            break;
        case  B4SOI_MOD_PAIGBCP2 :
            mod->B4SOIpaigbcp2 = value->rValue;
            mod->B4SOIpaigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_PBIGBCP2 :
            mod->B4SOIpbigbcp2 = value->rValue;
            mod->B4SOIpbigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_PCIGBCP2 :
            mod->B4SOIpcigbcp2 = value->rValue;
            mod->B4SOIpcigbcp2Given = TRUE;
            break;
        case  B4SOI_MOD_PNDIF :
            mod->B4SOIpndif = value->rValue;
            mod->B4SOIpndifGiven = TRUE;
            break;
        case  B4SOI_MOD_PNTRECF :
            mod->B4SOIpntrecf = value->rValue;
            mod->B4SOIpntrecfGiven = TRUE;
            break;
        case  B4SOI_MOD_PNTRECR :
            mod->B4SOIpntrecr = value->rValue;
            mod->B4SOIpntrecrGiven = TRUE;
            break;
        case  B4SOI_MOD_PXBJT :
            mod->B4SOIpxbjt = value->rValue;
            mod->B4SOIpxbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_PXDIFS :
            mod->B4SOIpxdif = value->rValue;
            mod->B4SOIpxdifGiven = TRUE;
            break;
        case  B4SOI_MOD_PXRECS :
            mod->B4SOIpxrec = value->rValue;
            mod->B4SOIpxrecGiven = TRUE;
            break;
        case  B4SOI_MOD_PXTUNS :
            mod->B4SOIpxtun = value->rValue;
            mod->B4SOIpxtunGiven = TRUE;
            break;
        case  B4SOI_MOD_PXDIFD :
            mod->B4SOIpxdifd = value->rValue;
            mod->B4SOIpxdifdGiven = TRUE;
            break;
        case  B4SOI_MOD_PXRECD :
            mod->B4SOIpxrecd = value->rValue;
            mod->B4SOIpxrecdGiven = TRUE;
            break;
        case  B4SOI_MOD_PXTUND :
            mod->B4SOIpxtund = value->rValue;
            mod->B4SOIpxtundGiven = TRUE;
            break;
        case  B4SOI_MOD_PCGDL :
            mod->B4SOIpcgdl = value->rValue;
            mod->B4SOIpcgdlGiven = TRUE;
            break;
        case  B4SOI_MOD_PCGSL :
            mod->B4SOIpcgsl = value->rValue;
            mod->B4SOIpcgslGiven = TRUE;
            break;
        case  B4SOI_MOD_PCKAPPA :
            mod->B4SOIpckappa = value->rValue;
            mod->B4SOIpckappaGiven = TRUE;
            break;
        case  B4SOI_MOD_PUTE :
            mod->B4SOIpute = value->rValue;
            mod->B4SOIputeGiven = TRUE;
            break;
        case  B4SOI_MOD_PKT1 :
            mod->B4SOIpkt1 = value->rValue;
            mod->B4SOIpkt1Given = TRUE;
            break;
        case  B4SOI_MOD_PKT2 :
            mod->B4SOIpkt2 = value->rValue;
            mod->B4SOIpkt2Given = TRUE;
            break;
        case  B4SOI_MOD_PKT1L :
            mod->B4SOIpkt1l = value->rValue;
            mod->B4SOIpkt1lGiven = TRUE;
            break;
        case  B4SOI_MOD_PUA1 :
            mod->B4SOIpua1 = value->rValue;
            mod->B4SOIpua1Given = TRUE;
            break;
        case  B4SOI_MOD_PUB1 :
            mod->B4SOIpub1 = value->rValue;
            mod->B4SOIpub1Given = TRUE;
            break;
        case  B4SOI_MOD_PUC1 :
            mod->B4SOIpuc1 = value->rValue;
            mod->B4SOIpuc1Given = TRUE;
            break;
        case  B4SOI_MOD_PAT :
            mod->B4SOIpat = value->rValue;
            mod->B4SOIpatGiven = TRUE;
            break;
        case  B4SOI_MOD_PPRT :
            mod->B4SOIpprt = value->rValue;
            mod->B4SOIpprtGiven = TRUE;
            break;

/* v3.0 */
        case  B4SOI_MOD_PAIGC :
            mod->B4SOIpaigc = value->rValue;
            mod->B4SOIpaigcGiven = TRUE;
            break;
        case  B4SOI_MOD_PBIGC :
            mod->B4SOIpbigc = value->rValue;
            mod->B4SOIpbigcGiven = TRUE;
            break;
        case  B4SOI_MOD_PCIGC :
            mod->B4SOIpcigc = value->rValue;
            mod->B4SOIpcigcGiven = TRUE;
            break;
        case  B4SOI_MOD_PAIGSD :
            mod->B4SOIpaigsd = value->rValue;
            mod->B4SOIpaigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_PBIGSD :
            mod->B4SOIpbigsd = value->rValue;
            mod->B4SOIpbigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_PCIGSD :
            mod->B4SOIpcigsd = value->rValue;
            mod->B4SOIpcigsdGiven = TRUE;
            break;
        case  B4SOI_MOD_PNIGC :
            mod->B4SOIpnigc = value->rValue;
            mod->B4SOIpnigcGiven = TRUE;
            break;
        case  B4SOI_MOD_PPIGCD :
            mod->B4SOIppigcd = value->rValue;
            mod->B4SOIppigcdGiven = TRUE;
            break;
        case  B4SOI_MOD_PPOXEDGE :
            mod->B4SOIppoxedge = value->rValue;
            mod->B4SOIppoxedgeGiven = TRUE;
            break;

        case B4SOI_MOD_PNPEAK:
            mod->B4SOIpnpeak = value->rValue;
            mod->B4SOIpnpeakGiven = TRUE;
            break;
        case B4SOI_MOD_PNSUB:
            mod->B4SOIpnsub = value->rValue;
            mod->B4SOIpnsubGiven = TRUE;
            break;
        case B4SOI_MOD_PNGATE:
            mod->B4SOIpngate = value->rValue;
            mod->B4SOIpngateGiven = TRUE;
            break;
        case B4SOI_MOD_PNSD:
            mod->B4SOIpnsd = value->rValue;
            mod->B4SOIpnsdGiven = TRUE;
            break;
        case B4SOI_MOD_PVTH0:
            mod->B4SOIpvth0 = value->rValue;
            mod->B4SOIpvth0Given = TRUE;
            break;
        case B4SOI_MOD_PVFB:
            mod->B4SOIpvfb = value->rValue;   
            mod->B4SOIpvfbGiven = TRUE;   
            break; /* v4.1 */ 
        case  B4SOI_MOD_PK1:
            mod->B4SOIpk1 = value->rValue;
            mod->B4SOIpk1Given = TRUE;
            break;
        case  B4SOI_MOD_PK1W1:
            mod->B4SOIpk1w1 = value->rValue;
            mod->B4SOIpk1w1Given = TRUE;
            break;
        case  B4SOI_MOD_PK1W2:
            mod->B4SOIpk1w2 = value->rValue;
            mod->B4SOIpk1w2Given = TRUE;
            break;
        case  B4SOI_MOD_PK2:
            mod->B4SOIpk2 = value->rValue;
            mod->B4SOIpk2Given = TRUE;
            break;
        case  B4SOI_MOD_PK3:
            mod->B4SOIpk3 = value->rValue;
            mod->B4SOIpk3Given = TRUE;
            break;
        case  B4SOI_MOD_PK3B:
            mod->B4SOIpk3b = value->rValue;
            mod->B4SOIpk3bGiven = TRUE;
            break;
        case  B4SOI_MOD_PKB1 :
            mod->B4SOIpkb1 = value->rValue;
            mod->B4SOIpkb1Given = TRUE;
            break;
        case  B4SOI_MOD_PW0:
            mod->B4SOIpw0 = value->rValue;
            mod->B4SOIpw0Given = TRUE;
            break;
        case  B4SOI_MOD_PLPE0:
            mod->B4SOIplpe0 = value->rValue;
            mod->B4SOIplpe0Given = TRUE;
            break;
        case  B4SOI_MOD_PLPEB:        /* v4.0 for Vth */
            mod->B4SOIplpeb = value->rValue;
            mod->B4SOIplpebGiven = TRUE;
            break;
        case  B4SOI_MOD_PDVT0:               
            mod->B4SOIpdvt0 = value->rValue;
            mod->B4SOIpdvt0Given = TRUE;
            break;
        case  B4SOI_MOD_PDVT1:             
            mod->B4SOIpdvt1 = value->rValue;
            mod->B4SOIpdvt1Given = TRUE;
            break;
        case  B4SOI_MOD_PDVT2:             
            mod->B4SOIpdvt2 = value->rValue;
            mod->B4SOIpdvt2Given = TRUE;
            break;
        case  B4SOI_MOD_PDVT0W:               
            mod->B4SOIpdvt0w = value->rValue;
            mod->B4SOIpdvt0wGiven = TRUE;
            break;
        case  B4SOI_MOD_PDVT1W:             
            mod->B4SOIpdvt1w = value->rValue;
            mod->B4SOIpdvt1wGiven = TRUE;
            break;
        case  B4SOI_MOD_PDVT2W:             
            mod->B4SOIpdvt2w = value->rValue;
            mod->B4SOIpdvt2wGiven = TRUE;
            break;
        case  B4SOI_MOD_PU0 :
            mod->B4SOIpu0 = value->rValue;
            mod->B4SOIpu0Given = TRUE;
            break;
        case B4SOI_MOD_PUA:
            mod->B4SOIpua = value->rValue;
            mod->B4SOIpuaGiven = TRUE;
            break;
        case B4SOI_MOD_PUB:
            mod->B4SOIpub = value->rValue;
            mod->B4SOIpubGiven = TRUE;
            break;
        case B4SOI_MOD_PUC:
            mod->B4SOIpuc = value->rValue;
            mod->B4SOIpucGiven = TRUE;
            break;
        case B4SOI_MOD_PVSAT:
            mod->B4SOIpvsat = value->rValue;
            mod->B4SOIpvsatGiven = TRUE;
            break;
        case B4SOI_MOD_PA0:
            mod->B4SOIpa0 = value->rValue;
            mod->B4SOIpa0Given = TRUE;
            break;
        case B4SOI_MOD_PAGS:
            mod->B4SOIpags= value->rValue;
            mod->B4SOIpagsGiven = TRUE;
            break;
        case  B4SOI_MOD_PB0 :
            mod->B4SOIpb0 = value->rValue;
            mod->B4SOIpb0Given = TRUE;
            break;
        case  B4SOI_MOD_PB1 :
            mod->B4SOIpb1 = value->rValue;
            mod->B4SOIpb1Given = TRUE;
            break;
        case B4SOI_MOD_PKETA:
            mod->B4SOIpketa = value->rValue;
            mod->B4SOIpketaGiven = TRUE;
            break;    
        case B4SOI_MOD_PKETAS:
            mod->B4SOIpketas = value->rValue;
            mod->B4SOIpketasGiven = TRUE;
            break;    
        case B4SOI_MOD_PA1:
            mod->B4SOIpa1 = value->rValue;
            mod->B4SOIpa1Given = TRUE;
            break;
        case B4SOI_MOD_PA2:
            mod->B4SOIpa2 = value->rValue;
            mod->B4SOIpa2Given = TRUE;
            break;
        case B4SOI_MOD_PRDSW:
            mod->B4SOIprdsw = value->rValue;
            mod->B4SOIprdswGiven = TRUE;
            break;                     
        case B4SOI_MOD_PRSW:
            mod->B4SOIprsw = value->rValue;
            mod->B4SOIprswGiven = TRUE;
            break;                     
        case B4SOI_MOD_PRDW:
            mod->B4SOIprdw = value->rValue;
            mod->B4SOIprdwGiven = TRUE;
            break;                     
        case B4SOI_MOD_PPRWB:
            mod->B4SOIpprwb = value->rValue;
            mod->B4SOIpprwbGiven = TRUE;
            break;                     
        case B4SOI_MOD_PPRWG:
            mod->B4SOIpprwg = value->rValue;
            mod->B4SOIpprwgGiven = TRUE;
            break;                     
        case  B4SOI_MOD_PWR :
            mod->B4SOIpwr = value->rValue;
            mod->B4SOIpwrGiven = TRUE;
            break;
        case  B4SOI_MOD_PNFACTOR :
            mod->B4SOIpnfactor = value->rValue;
            mod->B4SOIpnfactorGiven = TRUE;
            break;
        case  B4SOI_MOD_PDWG :
            mod->B4SOIpdwg = value->rValue;
            mod->B4SOIpdwgGiven = TRUE;
            break;
        case  B4SOI_MOD_PDWB :
            mod->B4SOIpdwb = value->rValue;
            mod->B4SOIpdwbGiven = TRUE;
            break;
        case B4SOI_MOD_PVOFF:
            mod->B4SOIpvoff = value->rValue;
            mod->B4SOIpvoffGiven = TRUE;
            break;
        case B4SOI_MOD_PETA0:
            mod->B4SOIpeta0 = value->rValue;
            mod->B4SOIpeta0Given = TRUE;
            break;                 
        case B4SOI_MOD_PETAB:
            mod->B4SOIpetab = value->rValue;
            mod->B4SOIpetabGiven = TRUE;
            break;                 
        case  B4SOI_MOD_PDSUB:             
            mod->B4SOIpdsub = value->rValue;
            mod->B4SOIpdsubGiven = TRUE;
            break;
        case  B4SOI_MOD_PCIT :
            mod->B4SOIpcit = value->rValue;
            mod->B4SOIpcitGiven = TRUE;
            break;
        case  B4SOI_MOD_PCDSC :
            mod->B4SOIpcdsc = value->rValue;
            mod->B4SOIpcdscGiven = TRUE;
            break;
        case  B4SOI_MOD_PCDSCB :
            mod->B4SOIpcdscb = value->rValue;
            mod->B4SOIpcdscbGiven = TRUE;
            break;
        case  B4SOI_MOD_PCDSCD :
            mod->B4SOIpcdscd = value->rValue;
            mod->B4SOIpcdscdGiven = TRUE;
            break;
        case B4SOI_MOD_PPCLM:
            mod->B4SOIppclm = value->rValue;
            mod->B4SOIppclmGiven = TRUE;
            break;                 
        case B4SOI_MOD_PPDIBL1:
            mod->B4SOIppdibl1 = value->rValue;
            mod->B4SOIppdibl1Given = TRUE;
            break;                 
        case B4SOI_MOD_PPDIBL2:
            mod->B4SOIppdibl2 = value->rValue;
            mod->B4SOIppdibl2Given = TRUE;
            break;                 
        case B4SOI_MOD_PPDIBLB:
            mod->B4SOIppdiblb = value->rValue;
            mod->B4SOIppdiblbGiven = TRUE;
            break;                 
        case  B4SOI_MOD_PDROUT:             
            mod->B4SOIpdrout = value->rValue;
            mod->B4SOIpdroutGiven = TRUE;
            break;
        case B4SOI_MOD_PPVAG:
            mod->B4SOIppvag = value->rValue;
            mod->B4SOIppvagGiven = TRUE;
            break;                 
        case  B4SOI_MOD_PDELTA :
            mod->B4SOIpdelta = value->rValue;
            mod->B4SOIpdeltaGiven = TRUE;
            break;
        case  B4SOI_MOD_PALPHA0 :
            mod->B4SOIpalpha0 = value->rValue;
            mod->B4SOIpalpha0Given = TRUE;
            break;
        case  B4SOI_MOD_PFBJTII :
            mod->B4SOIpfbjtii = value->rValue;
            mod->B4SOIpfbjtiiGiven = TRUE;
            break;
                        /*4.1 Iii model*/
        case  B4SOI_MOD_PEBJTII :
            mod->B4SOIpebjtii = value->rValue;
            mod->B4SOIpebjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_PCBJTII :
            mod->B4SOIpcbjtii = value->rValue;
            mod->B4SOIpcbjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_PVBCI :
            mod->B4SOIpvbci = value->rValue;
            mod->B4SOIpvbciGiven = TRUE;
            break;
        case  B4SOI_MOD_PABJTII :
            mod->B4SOIpabjtii = value->rValue;
            mod->B4SOIpabjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_PMBJTII :
            mod->B4SOIpmbjtii = value->rValue;
            mod->B4SOIpmbjtiiGiven = TRUE;
            break;
        case  B4SOI_MOD_PBETA0 :
            mod->B4SOIpbeta0 = value->rValue;
            mod->B4SOIpbeta0Given = TRUE;
            break;
        case  B4SOI_MOD_PBETA1 :
            mod->B4SOIpbeta1 = value->rValue;
            mod->B4SOIpbeta1Given = TRUE;
            break;
        case  B4SOI_MOD_PBETA2 :
            mod->B4SOIpbeta2 = value->rValue;
            mod->B4SOIpbeta2Given = TRUE;
            break;
        case  B4SOI_MOD_PVDSATII0 :
            mod->B4SOIpvdsatii0 = value->rValue;
            mod->B4SOIpvdsatii0Given = TRUE;
            break;
        case  B4SOI_MOD_PLII :
            mod->B4SOIplii = value->rValue;
            mod->B4SOIpliiGiven = TRUE;
            break;
        case  B4SOI_MOD_PESATII :
            mod->B4SOIpesatii = value->rValue;
            mod->B4SOIpesatiiGiven = TRUE;
            break;
        case  B4SOI_MOD_PSII0 :
            mod->B4SOIpsii0 = value->rValue;
            mod->B4SOIpsii0Given = TRUE;
            break;
        case  B4SOI_MOD_PSII1 :
            mod->B4SOIpsii1 = value->rValue;
            mod->B4SOIpsii1Given = TRUE;
            break;
        case  B4SOI_MOD_PSII2 :
            mod->B4SOIpsii2 = value->rValue;
            mod->B4SOIpsii2Given = TRUE;
            break;
        case  B4SOI_MOD_PSIID :
            mod->B4SOIpsiid = value->rValue;
            mod->B4SOIpsiidGiven = TRUE;
            break;
        case  B4SOI_MOD_PAGIDL :
            mod->B4SOIpagidl = value->rValue;
            mod->B4SOIpagidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PBGIDL :
            mod->B4SOIpbgidl = value->rValue;
            mod->B4SOIpbgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PCGIDL :
            mod->B4SOIpcgidl = value->rValue;
            mod->B4SOIpcgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PEGIDL :
            mod->B4SOIpegidl = value->rValue;
            mod->B4SOIpegidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PRGIDL :
            mod->B4SOIprgidl = value->rValue;
            mod->B4SOIprgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PKGIDL :
            mod->B4SOIpkgidl = value->rValue;
            mod->B4SOIpkgidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PFGIDL :
            mod->B4SOIpfgidl = value->rValue;
            mod->B4SOIpfgidlGiven = TRUE;
            break;
                        
        case  B4SOI_MOD_PAGISL :
            mod->B4SOIpagisl = value->rValue;
            mod->B4SOIpagislGiven = TRUE;
            break;
        case  B4SOI_MOD_PBGISL :
            mod->B4SOIpbgisl = value->rValue;
            mod->B4SOIpbgislGiven = TRUE;
            break;
        case  B4SOI_MOD_PCGISL :
            mod->B4SOIpcgisl = value->rValue;
            mod->B4SOIpcgislGiven = TRUE;
            break;
        case  B4SOI_MOD_PEGISL :
            mod->B4SOIpegisl = value->rValue;
            mod->B4SOIpegislGiven = TRUE;
            break;
        case  B4SOI_MOD_PRGISL :
            mod->B4SOIprgisl = value->rValue;
            mod->B4SOIprgislGiven = TRUE;
            break;
        case  B4SOI_MOD_PKGISL :
            mod->B4SOIpkgisl = value->rValue;
            mod->B4SOIpkgislGiven = TRUE;
            break;
        case  B4SOI_MOD_PFGISL :
            mod->B4SOIpfgisl = value->rValue;
            mod->B4SOIpfgislGiven = TRUE;
            break;        
        case  B4SOI_MOD_PNTUNS :        /* v4.0 */
            mod->B4SOIpntun = value->rValue;
            mod->B4SOIpntunGiven = TRUE;
            break;
        case  B4SOI_MOD_PNTUND :        /* v4.0 */
            mod->B4SOIpntund = value->rValue;
            mod->B4SOIpntundGiven = TRUE;
            break;
        case  B4SOI_MOD_PNDIODES :        /* v4.0 */
            mod->B4SOIpndiode = value->rValue;
            mod->B4SOIpndiodeGiven = TRUE;
            break;
        case  B4SOI_MOD_PNDIODED :        /* v4.0 */
            mod->B4SOIpndioded = value->rValue;
            mod->B4SOIpndiodedGiven = TRUE;
            break;
        case  B4SOI_MOD_PNRECF0S :        /* v4.0 */
            mod->B4SOIpnrecf0 = value->rValue;
            mod->B4SOIpnrecf0Given = TRUE;
            break;
        case  B4SOI_MOD_PNRECF0D :        /* v4.0 */
            mod->B4SOIpnrecf0d = value->rValue;
            mod->B4SOIpnrecf0dGiven = TRUE;
            break;
        case  B4SOI_MOD_PNRECR0S :        /* v4.0 */
            mod->B4SOIpnrecr0 = value->rValue;
            mod->B4SOIpnrecr0Given = TRUE;
            break;
        case  B4SOI_MOD_PNRECR0D :        /* v4.0 */
            mod->B4SOIpnrecr0d = value->rValue;
            mod->B4SOIpnrecr0dGiven = TRUE;
            break;
        case  B4SOI_MOD_PISBJT :
            mod->B4SOIpisbjt = value->rValue;
            mod->B4SOIpisbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_PIDBJT :        /* v4.0 */
            mod->B4SOIpidbjt = value->rValue;
            mod->B4SOIpidbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_PISDIF :
            mod->B4SOIpisdif = value->rValue;
            mod->B4SOIpisdifGiven = TRUE;
            break;
        case  B4SOI_MOD_PIDDIF :        /* v4.0 */
            mod->B4SOIpiddif = value->rValue;
            mod->B4SOIpiddifGiven = TRUE;
            break;
        case  B4SOI_MOD_PISREC :
            mod->B4SOIpisrec = value->rValue;
            mod->B4SOIpisrecGiven = TRUE;
            break;
        case  B4SOI_MOD_PIDREC :        /* v4.0 */
            mod->B4SOIpidrec = value->rValue;
            mod->B4SOIpidrecGiven = TRUE;
            break;
        case  B4SOI_MOD_PISTUN :
            mod->B4SOIpistun = value->rValue;
            mod->B4SOIpistunGiven = TRUE;
            break;
        case  B4SOI_MOD_PIDTUN :        /* v4.0 */
            mod->B4SOIpidtun = value->rValue;
            mod->B4SOIpidtunGiven = TRUE;
            break;
        case  B4SOI_MOD_PVREC0S :        /* v4.0 */
            mod->B4SOIpvrec0 = value->rValue;
            mod->B4SOIpvrec0Given = TRUE;
            break;
        case  B4SOI_MOD_PVREC0D :        /* v4.0 */
            mod->B4SOIpvrec0d = value->rValue;
            mod->B4SOIpvrec0dGiven = TRUE;
            break;
        case  B4SOI_MOD_PVTUN0S :        /* v4.0 */
            mod->B4SOIpvtun0 = value->rValue;
            mod->B4SOIpvtun0Given = TRUE;
            break;
        case  B4SOI_MOD_PVTUN0D :        /* v4.0 */
            mod->B4SOIpvtun0d = value->rValue;
            mod->B4SOIpvtun0dGiven = TRUE;
            break;
        case  B4SOI_MOD_PNBJT :
            mod->B4SOIpnbjt = value->rValue;
            mod->B4SOIpnbjtGiven = TRUE;
            break;
        case  B4SOI_MOD_PLBJT0 :
            mod->B4SOIplbjt0 = value->rValue;
            mod->B4SOIplbjt0Given = TRUE;
            break;
        case  B4SOI_MOD_PVABJT :
            mod->B4SOIpvabjt = value->rValue;
            mod->B4SOIpvabjtGiven = TRUE;
            break;
        case  B4SOI_MOD_PAELY :
            mod->B4SOIpaely = value->rValue;
            mod->B4SOIpaelyGiven = TRUE;
            break;
        case  B4SOI_MOD_PAHLIS :        /* v4.0 */
            mod->B4SOIpahli = value->rValue;
            mod->B4SOIpahliGiven = TRUE;
            break;
        case  B4SOI_MOD_PAHLID :        /* v4.0 */
            mod->B4SOIpahlid = value->rValue;
            mod->B4SOIpahlidGiven = TRUE;
            break;

/* v3.1 for RF */
        case B4SOI_MOD_PXRCRG1 :
            mod->B4SOIpxrcrg1 = value->rValue;
            mod->B4SOIpxrcrg1Given = TRUE;
            break;
        case B4SOI_MOD_PXRCRG2 :
            mod->B4SOIpxrcrg2 = value->rValue;
            mod->B4SOIpxrcrg2Given = TRUE;
            break;
/* v3.1 for RF end */

        /* CV Model */
        case  B4SOI_MOD_PVSDFB :
            mod->B4SOIpvsdfb = value->rValue;
            mod->B4SOIpvsdfbGiven = TRUE;
            break;
        case  B4SOI_MOD_PVSDTH :
            mod->B4SOIpvsdth = value->rValue;
            mod->B4SOIpvsdthGiven = TRUE;
            break;
        case  B4SOI_MOD_PDELVT :
            mod->B4SOIpdelvt = value->rValue;
            mod->B4SOIpdelvtGiven = TRUE;
            break;
        case  B4SOI_MOD_PACDE :
            mod->B4SOIpacde = value->rValue;
            mod->B4SOIpacdeGiven = TRUE;
            break;
        case  B4SOI_MOD_PMOIN :
            mod->B4SOIpmoin = value->rValue;
            mod->B4SOIpmoinGiven = TRUE;
            break;
        case  B4SOI_MOD_PNOFF :
            mod->B4SOIpnoff = value->rValue;
            mod->B4SOIpnoffGiven = TRUE;
            break; /* v3.2 */
/* Added for binning - END */

        /* 4.0 backward compatibility  */
        case  B4SOI_MOD_NLX:
            mod->B4SOInlx = value->rValue;
            mod->B4SOInlxGiven = TRUE;
            break;
        case  B4SOI_MOD_LNLX:
            mod->B4SOIlnlx = value->rValue;
            mod->B4SOIlnlxGiven = TRUE;
            break;
        case  B4SOI_MOD_WNLX:
            mod->B4SOIwnlx = value->rValue;
            mod->B4SOIwnlxGiven = TRUE;
            break;
        case  B4SOI_MOD_PNLX:
            mod->B4SOIpnlx = value->rValue;
            mod->B4SOIpnlxGiven = TRUE;
            break;
        case  B4SOI_MOD_NGIDL:
            mod->B4SOIngidl = value->rValue;
            mod->B4SOIngidlGiven = TRUE;
            break;
        case  B4SOI_MOD_LNGIDL:
            mod->B4SOIlngidl = value->rValue;
            mod->B4SOIlngidlGiven = TRUE;
            break;
        case  B4SOI_MOD_WNGIDL:
            mod->B4SOIwngidl = value->rValue;
            mod->B4SOIwngidlGiven = TRUE;
            break;
        case  B4SOI_MOD_PNGIDL:
            mod->B4SOIpngidl = value->rValue;
            mod->B4SOIpngidlGiven = TRUE;
            break;

        case B4SOI_MOD_VGS_MAX:
            mod->B4SOIvgsMax = value->rValue;
            mod->B4SOIvgsMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VGD_MAX:
            mod->B4SOIvgdMax = value->rValue;
            mod->B4SOIvgdMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VGB_MAX:
            mod->B4SOIvgbMax = value->rValue;
            mod->B4SOIvgbMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VDS_MAX:
            mod->B4SOIvdsMax = value->rValue;
            mod->B4SOIvdsMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VBS_MAX:
            mod->B4SOIvbsMax = value->rValue;
            mod->B4SOIvbsMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VBD_MAX:
            mod->B4SOIvbdMax = value->rValue;
            mod->B4SOIvbdMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VGSR_MAX:
            mod->B4SOIvgsrMax = value->rValue;
            mod->B4SOIvgsrMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VGDR_MAX:
            mod->B4SOIvgdrMax = value->rValue;
            mod->B4SOIvgdrMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VGBR_MAX:
            mod->B4SOIvgbrMax = value->rValue;
            mod->B4SOIvgbrMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VBSR_MAX:
            mod->B4SOIvbsrMax = value->rValue;
            mod->B4SOIvbsrMaxGiven = TRUE;
            break;
        case B4SOI_MOD_VBDR_MAX:
            mod->B4SOIvbdrMax = value->rValue;
            mod->B4SOIvbdrMaxGiven = TRUE;
            break;

        case  B4SOI_MOD_NMOS  :
            if(value->iValue) {
                mod->B4SOItype = 1;
                mod->B4SOItypeGiven = TRUE;
            }
            break;
        case  B4SOI_MOD_PMOS  :
            if(value->iValue) {
                mod->B4SOItype = - 1;
                mod->B4SOItypeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


