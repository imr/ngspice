/***  B4SOI 12/16/2010 Released by Tanvir Morshed  ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soimask.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * File: b4soimask.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 * Modified by Tanvir Morshed 12/16/2010
 **********/

#include "ngspice/ngspice.h"

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b4soidef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B4SOImAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    B4SOImodel *model = (B4SOImodel *)inst;
    NG_IGNORE(ckt);
    switch(which) 
    {   case B4SOI_MOD_MOBMOD:
            value->iValue = model->B4SOImobMod; 
            return(OK);
        case B4SOI_MOD_PARAMCHK:
            value->iValue = model->B4SOIparamChk; 
            return(OK);
        case B4SOI_MOD_BINUNIT:
            value->iValue = model->B4SOIbinUnit; 
            return(OK);
        case B4SOI_MOD_CAPMOD:
            value->iValue = model->B4SOIcapMod; 
            return(OK);
        case B4SOI_MOD_SHMOD:
            value->iValue = model->B4SOIshMod; 
            return(OK);

/*        case B4SOI_MOD_NOIMOD:
            value->iValue = model->B4SOInoiMod; 
            return(OK); v3.2 */

        case  B4SOI_MOD_VERSION :
          value->rValue = model->B4SOIversion;
            return(OK);
        case  B4SOI_MOD_TOX :
          value->rValue = model->B4SOItox;
            return(OK);
            case  B4SOI_MOD_TOXP :
          value->rValue = model->B4SOItoxp;
            return(OK);
            case  B4SOI_MOD_LEFFEOT :
          value->rValue = model->B4SOIleffeot;
            return(OK);
            case  B4SOI_MOD_WEFFEOT :
          value->rValue = model->B4SOIweffeot;
            return(OK);
            case  B4SOI_MOD_VDDEOT :
          value->rValue = model->B4SOIvddeot;
            return(OK);
            case  B4SOI_MOD_TEMPEOT :
          value->rValue = model->B4SOItempeot;
            return(OK);
            case  B4SOI_MOD_ADOS :
          value->rValue = model->B4SOIados;
            return(OK);
            case  B4SOI_MOD_BDOS :
          value->rValue = model->B4SOIbdos;
            return(OK);
        case B4SOI_MOD_EPSRGATE:
            value->rValue = model->B4SOIepsrgate;
                return(OK);
        case B4SOI_MOD_PHIG:
            value->rValue = model->B4SOIphig;
                return(OK);                
        case B4SOI_MOD_EASUB:
            value->rValue = model->B4SOIeasub;
            return(OK);                
                
        case  B4SOI_MOD_TOXM :
          value->rValue = model->B4SOItoxm;
            return(OK); /* v3.2 */
                        
            /*4.1*/
                case  B4SOI_MOD_EOT :
          value->rValue = model->B4SOIeot;
                  return(OK);
                case  B4SOI_MOD_EPSROX :
          value->rValue = model->B4SOIepsrox;
            return(OK);
            case B4SOI_MOD_EPSRSUB:
            value->rValue = model->B4SOIepsrsub;
            return(OK);
            case B4SOI_MOD_NI0SUB:
            value->rValue = model->B4SOIni0sub;
            return(OK);
            case B4SOI_MOD_BG0SUB:
            value->rValue = model->B4SOIbg0sub;
            return(OK);
            case B4SOI_MOD_TBGASUB:
            value->rValue = model->B4SOItbgasub;
            return(OK);
            case B4SOI_MOD_TBGBSUB:
            value->rValue = model->B4SOItbgbsub;
            return(OK);
/* v2.2.3 */
        case  B4SOI_MOD_DTOXCV :
          value->rValue = model->B4SOIdtoxcv;
            return(OK);

        case  B4SOI_MOD_CDSC :
          value->rValue = model->B4SOIcdsc;
            return(OK);
        case  B4SOI_MOD_CDSCB :
          value->rValue = model->B4SOIcdscb;
            return(OK);

        case  B4SOI_MOD_CDSCD :
          value->rValue = model->B4SOIcdscd;
            return(OK);

        case  B4SOI_MOD_CIT :
          value->rValue = model->B4SOIcit;
            return(OK);
        case  B4SOI_MOD_NFACTOR :
          value->rValue = model->B4SOInfactor;
            return(OK);
        case B4SOI_MOD_VSAT:
            value->rValue = model->B4SOIvsat;
            return(OK);
        case B4SOI_MOD_AT:
            value->rValue = model->B4SOIat;
            return(OK);
        case B4SOI_MOD_A0:
            value->rValue = model->B4SOIa0;
            return(OK);

        case B4SOI_MOD_AGS:
            value->rValue = model->B4SOIags;
            return(OK);

        case B4SOI_MOD_A1:
            value->rValue = model->B4SOIa1;
            return(OK);
        case B4SOI_MOD_A2:
            value->rValue = model->B4SOIa2;
            return(OK);
        case B4SOI_MOD_KETA:
            value->rValue = model->B4SOIketa;
            return(OK);   
        case B4SOI_MOD_NSUB:
            value->rValue = model->B4SOInsub;
            return(OK);
        case B4SOI_MOD_NPEAK:
            value->rValue = model->B4SOInpeak;
            return(OK);
        case B4SOI_MOD_NGATE:
            value->rValue = model->B4SOIngate;
            return(OK);
        case B4SOI_MOD_NSD:
            value->rValue = model->B4SOInsd;
            return(OK);
        case B4SOI_MOD_GAMMA1:
            value->rValue = model->B4SOIgamma1;
            return(OK);
        case B4SOI_MOD_GAMMA2:
            value->rValue = model->B4SOIgamma2;
            return(OK);
        case B4SOI_MOD_VBX:
            value->rValue = model->B4SOIvbx;
            return(OK);
        case B4SOI_MOD_VBM:
            value->rValue = model->B4SOIvbm;
            return(OK);
        case B4SOI_MOD_XT:
            value->rValue = model->B4SOIxt;
            return(OK);
        case  B4SOI_MOD_K1:
          value->rValue = model->B4SOIk1;
            return(OK);
        case  B4SOI_MOD_KT1:
          value->rValue = model->B4SOIkt1;
            return(OK);
        case  B4SOI_MOD_KT1L:
          value->rValue = model->B4SOIkt1l;
            return(OK);
        case  B4SOI_MOD_KT2 :
          value->rValue = model->B4SOIkt2;
            return(OK);
        case  B4SOI_MOD_K2 :
          value->rValue = model->B4SOIk2;
            return(OK);
        case  B4SOI_MOD_K3:
          value->rValue = model->B4SOIk3;
            return(OK);
        case  B4SOI_MOD_K3B:
          value->rValue = model->B4SOIk3b;
            return(OK);
        case  B4SOI_MOD_W0:
          value->rValue = model->B4SOIw0;
            return(OK);
        case  B4SOI_MOD_LPE0:
          value->rValue = model->B4SOIlpe0;
            return(OK);
        case  B4SOI_MOD_LPEB: /* v4.0 for Vth */
          value->rValue = model->B4SOIlpeb;
            return(OK);
        case  B4SOI_MOD_DVT0 :                
          value->rValue = model->B4SOIdvt0;
            return(OK);
        case  B4SOI_MOD_DVT1 :             
          value->rValue = model->B4SOIdvt1;
            return(OK);
        case  B4SOI_MOD_DVT2 :             
          value->rValue = model->B4SOIdvt2;
            return(OK);
        case  B4SOI_MOD_DVT0W :                
          value->rValue = model->B4SOIdvt0w;
            return(OK);
        case  B4SOI_MOD_DVT1W :             
          value->rValue = model->B4SOIdvt1w;
            return(OK);
        case  B4SOI_MOD_DVT2W :             
          value->rValue = model->B4SOIdvt2w;
            return(OK);
        case  B4SOI_MOD_DROUT :           
          value->rValue = model->B4SOIdrout;
            return(OK);
        case  B4SOI_MOD_DSUB :           
          value->rValue = model->B4SOIdsub;
            return(OK);
        case B4SOI_MOD_VTH0:
            value->rValue = model->B4SOIvth0; 
            return(OK);
        case B4SOI_MOD_VFB:
            value->rValue = model->B4SOIvfb;
            return(OK); /* v4.1 */
        case B4SOI_MOD_UA:
            value->rValue = model->B4SOIua; 
            return(OK);
        case B4SOI_MOD_UA1:
            value->rValue = model->B4SOIua1; 
            return(OK);
        case B4SOI_MOD_UB:
            value->rValue = model->B4SOIub;  
            return(OK);
        case B4SOI_MOD_UB1:
            value->rValue = model->B4SOIub1;  
            return(OK);
        case B4SOI_MOD_UC:
            value->rValue = model->B4SOIuc; 
            return(OK);
        case B4SOI_MOD_UC1:
            value->rValue = model->B4SOIuc1; 
            return(OK);
        case B4SOI_MOD_U0:
            value->rValue = model->B4SOIu0;
            return(OK);
        case B4SOI_MOD_UTE:
            value->rValue = model->B4SOIute;
            return(OK);
                        
                        /*4.1 mobmod=4*/
        case B4SOI_MOD_UD:
                    value->rValue = model->B4SOIud;
            return(OK);        
        case B4SOI_MOD_LUD:
                    value->rValue = model->B4SOIlud;
            return(OK);        
        case B4SOI_MOD_WUD:
                    value->rValue = model->B4SOIwud;
            return(OK);        
        case B4SOI_MOD_PUD:
                    value->rValue = model->B4SOIpud;
            return(OK);        
        case B4SOI_MOD_UD1:
                    value->rValue = model->B4SOIud1;
            return(OK);        
        case B4SOI_MOD_LUD1:
                    value->rValue = model->B4SOIlud1;
            return(OK);        
        case B4SOI_MOD_WUD1:
                    value->rValue = model->B4SOIwud1;
            return(OK);        
        case B4SOI_MOD_PUD1:
                    value->rValue = model->B4SOIpud1;
            return(OK);                                
            case B4SOI_MOD_EU:
                    value->rValue = model->B4SOIeu;
            return(OK);
            case B4SOI_MOD_LEU:
                    value->rValue = model->B4SOIleu;
            return(OK);
            case B4SOI_MOD_WEU:
                    value->rValue = model->B4SOIweu;
            return(OK);
            case B4SOI_MOD_PEU:
                    value->rValue = model->B4SOIpeu;
            return(OK);
            case B4SOI_MOD_UCS:
                    value->rValue = model->B4SOIucs;
            return(OK);
                case B4SOI_MOD_LUCS:
                    value->rValue = model->B4SOIlucs;
            return(OK);
                case B4SOI_MOD_WUCS:
                    value->rValue = model->B4SOIwucs;
            return(OK);
                case B4SOI_MOD_PUCS:
                    value->rValue = model->B4SOIpucs;
            return(OK);
                case B4SOI_MOD_UCSTE:
            value->rValue = model->B4SOIucste;
            return(OK);
                case B4SOI_MOD_LUCSTE:
            value->rValue = model->B4SOIlucste;
            return(OK);
                case B4SOI_MOD_WUCSTE:
            value->rValue = model->B4SOIwucste;
            return(OK);
                case B4SOI_MOD_PUCSTE:
            value->rValue = model->B4SOIpucste;
            return(OK);                        
                        
        case B4SOI_MOD_VOFF:
            value->rValue = model->B4SOIvoff;
            return(OK);
        case B4SOI_MOD_DELTA:
            value->rValue = model->B4SOIdelta;
            return(OK);
        case B4SOI_MOD_RDSW:
            value->rValue = model->B4SOIrdsw; 
            return(OK);            
        case B4SOI_MOD_RDWMIN:
            value->rValue = model->B4SOIrdwmin;
            return(OK);
        case B4SOI_MOD_RSWMIN:
            value->rValue = model->B4SOIrswmin;
            return(OK);
        case B4SOI_MOD_RDW:
            value->rValue = model->B4SOIrdw;
            return(OK);
        case B4SOI_MOD_RSW:
            value->rValue = model->B4SOIrsw;
            return(OK);
        case B4SOI_MOD_PRWG:
            value->rValue = model->B4SOIprwg; 
            return(OK);             
        case B4SOI_MOD_PRWB:
            value->rValue = model->B4SOIprwb; 
            return(OK);             
        case B4SOI_MOD_PRT:
            value->rValue = model->B4SOIprt; 
            return(OK);              
        case B4SOI_MOD_ETA0:
            value->rValue = model->B4SOIeta0; 
            return(OK);               
        case B4SOI_MOD_ETAB:
            value->rValue = model->B4SOIetab; 
            return(OK);               
        case B4SOI_MOD_PCLM:
            value->rValue = model->B4SOIpclm; 
            return(OK);               
        case B4SOI_MOD_PDIBL1:
            value->rValue = model->B4SOIpdibl1; 
            return(OK);               
        case B4SOI_MOD_PDIBL2:
            value->rValue = model->B4SOIpdibl2; 
            return(OK);               
        case B4SOI_MOD_PDIBLB:
            value->rValue = model->B4SOIpdiblb; 
            return(OK);               
        case B4SOI_MOD_PVAG:
            value->rValue = model->B4SOIpvag; 
            return(OK);               
        case B4SOI_MOD_WR:
            value->rValue = model->B4SOIwr;
            return(OK);
        case B4SOI_MOD_DWG:
            value->rValue = model->B4SOIdwg;
            return(OK);
        case B4SOI_MOD_DWB:
            value->rValue = model->B4SOIdwb;
            return(OK);
        case B4SOI_MOD_B0:
            value->rValue = model->B4SOIb0;
            return(OK);
        case B4SOI_MOD_B1:
            value->rValue = model->B4SOIb1;
            return(OK);
        case B4SOI_MOD_ALPHA0:
            value->rValue = model->B4SOIalpha0;
            return(OK);

        case B4SOI_MOD_CGSL:
            value->rValue = model->B4SOIcgsl;
            return(OK);
        case B4SOI_MOD_CGDL:
            value->rValue = model->B4SOIcgdl;
            return(OK);
        case B4SOI_MOD_CKAPPA:
            value->rValue = model->B4SOIckappa;
            return(OK);
        case B4SOI_MOD_CF:
            value->rValue = model->B4SOIcf;
            return(OK);
        case B4SOI_MOD_CLC:
            value->rValue = model->B4SOIclc;
            return(OK);
        case B4SOI_MOD_CLE:
            value->rValue = model->B4SOIcle;
            return(OK);
        case B4SOI_MOD_DWC:
            value->rValue = model->B4SOIdwc;
            return(OK);
        case B4SOI_MOD_DLC:
            value->rValue = model->B4SOIdlc;
            return(OK);


        case B4SOI_MOD_TBOX:
            value->rValue = model->B4SOItbox; 
            return(OK);
        case B4SOI_MOD_TSI:
            value->rValue = model->B4SOItsi; 
            return(OK);
                case B4SOI_MOD_ETSI:
            value->rValue = model->B4SOIetsi; 
            return(OK);
        case B4SOI_MOD_RTH0:
            value->rValue = model->B4SOIrth0; 
            return(OK);
        case B4SOI_MOD_CTH0:
            value->rValue = model->B4SOIcth0; 
            return(OK);
        case B4SOI_MOD_NDIODES: /* v4.0 */
            value->rValue = model->B4SOIndiode; 
            return(OK);
        case B4SOI_MOD_NDIODED: /* v4.0 */
            value->rValue = model->B4SOIndioded; 
            return(OK);
        case B4SOI_MOD_XBJT:
            value->rValue = model->B4SOIxbjt; 
            return(OK);
        case B4SOI_MOD_XDIFS:
            value->rValue = model->B4SOIxdif;
            return(OK);
        case B4SOI_MOD_XRECS:
            value->rValue = model->B4SOIxrec;
            return(OK);
        case B4SOI_MOD_XTUNS:
            value->rValue = model->B4SOIxtun;
            return(OK);
        case B4SOI_MOD_XDIFD:
            value->rValue = model->B4SOIxdifd;
            return(OK);
        case B4SOI_MOD_XRECD:
            value->rValue = model->B4SOIxrecd;
            return(OK);
        case B4SOI_MOD_XTUND:
            value->rValue = model->B4SOIxtund;
            return(OK);

         case B4SOI_MOD_TT:
            value->rValue = model->B4SOItt; 
            return(OK);
        case B4SOI_MOD_VSDTH:
            value->rValue = model->B4SOIvsdth; 
            return(OK);
        case B4SOI_MOD_VSDFB:
            value->rValue = model->B4SOIvsdfb; 
            return(OK);
        case B4SOI_MOD_CSDMIN:
            value->rValue = model->B4SOIcsdmin; 
            return(OK);
        case B4SOI_MOD_ASD:
            value->rValue = model->B4SOIasd; 
            return(OK);

        case  B4SOI_MOD_TNOM :
          value->rValue = model->B4SOItnom;
            return(OK);
        case B4SOI_MOD_CGSO:
            value->rValue = model->B4SOIcgso; 
            return(OK);
        case B4SOI_MOD_CGDO:
            value->rValue = model->B4SOIcgdo; 
            return(OK);
        case B4SOI_MOD_CGEO:
            value->rValue = model->B4SOIcgeo; 
            return(OK);


        case B4SOI_MOD_XPART:
            value->rValue = model->B4SOIxpart; 
            return(OK);
        case B4SOI_MOD_RSH:
            value->rValue = model->B4SOIsheetResistance; 
            return(OK);
        case B4SOI_MOD_PBSWGS:        /* v4.0 */
            value->rValue = model->B4SOIGatesidewallJctSPotential; 
            return(OK);
        case B4SOI_MOD_PBSWGD:        /* v4.0 */
            value->rValue = model->B4SOIGatesidewallJctDPotential; 
            return(OK);
        case B4SOI_MOD_MJSWGS:        /* v4.0 */
            value->rValue = model->B4SOIbodyJctGateSideSGradingCoeff; 
            return(OK);
        case B4SOI_MOD_MJSWGD:        /* v4.0 */
            value->rValue = model->B4SOIbodyJctGateSideDGradingCoeff; 
            return(OK);
        case B4SOI_MOD_CJSWGS:        /* v4.0 */
            value->rValue = model->B4SOIunitLengthGateSidewallJctCapS; 
            return(OK);
        case B4SOI_MOD_CJSWGD:        /* v4.0 */
            value->rValue = model->B4SOIunitLengthGateSidewallJctCapD; 
            return(OK);
        case B4SOI_MOD_CSDESW:
            value->rValue = model->B4SOIcsdesw; 
            return(OK);
        case B4SOI_MOD_LINT:
            value->rValue = model->B4SOILint; 
            return(OK);
        case B4SOI_MOD_LL:
            value->rValue = model->B4SOILl;
            return(OK);
/* v2.2.3 */
        case B4SOI_MOD_LLC:
            value->rValue = model->B4SOILlc;
            return(OK);

        case B4SOI_MOD_LLN:
            value->rValue = model->B4SOILln;
            return(OK);
        case B4SOI_MOD_LW:
            value->rValue = model->B4SOILw;
            return(OK);
/* v2.2.3 */
        case B4SOI_MOD_LWC:
            value->rValue = model->B4SOILwc;
            return(OK);

        case B4SOI_MOD_LWN:
            value->rValue = model->B4SOILwn;
            return(OK);
        case B4SOI_MOD_LWL:
            value->rValue = model->B4SOILwl;
            return(OK);
/* v2.2.3 */
        case B4SOI_MOD_LWLC:
            value->rValue = model->B4SOILwlc;
            return(OK);

        case B4SOI_MOD_WINT:
            value->rValue = model->B4SOIWint;
            return(OK);
        case B4SOI_MOD_WL:
            value->rValue = model->B4SOIWl;
            return(OK);
/* v2.2.3 */
        case B4SOI_MOD_WLC:
            value->rValue = model->B4SOIWlc;
            return(OK);

        case B4SOI_MOD_WLN:
            value->rValue = model->B4SOIWln;
            return(OK);
        case B4SOI_MOD_WW:
            value->rValue = model->B4SOIWw;
            return(OK);
/* v2.2.3 */
        case B4SOI_MOD_WWC:
            value->rValue = model->B4SOIWwc;
            return(OK);

        case B4SOI_MOD_WWN:
            value->rValue = model->B4SOIWwn;
            return(OK);
        case B4SOI_MOD_WWL:
            value->rValue = model->B4SOIWwl;
            return(OK);
/* v2.2.3 */
        case B4SOI_MOD_WWLC:
            value->rValue = model->B4SOIWwlc;
            return(OK);

        /* stress effect */
        case B4SOI_MOD_SAREF:
            value->rValue = model->B4SOIsaref;
            return(OK);
        case B4SOI_MOD_SBREF:
            value->rValue = model->B4SOIsbref;
            return(OK);
        case B4SOI_MOD_WLOD:
            value->rValue = model->B4SOIwlod;
            return(OK);
        case B4SOI_MOD_KU0:
            value->rValue = model->B4SOIku0;
            return(OK);
        case B4SOI_MOD_KVSAT:
            value->rValue = model->B4SOIkvsat;
            return(OK);
        case B4SOI_MOD_KVTH0:
            value->rValue = model->B4SOIkvth0;
            return(OK);
        case B4SOI_MOD_TKU0:
            value->rValue = model->B4SOItku0;
            return(OK);
        case B4SOI_MOD_LLODKU0:
            value->rValue = model->B4SOIllodku0;
            return(OK);
        case B4SOI_MOD_WLODKU0:
            value->rValue = model->B4SOIwlodku0;
            return(OK);
        case B4SOI_MOD_LLODVTH:
            value->rValue = model->B4SOIllodvth;
            return(OK);
        case B4SOI_MOD_WLODVTH:
            value->rValue = model->B4SOIwlodvth;
            return(OK);
        case B4SOI_MOD_LKU0:
            value->rValue = model->B4SOIlku0;
            return(OK);
        case B4SOI_MOD_WKU0:
            value->rValue = model->B4SOIwku0;
            return(OK);
        case B4SOI_MOD_PKU0:
            value->rValue = model->B4SOIpku0;
            return(OK);
        case B4SOI_MOD_LKVTH0:
            value->rValue = model->B4SOIlkvth0;
            return(OK);
        case B4SOI_MOD_WKVTH0:
            value->rValue = model->B4SOIwkvth0;
            return(OK);
        case B4SOI_MOD_PKVTH0:
            value->rValue = model->B4SOIpkvth0;
            return(OK);
        case B4SOI_MOD_STK2:
            value->rValue = model->B4SOIstk2;
            return(OK);
        case B4SOI_MOD_LODK2:
            value->rValue = model->B4SOIlodk2;
            return(OK);
        case B4SOI_MOD_STETA0:
            value->rValue = model->B4SOIsteta0;
            return(OK);
        case B4SOI_MOD_LODETA0:
            value->rValue = model->B4SOIlodeta0;
            return(OK);

/* added for stress end */



        case B4SOI_MOD_NOIA:
            value->rValue = model->B4SOIoxideTrapDensityA;
            return(OK);
        case B4SOI_MOD_NOIB:
            value->rValue = model->B4SOIoxideTrapDensityB;
            return(OK);
        case B4SOI_MOD_NOIC:
            value->rValue = model->B4SOIoxideTrapDensityC;
            return(OK);
        case B4SOI_MOD_NOIF:
            value->rValue = model->B4SOInoif;
            return(OK);
        case B4SOI_MOD_EM:
            value->rValue = model->B4SOIem;
            return(OK);
        case B4SOI_MOD_EF:
            value->rValue = model->B4SOIef;
            return(OK);
        case B4SOI_MOD_AF:
            value->rValue = model->B4SOIaf;
            return(OK);
        case B4SOI_MOD_KF:
            value->rValue = model->B4SOIkf;
            return(OK);
        case B4SOI_MOD_BF:
            value->rValue = model->B4SOIbf;
            return(OK);
        case B4SOI_MOD_W0FLK:
            value->rValue = model->B4SOIw0flk;
            return(OK);


/* v2.0 release */
        case B4SOI_MOD_K1W1:                    
            value->rValue = model->B4SOIk1w1;
            return(OK);
        case B4SOI_MOD_K1W2:
            value->rValue = model->B4SOIk1w2;
            return(OK);
        case B4SOI_MOD_KETAS:
            value->rValue = model->B4SOIketas;
            return(OK);
        case B4SOI_MOD_DWBC:
            value->rValue = model->B4SOIdwbc;
            return(OK);
        case B4SOI_MOD_BETA0:
            value->rValue = model->B4SOIbeta0;
            return(OK);
        case B4SOI_MOD_BETA1:
            value->rValue = model->B4SOIbeta1;
            return(OK);
        case B4SOI_MOD_BETA2:
            value->rValue = model->B4SOIbeta2;
            return(OK);
        case B4SOI_MOD_VDSATII0:
            value->rValue = model->B4SOIvdsatii0;
            return(OK);
        case B4SOI_MOD_TII:
            value->rValue = model->B4SOItii;
            return(OK);
                        /*4.1 Iii model*/
                case B4SOI_MOD_TVBCI:
            value->rValue = model->B4SOItvbci;
            return(OK);        
        case B4SOI_MOD_LII:
            value->rValue = model->B4SOIlii;
            return(OK);
        case B4SOI_MOD_SII0:
            value->rValue = model->B4SOIsii0;
            return(OK);
        case B4SOI_MOD_SII1:
            value->rValue = model->B4SOIsii1;
            return(OK);
        case B4SOI_MOD_SII2:
            value->rValue = model->B4SOIsii2;
            return(OK);
        case B4SOI_MOD_SIID:
            value->rValue = model->B4SOIsiid;
            return(OK);
        case B4SOI_MOD_FBJTII:
            value->rValue = model->B4SOIfbjtii;
            return(OK);
        /*4.1 Iii model*/
        case B4SOI_MOD_EBJTII:
            value->rValue = model->B4SOIebjtii;
            return(OK);
        case B4SOI_MOD_CBJTII:
            value->rValue = model->B4SOIcbjtii;
            return(OK);
        case B4SOI_MOD_VBCI:
            value->rValue = model->B4SOIvbci;
            return(OK);
        case B4SOI_MOD_ABJTII:
            value->rValue = model->B4SOIabjtii;
            return(OK);
        case B4SOI_MOD_MBJTII:
            value->rValue = model->B4SOImbjtii;
            return(OK);
        case B4SOI_MOD_ESATII:
            value->rValue = model->B4SOIesatii;
            return(OK);
        case B4SOI_MOD_NTUNS:                /* v4.0 */
            value->rValue = model->B4SOIntun;
            return(OK);
        case B4SOI_MOD_NTUND:                /* v4.0 */
            value->rValue = model->B4SOIntund;
            return(OK);
        case B4SOI_MOD_NRECF0S:                /* v4.0 */                
            value->rValue = model->B4SOInrecf0;
            return(OK);
        case B4SOI_MOD_NRECF0D:                /* v4.0 */                
            value->rValue = model->B4SOInrecf0d;
            return(OK);
        case B4SOI_MOD_NRECR0S:                /* v4.0 */
            value->rValue = model->B4SOInrecr0;
            return(OK);
        case B4SOI_MOD_NRECR0D:                /* v4.0 */
            value->rValue = model->B4SOInrecr0d;
            return(OK);
        case B4SOI_MOD_ISBJT:
            value->rValue = model->B4SOIisbjt;
            return(OK);
        case B4SOI_MOD_IDBJT:        /* v4.0 */
            value->rValue = model->B4SOIidbjt;
            return(OK);
        case B4SOI_MOD_ISDIF:
            value->rValue = model->B4SOIisdif;
            return(OK);
        case B4SOI_MOD_IDDIF:        /* v4.0 */
            value->rValue = model->B4SOIiddif;
            return(OK);
        case B4SOI_MOD_ISREC:
            value->rValue = model->B4SOIisrec;
            return(OK);
        case B4SOI_MOD_IDREC:        /* v4.0 */
            value->rValue = model->B4SOIidrec;
            return(OK);
        case B4SOI_MOD_ISTUN:
            value->rValue = model->B4SOIistun;
            return(OK);
        case B4SOI_MOD_IDTUN:        /* v4.0 */
            value->rValue = model->B4SOIidtun;
            return(OK);
        case B4SOI_MOD_LN:
            value->rValue = model->B4SOIln;
            return(OK);
        case B4SOI_MOD_VREC0S:        /* v4.0 */
            value->rValue = model->B4SOIvrec0;
            return(OK);
        case B4SOI_MOD_VREC0D:        /* v4.0 */
            value->rValue = model->B4SOIvrec0d;
            return(OK);
        case B4SOI_MOD_VTUN0S:        /* v4.0 */
            value->rValue = model->B4SOIvtun0;
            return(OK);
        case B4SOI_MOD_VTUN0D:        /* v4.0 */
            value->rValue = model->B4SOIvtun0d;
            return(OK);
        case B4SOI_MOD_NBJT:
            value->rValue = model->B4SOInbjt;
            return(OK);
        case B4SOI_MOD_LBJT0:
            value->rValue = model->B4SOIlbjt0;
            return(OK);
        case B4SOI_MOD_LDIF0:
            value->rValue = model->B4SOIldif0;
            return(OK);
        case B4SOI_MOD_VABJT:
            value->rValue = model->B4SOIvabjt;
            return(OK);
        case B4SOI_MOD_AELY:
            value->rValue = model->B4SOIaely;
            return(OK);
        case B4SOI_MOD_AHLIS:        /* v4.0 */
            value->rValue = model->B4SOIahli;
            return(OK);
        case B4SOI_MOD_AHLID:        /* v4.0 */
            value->rValue = model->B4SOIahlid;
            return(OK);
        case B4SOI_MOD_RBODY:
            value->rValue = model->B4SOIrbody;
            return(OK);
        case B4SOI_MOD_RBSH:
            value->rValue = model->B4SOIrbsh;
            return(OK);
        case B4SOI_MOD_NTRECF:
            value->rValue = model->B4SOIntrecf;
            return(OK);
        case B4SOI_MOD_NTRECR:
            value->rValue = model->B4SOIntrecr;
            return(OK);
        case B4SOI_MOD_NDIF:
            value->rValue = model->B4SOIndif;
            return(OK);
        case B4SOI_MOD_DLCB:
            value->rValue = model->B4SOIdlcb;
            return(OK);
        case B4SOI_MOD_FBODY:
            value->rValue = model->B4SOIfbody;
            return(OK);
        case B4SOI_MOD_TCJSWGS:
            value->rValue = model->B4SOItcjswg;
            return(OK);
        case B4SOI_MOD_TPBSWGS:
            value->rValue = model->B4SOItpbswg;
            return(OK);
        case B4SOI_MOD_TCJSWGD:
            value->rValue = model->B4SOItcjswgd;
            return(OK);
        case B4SOI_MOD_TPBSWGD:
            value->rValue = model->B4SOItpbswgd;
            return(OK);
        case B4SOI_MOD_ACDE:
            value->rValue = model->B4SOIacde;
            return(OK);
        case B4SOI_MOD_MOIN:
            value->rValue = model->B4SOImoin;
            return(OK);
        case B4SOI_MOD_NOFF:
            value->rValue = model->B4SOInoff;
            return(OK); /* v3.2 */
        case B4SOI_MOD_DELVT:
            value->rValue = model->B4SOIdelvt;
            return(OK);
        case  B4SOI_MOD_KB1:
            value->rValue = model->B4SOIkb1;
            return(OK);
        case B4SOI_MOD_DLBG:
            value->rValue = model->B4SOIdlbg;
            return(OK);
/* v4.4 */
        case B4SOI_MOD_CFRCOEFF:
            value->rValue = model->B4SOIcfrcoeff;
            return(OK);
        case B4SOI_MOD_EGIDL:
            value->rValue = model->B4SOIegidl;
            return(OK);
        case B4SOI_MOD_AGIDL:
            value->rValue = model->B4SOIagidl;
            return(OK);
        case B4SOI_MOD_BGIDL:
            value->rValue = model->B4SOIbgidl;
            return(OK);
        case B4SOI_MOD_CGIDL:
            value->rValue = model->B4SOIcgidl;
            return(OK);
        case B4SOI_MOD_RGIDL:
            value->rValue = model->B4SOIrgidl;
            return(OK);
        case B4SOI_MOD_KGIDL:
            value->rValue = model->B4SOIkgidl;
            return(OK);
        case B4SOI_MOD_FGIDL:
            value->rValue = model->B4SOIfgidl;
            return(OK);
                        
                 case B4SOI_MOD_EGISL:
            value->rValue = model->B4SOIegisl;
            return(OK);
        case B4SOI_MOD_AGISL:
            value->rValue = model->B4SOIagisl;
            return(OK);
        case B4SOI_MOD_BGISL:
            value->rValue = model->B4SOIbgisl;
            return(OK);
        case B4SOI_MOD_CGISL:
            value->rValue = model->B4SOIcgisl;
            return(OK);
        case B4SOI_MOD_RGISL:
            value->rValue = model->B4SOIrgisl;
            return(OK);
        case B4SOI_MOD_KGISL:
            value->rValue = model->B4SOIkgisl;
            return(OK);
        case B4SOI_MOD_FGISL:
            value->rValue = model->B4SOIfgisl;
            return(OK);        
                        
        case B4SOI_MOD_FDMOD:
            value->rValue = model->B4SOIfdMod;
            return(OK);
        case B4SOI_MOD_VSCE:
            value->rValue = model->B4SOIvsce;
            return(OK);
        case B4SOI_MOD_CDSBS:
            value->rValue = model->B4SOIcdsbs;
            return(OK);        

        case B4SOI_MOD_MINVCV:
            value->rValue = model->B4SOIminvcv;
            return(OK);
        case B4SOI_MOD_LMINVCV:
            value->rValue = model->B4SOIlminvcv;
            return(OK);
        case B4SOI_MOD_WMINVCV:
            value->rValue = model->B4SOIwminvcv;
            return(OK);
        case B4SOI_MOD_PMINVCV:
            value->rValue = model->B4SOIpminvcv;
            return(OK);                        
        case B4SOI_MOD_VOFFCV:
            value->rValue = model->B4SOIvoffcv;
            return(OK);
        case B4SOI_MOD_LVOFFCV:
            value->rValue = model->B4SOIlvoffcv;                              
            return(OK);
        case B4SOI_MOD_WVOFFCV:
            value->rValue = model->B4SOIwvoffcv;                              
            return(OK); 
        case B4SOI_MOD_PVOFFCV:
            value->rValue = model->B4SOIpvoffcv;
            return(OK);                        
/* v3.0 */
        case B4SOI_MOD_SOIMOD:
            value->iValue = model->B4SOIsoiMod;
            return(OK); /* v3.2 bug fix */
        case B4SOI_MOD_VBS0PD:
            value->rValue = model->B4SOIvbs0pd;
            return(OK); /* v3.2 */
        case B4SOI_MOD_VBS0FD:
            value->rValue = model->B4SOIvbs0fd;
            return(OK); /* v3.2 */
        case B4SOI_MOD_VBSA:
            value->rValue = model->B4SOIvbsa;
            return(OK);
        case B4SOI_MOD_NOFFFD:
            value->rValue = model->B4SOInofffd;
            return(OK);
        case B4SOI_MOD_VOFFFD:
            value->rValue = model->B4SOIvofffd;
            return(OK);
        case B4SOI_MOD_K1B:
            value->rValue = model->B4SOIk1b;
            return(OK);
        case B4SOI_MOD_K2B:
            value->rValue = model->B4SOIk2b;
            return(OK);
        case B4SOI_MOD_DK2B:
            value->rValue = model->B4SOIdk2b;
            return(OK);
        case B4SOI_MOD_DVBD0:
            value->rValue = model->B4SOIdvbd0;
            return(OK);
        case B4SOI_MOD_DVBD1:
            value->rValue = model->B4SOIdvbd1;
            return(OK);
        case B4SOI_MOD_MOINFD:
            value->rValue = model->B4SOImoinFD;
            return(OK);



/* v2.2 release */
        case B4SOI_MOD_WTH0:
            value->rValue = model->B4SOIwth0;
            return(OK);
        case B4SOI_MOD_RHALO:
            value->rValue = model->B4SOIrhalo;
            return(OK);
        case B4SOI_MOD_NTOX:
            value->rValue = model->B4SOIntox;
            return(OK);
        case B4SOI_MOD_TOXREF:
            value->rValue = model->B4SOItoxref;
            return(OK);
        case B4SOI_MOD_EBG:
            value->rValue = model->B4SOIebg;
            return(OK);
        case B4SOI_MOD_VEVB:
            value->rValue = model->B4SOIvevb;
            return(OK);
        case B4SOI_MOD_ALPHAGB1:
            value->rValue = model->B4SOIalphaGB1;
            return(OK);
        case B4SOI_MOD_BETAGB1:
            value->rValue = model->B4SOIbetaGB1;
            return(OK);
        case B4SOI_MOD_VGB1:
            value->rValue = model->B4SOIvgb1;
            return(OK);
        case B4SOI_MOD_VECB:
            value->rValue = model->B4SOIvecb;
            return(OK);
        case B4SOI_MOD_ALPHAGB2:
            value->rValue = model->B4SOIalphaGB2;
            return(OK);
        case B4SOI_MOD_BETAGB2:
            value->rValue = model->B4SOIbetaGB2;
            return(OK);
        case B4SOI_MOD_VGB2:
            value->rValue = model->B4SOIvgb2;
            return(OK);
        case B4SOI_MOD_AIGBCP2:
            value->rValue = model->B4SOIaigbcp2;
            return(OK);
        case B4SOI_MOD_BIGBCP2:
            value->rValue = model->B4SOIbigbcp2;
            return(OK);
        case B4SOI_MOD_CIGBCP2:
            value->rValue = model->B4SOIcigbcp2;
            return(OK);
        case B4SOI_MOD_TOXQM:
            value->rValue = model->B4SOItoxqm;
            return(OK);
        case B4SOI_MOD_VOXH:
            value->rValue = model->B4SOIvoxh;
            return(OK);
        case B4SOI_MOD_DELTAVOX:
            value->rValue = model->B4SOIdeltavox;
            return(OK);
/* v4.0 */
        case B4SOI_MOD_RDSMOD :
            value->iValue = model->B4SOIrdsMod;
            return(OK);
        case B4SOI_MOD_RBODYMOD :
            value->iValue = model->B4SOIrbodyMod;
            return(OK);
        case B4SOI_MOD_GBMIN:
            value->rValue = model->B4SOIgbmin;
            return(OK);
        case B4SOI_MOD_RBDB:
            value->rValue = model->B4SOIrbdb;
            return(OK);
        case B4SOI_MOD_RBSB:
            value->rValue = model->B4SOIrbsb;
            return(OK);
        case B4SOI_MOD_FRBODY:
            value->rValue = model->B4SOIfrbody;
            return(OK);
        case  B4SOI_MOD_DVTP0:
          value->rValue = model->B4SOIdvtp0;
            return(OK);
        case  B4SOI_MOD_DVTP1:
          value->rValue = model->B4SOIdvtp1;
            return(OK);
        case  B4SOI_MOD_DVTP2:
          value->rValue = model->B4SOIdvtp2;
            return(OK);
        case  B4SOI_MOD_DVTP3:
          value->rValue = model->B4SOIdvtp3;
            return(OK);
        case  B4SOI_MOD_DVTP4:
          value->rValue = model->B4SOIdvtp4;
            return(OK);
        case  B4SOI_MOD_LDVTP0:
          value->rValue = model->B4SOIldvtp0;
            return(OK);
        case  B4SOI_MOD_LDVTP1:
          value->rValue = model->B4SOIldvtp1;
            return(OK);
        case  B4SOI_MOD_LDVTP2:
          value->rValue = model->B4SOIldvtp2;
            return(OK);
        case  B4SOI_MOD_LDVTP3:
          value->rValue = model->B4SOIldvtp3;
            return(OK);
        case  B4SOI_MOD_LDVTP4:
          value->rValue = model->B4SOIldvtp4;
            return(OK);
        case  B4SOI_MOD_WDVTP0:
          value->rValue = model->B4SOIwdvtp0;
            return(OK);
        case  B4SOI_MOD_WDVTP1:
          value->rValue = model->B4SOIwdvtp1;
            return(OK);
        case  B4SOI_MOD_WDVTP2:
          value->rValue = model->B4SOIwdvtp2;
            return(OK);
        case  B4SOI_MOD_WDVTP3:
          value->rValue = model->B4SOIwdvtp3;
            return(OK);
        case  B4SOI_MOD_WDVTP4:
          value->rValue = model->B4SOIwdvtp4;
            return(OK);
        case  B4SOI_MOD_PDVTP0:
          value->rValue = model->B4SOIpdvtp0;
            return(OK);
        case  B4SOI_MOD_PDVTP1:
          value->rValue = model->B4SOIpdvtp1;
            return(OK);
        case  B4SOI_MOD_PDVTP2:
          value->rValue = model->B4SOIpdvtp2;
            return(OK);
        case  B4SOI_MOD_PDVTP3:
          value->rValue = model->B4SOIpdvtp3;
            return(OK);
        case  B4SOI_MOD_PDVTP4:
          value->rValue = model->B4SOIpdvtp4;
            return(OK);
        case B4SOI_MOD_MINV:
            value->rValue = model->B4SOIminv;
            return(OK);
        case B4SOI_MOD_LMINV:
            value->rValue = model->B4SOIlminv;
            return(OK);
        case B4SOI_MOD_WMINV:
            value->rValue = model->B4SOIwminv;
            return(OK);
        case B4SOI_MOD_PMINV:
            value->rValue = model->B4SOIpminv;
            return(OK);
        case B4SOI_MOD_FPROUT:
            value->rValue = model->B4SOIfprout;
            return(OK);
        case B4SOI_MOD_PDITS:
            value->rValue = model->B4SOIpdits;
            return(OK);
        case B4SOI_MOD_PDITSD:
            value->rValue = model->B4SOIpditsd;
            return(OK);
        case B4SOI_MOD_PDITSL:
            value->rValue = model->B4SOIpditsl;
            return(OK);
        case B4SOI_MOD_LFPROUT:
            value->rValue = model->B4SOIlfprout;
            return(OK);
        case B4SOI_MOD_LPDITS:
            value->rValue = model->B4SOIlpdits;
            return(OK);
        case B4SOI_MOD_LPDITSD:
            value->rValue = model->B4SOIlpditsd;
            return(OK);
        case B4SOI_MOD_WFPROUT:
            value->rValue = model->B4SOIwfprout;
            return(OK);
        case B4SOI_MOD_WPDITS:
            value->rValue = model->B4SOIwpdits;
            return(OK);
        case B4SOI_MOD_WPDITSD:
            value->rValue = model->B4SOIwpditsd;
            return(OK);
        case B4SOI_MOD_PFPROUT:
            value->rValue = model->B4SOIpfprout;
            return(OK);
        case B4SOI_MOD_PPDITS:
            value->rValue = model->B4SOIppdits;
            return(OK);
        case B4SOI_MOD_PPDITSD:
            value->rValue = model->B4SOIppditsd;
            return(OK);

/* v4.0 end */

/* v3.2 */
        case B4SOI_MOD_FNOIMOD :
            value->iValue = model->B4SOIfnoiMod;
            return(OK);
        case B4SOI_MOD_TNOIMOD :
            value->iValue = model->B4SOItnoiMod;
            return(OK);
        case B4SOI_MOD_TNOIA:
            value->rValue = model->B4SOItnoia;
            return(OK);
        case B4SOI_MOD_TNOIB:
            value->rValue = model->B4SOItnoib;
            return(OK);
        case B4SOI_MOD_RNOIA:
            value->rValue = model->B4SOIrnoia;
            return(OK);
        case B4SOI_MOD_RNOIB:
            value->rValue = model->B4SOIrnoib;
            return(OK);
        case B4SOI_MOD_NTNOI:
            value->rValue = model->B4SOIntnoi;
            return(OK);
/* v3.2 */

/* v3.1 added for RF */
        case B4SOI_MOD_RGATEMOD :
            value->iValue = model->B4SOIrgateMod;
            return(OK);
        case B4SOI_MOD_XRCRG1:
            value->rValue = model->B4SOIxrcrg1;
            return(OK);
        case B4SOI_MOD_XRCRG2:
            value->rValue = model->B4SOIxrcrg2;
            return(OK);
        case B4SOI_MOD_RSHG:
            value->rValue = model->B4SOIrshg;
            return(OK);
        case B4SOI_MOD_NGCON:
            value->rValue = model->B4SOIngcon;
            return(OK);
        case B4SOI_MOD_XGW:
            value->rValue = model->B4SOIxgw;
            return(OK);
        case B4SOI_MOD_XGL:
            value->rValue = model->B4SOIxgl;
            return(OK);
/* v3.1 added for RF end */
/*4.1*/

        case B4SOI_MOD_MTRLMOD :
            value->iValue = model->B4SOImtrlMod;
            return(OK);
                case B4SOI_MOD_VGSTCVMOD:
            value->iValue = model->B4SOIvgstcvMod;
            return(OK);
        case B4SOI_MOD_GIDLMOD :
            value->iValue = model->B4SOIgidlMod;
            return(OK);
           case B4SOI_MOD_IIIMOD :
            value->iValue = model->B4SOIiiiMod;
            return(OK);        
/* v3.0 */
        case B4SOI_MOD_IGBMOD:
            value->iValue = model->B4SOIigbMod;
            return(OK);
        case B4SOI_MOD_IGCMOD:
            value->iValue = model->B4SOIigcMod;
            return(OK);
        case B4SOI_MOD_AIGC:
            value->rValue = model->B4SOIaigc;
            return(OK);
        case B4SOI_MOD_BIGC:
            value->rValue = model->B4SOIbigc;
            return(OK);
        case B4SOI_MOD_CIGC:
            value->rValue = model->B4SOIcigc;
            return(OK);
        case B4SOI_MOD_AIGSD:
            value->rValue = model->B4SOIaigsd;
            return(OK);
        case B4SOI_MOD_BIGSD:
            value->rValue = model->B4SOIbigsd;
            return(OK);
        case B4SOI_MOD_CIGSD:
            value->rValue = model->B4SOIcigsd;
            return(OK);
        case B4SOI_MOD_NIGC:
            value->rValue = model->B4SOInigc;
            return(OK);
        case B4SOI_MOD_PIGCD:
            value->rValue = model->B4SOIpigcd;
            return(OK);
        case B4SOI_MOD_POXEDGE:
            value->rValue = model->B4SOIpoxedge;
            return(OK);
        case B4SOI_MOD_DLCIG:
            value->rValue = model->B4SOIdlcig;
            return(OK);



/* Added for binning - START */
        /* Length Dependence */
/* v3.1 */
        case B4SOI_MOD_LXJ:
            value->rValue = model->B4SOIlxj;
            return(OK);
        case B4SOI_MOD_LALPHAGB1:
            value->rValue = model->B4SOIlalphaGB1;
            return(OK);
        case B4SOI_MOD_LALPHAGB2:
            value->rValue = model->B4SOIlalphaGB2;
            return(OK);
        case B4SOI_MOD_LBETAGB1:
            value->rValue = model->B4SOIlbetaGB1;
            return(OK);
        case B4SOI_MOD_LBETAGB2:
            value->rValue = model->B4SOIlbetaGB2;
            return(OK);
        case B4SOI_MOD_LAIGBCP2:
            value->rValue = model->B4SOIlaigbcp2;
            return(OK);
        case B4SOI_MOD_LBIGBCP2:
            value->rValue = model->B4SOIlbigbcp2;
            return(OK);
        case B4SOI_MOD_LCIGBCP2:
            value->rValue = model->B4SOIlcigbcp2;
            return(OK);
        case B4SOI_MOD_LNDIF:
            value->rValue = model->B4SOIlndif;
            return(OK);
        case B4SOI_MOD_LNTRECF:
            value->rValue = model->B4SOIlntrecf;
            return(OK);
        case B4SOI_MOD_LNTRECR:
            value->rValue = model->B4SOIlntrecr;
            return(OK);
        case B4SOI_MOD_LXBJT:
            value->rValue = model->B4SOIlxbjt;
            return(OK);
        case B4SOI_MOD_LXDIFS:
            value->rValue = model->B4SOIlxdif;
            return(OK);
        case B4SOI_MOD_LXRECS:
            value->rValue = model->B4SOIlxrec;
            return(OK);
        case B4SOI_MOD_LXTUNS:
            value->rValue = model->B4SOIlxtun;
            return(OK);
        case B4SOI_MOD_LXDIFD:
            value->rValue = model->B4SOIlxdifd;
            return(OK);
        case B4SOI_MOD_LXRECD:
            value->rValue = model->B4SOIlxrecd;
            return(OK);
        case B4SOI_MOD_LXTUND:
            value->rValue = model->B4SOIlxtund;
            return(OK);
        case B4SOI_MOD_LCGDL:
            value->rValue = model->B4SOIlcgdl;
            return(OK);
        case B4SOI_MOD_LCGSL:
            value->rValue = model->B4SOIlcgsl;
            return(OK);
        case B4SOI_MOD_LCKAPPA:
            value->rValue = model->B4SOIlckappa;
            return(OK);
        case B4SOI_MOD_LUTE:
            value->rValue = model->B4SOIlute;
            return(OK);
        case B4SOI_MOD_LKT1:
            value->rValue = model->B4SOIlkt1;
            return(OK);
        case B4SOI_MOD_LKT2:
            value->rValue = model->B4SOIlkt2;
            return(OK);
        case B4SOI_MOD_LKT1L:
            value->rValue = model->B4SOIlkt1l;
            return(OK);
        case B4SOI_MOD_LUA1:
            value->rValue = model->B4SOIlua1;
            return(OK);
        case B4SOI_MOD_LUB1:
            value->rValue = model->B4SOIlub1;
            return(OK);
        case B4SOI_MOD_LUC1:
            value->rValue = model->B4SOIluc1;
            return(OK);
        case B4SOI_MOD_LAT:
            value->rValue = model->B4SOIlat;
            return(OK);
        case B4SOI_MOD_LPRT:
            value->rValue = model->B4SOIlprt;
            return(OK);


/* v3.0 */
        case B4SOI_MOD_LAIGC:
            value->rValue = model->B4SOIlaigc;
            return(OK);
        case B4SOI_MOD_LBIGC:
            value->rValue = model->B4SOIlbigc;
            return(OK);
        case B4SOI_MOD_LCIGC:
            value->rValue = model->B4SOIlcigc;
            return(OK);
        case B4SOI_MOD_LAIGSD:
            value->rValue = model->B4SOIlaigsd;
            return(OK);
        case B4SOI_MOD_LBIGSD:
            value->rValue = model->B4SOIlbigsd;
            return(OK);
        case B4SOI_MOD_LCIGSD:
            value->rValue = model->B4SOIlcigsd;
            return(OK);
        case B4SOI_MOD_LNIGC:
            value->rValue = model->B4SOIlnigc;
            return(OK);
        case B4SOI_MOD_LPIGCD:
            value->rValue = model->B4SOIlpigcd;
            return(OK);
        case B4SOI_MOD_LPOXEDGE:
            value->rValue = model->B4SOIlpoxedge;
            return(OK);

        case B4SOI_MOD_LNPEAK:
            value->rValue = model->B4SOIlnpeak;
            return(OK);
        case B4SOI_MOD_LNSUB:
            value->rValue = model->B4SOIlnsub;
            return(OK);
        case B4SOI_MOD_LNGATE:
            value->rValue = model->B4SOIlngate;
            return(OK);
        case B4SOI_MOD_LNSD:
            value->rValue = model->B4SOIlnsd;
            return(OK);                
        case B4SOI_MOD_LVTH0:
            value->rValue = model->B4SOIlvth0;
            return(OK);
        case B4SOI_MOD_LVFB: 
            value->rValue = model->B4SOIlvfb; 
            return(OK);  /* v4.1 */
        case  B4SOI_MOD_LK1:
          value->rValue = model->B4SOIlk1;
            return(OK);
        case  B4SOI_MOD_LK1W1:
          value->rValue = model->B4SOIlk1w1;
            return(OK);
        case  B4SOI_MOD_LK1W2:
          value->rValue = model->B4SOIlk1w2;
            return(OK);
        case  B4SOI_MOD_LK2:
          value->rValue = model->B4SOIlk2;
            return(OK);
        case  B4SOI_MOD_LK3:
          value->rValue = model->B4SOIlk3;
            return(OK);
        case  B4SOI_MOD_LK3B:
          value->rValue = model->B4SOIlk3b;
            return(OK);
        case  B4SOI_MOD_LKB1:
            value->rValue = model->B4SOIlkb1;
            return(OK);
        case  B4SOI_MOD_LW0:
          value->rValue = model->B4SOIlw0;
            return(OK);
        case  B4SOI_MOD_LLPE0:
          value->rValue = model->B4SOIllpe0;
            return(OK);
        case  B4SOI_MOD_LLPEB: /* v4.0 for Vth */
          value->rValue = model->B4SOIllpeb;
            return(OK);
        case  B4SOI_MOD_LDVT0 :
          value->rValue = model->B4SOIldvt0;
            return(OK);
        case  B4SOI_MOD_LDVT1 :
          value->rValue = model->B4SOIldvt1;
            return(OK);
        case  B4SOI_MOD_LDVT2 :
          value->rValue = model->B4SOIldvt2;
            return(OK);
        case  B4SOI_MOD_LDVT0W :
          value->rValue = model->B4SOIldvt0w;
            return(OK);
        case  B4SOI_MOD_LDVT1W :
          value->rValue = model->B4SOIldvt1w;
            return(OK);
        case  B4SOI_MOD_LDVT2W :
          value->rValue = model->B4SOIldvt2w;
            return(OK);
        case B4SOI_MOD_LU0:
            value->rValue = model->B4SOIlu0;
            return(OK);
        case B4SOI_MOD_LUA:
            value->rValue = model->B4SOIlua;
            return(OK);
        case B4SOI_MOD_LUB:
            value->rValue = model->B4SOIlub;
            return(OK);
        case B4SOI_MOD_LUC:
            value->rValue = model->B4SOIluc;
            return(OK);
        case B4SOI_MOD_LVSAT:
            value->rValue = model->B4SOIlvsat;
            return(OK);
        case B4SOI_MOD_LA0:
            value->rValue = model->B4SOIla0;
            return(OK);
        case B4SOI_MOD_LAGS:
            value->rValue = model->B4SOIlags;
            return(OK);
        case B4SOI_MOD_LB0:
            value->rValue = model->B4SOIlb0;
            return(OK);
        case B4SOI_MOD_LB1:
            value->rValue = model->B4SOIlb1;
            return(OK);
        case B4SOI_MOD_LKETA:
            value->rValue = model->B4SOIlketa;
            return(OK);
        case B4SOI_MOD_LKETAS:
            value->rValue = model->B4SOIlketas;
            return(OK);
        case B4SOI_MOD_LA1:
            value->rValue = model->B4SOIla1;
            return(OK);
        case B4SOI_MOD_LA2:
            value->rValue = model->B4SOIla2;
            return(OK);
        case B4SOI_MOD_LRDSW:
            value->rValue = model->B4SOIlrdsw;
            return(OK);
        case B4SOI_MOD_LRDW:
            value->rValue = model->B4SOIlrdw;
            return(OK);
        case B4SOI_MOD_LRSW:
            value->rValue = model->B4SOIlrsw;
            return(OK);
        case B4SOI_MOD_LPRWB:
            value->rValue = model->B4SOIlprwb;
            return(OK);
        case B4SOI_MOD_LPRWG:
            value->rValue = model->B4SOIlprwg;
            return(OK);
        case B4SOI_MOD_LWR:
            value->rValue = model->B4SOIlwr;
            return(OK);
        case  B4SOI_MOD_LNFACTOR :
          value->rValue = model->B4SOIlnfactor;
            return(OK);
        case B4SOI_MOD_LDWG:
            value->rValue = model->B4SOIldwg;
            return(OK);
        case B4SOI_MOD_LDWB:
            value->rValue = model->B4SOIldwb;
            return(OK);
        case B4SOI_MOD_LVOFF:
            value->rValue = model->B4SOIlvoff;
            return(OK);
        case B4SOI_MOD_LETA0:
            value->rValue = model->B4SOIleta0;
            return(OK);
        case B4SOI_MOD_LETAB:
            value->rValue = model->B4SOIletab;
            return(OK);
        case  B4SOI_MOD_LDSUB :
          value->rValue = model->B4SOIldsub;
            return(OK);
        case  B4SOI_MOD_LCIT :
          value->rValue = model->B4SOIlcit;
            return(OK);
        case  B4SOI_MOD_LCDSC :
          value->rValue = model->B4SOIlcdsc;
            return(OK);
        case  B4SOI_MOD_LCDSCB :
          value->rValue = model->B4SOIlcdscb;
            return(OK);
        case  B4SOI_MOD_LCDSCD :
          value->rValue = model->B4SOIlcdscd;
            return(OK);
        case B4SOI_MOD_LPCLM:
            value->rValue = model->B4SOIlpclm;
            return(OK);
        case B4SOI_MOD_LPDIBL1:
            value->rValue = model->B4SOIlpdibl1;
            return(OK);
        case B4SOI_MOD_LPDIBL2:
            value->rValue = model->B4SOIlpdibl2;
            return(OK);
        case B4SOI_MOD_LPDIBLB:
            value->rValue = model->B4SOIlpdiblb;
            return(OK);
        case  B4SOI_MOD_LDROUT :
          value->rValue = model->B4SOIldrout;
            return(OK);
        case B4SOI_MOD_LPVAG:
            value->rValue = model->B4SOIlpvag;
            return(OK);
        case B4SOI_MOD_LDELTA:
            value->rValue = model->B4SOIldelta;
            return(OK);
        case B4SOI_MOD_LALPHA0:
            value->rValue = model->B4SOIlalpha0;
            return(OK);
        case B4SOI_MOD_LFBJTII:
            value->rValue = model->B4SOIlfbjtii;
            return(OK);
                        /*4.1 Iii model*/
                        case B4SOI_MOD_LEBJTII:
            value->rValue = model->B4SOIlebjtii;
            return(OK);
        case B4SOI_MOD_LCBJTII:
            value->rValue = model->B4SOIlcbjtii;
            return(OK);
        case B4SOI_MOD_LVBCI:
            value->rValue = model->B4SOIlvbci;
            return(OK);
        case B4SOI_MOD_LABJTII:
            value->rValue = model->B4SOIlabjtii;
            return(OK);
        case B4SOI_MOD_LMBJTII:
            value->rValue = model->B4SOIlmbjtii;
            return(OK);
        case B4SOI_MOD_LBETA0:
            value->rValue = model->B4SOIlbeta0;
            return(OK);
        case B4SOI_MOD_LBETA1:
            value->rValue = model->B4SOIlbeta1;
            return(OK);
        case B4SOI_MOD_LBETA2:
            value->rValue = model->B4SOIlbeta2;
            return(OK);
        case B4SOI_MOD_LVDSATII0:
            value->rValue = model->B4SOIlvdsatii0;
            return(OK);
        case B4SOI_MOD_LLII:
            value->rValue = model->B4SOIllii;
            return(OK);
        case B4SOI_MOD_LESATII:
            value->rValue = model->B4SOIlesatii;
            return(OK);
        case B4SOI_MOD_LSII0:
            value->rValue = model->B4SOIlsii0;
            return(OK);
        case B4SOI_MOD_LSII1:
            value->rValue = model->B4SOIlsii1;
            return(OK);
        case B4SOI_MOD_LSII2:
            value->rValue = model->B4SOIlsii2;
            return(OK);
        case B4SOI_MOD_LSIID:
            value->rValue = model->B4SOIlsiid;
            return(OK);
        case B4SOI_MOD_LAGIDL:
            value->rValue = model->B4SOIlagidl;
            return(OK);
        case B4SOI_MOD_LBGIDL:
            value->rValue = model->B4SOIlbgidl;
            return(OK);
        case B4SOI_MOD_LCGIDL:
            value->rValue = model->B4SOIlcgidl;
            return(OK);
        case B4SOI_MOD_LEGIDL:
            value->rValue = model->B4SOIlegidl;
            return(OK);
        case B4SOI_MOD_LRGIDL:
            value->rValue = model->B4SOIlrgidl;
            return(OK);
        case B4SOI_MOD_LKGIDL:
            value->rValue = model->B4SOIlkgidl;
            return(OK);
        case B4SOI_MOD_LFGIDL:
            value->rValue = model->B4SOIlfgidl;
            return(OK);
                        
                case B4SOI_MOD_LAGISL:
            value->rValue = model->B4SOIlagisl;
            return(OK);
        case B4SOI_MOD_LBGISL:
            value->rValue = model->B4SOIlbgisl;
            return(OK);
        case B4SOI_MOD_LCGISL:
            value->rValue = model->B4SOIlcgisl;
            return(OK);
        case B4SOI_MOD_LEGISL:
            value->rValue = model->B4SOIlegisl;
            return(OK);
        case B4SOI_MOD_LRGISL:
            value->rValue = model->B4SOIlrgisl;
            return(OK);
        case B4SOI_MOD_LKGISL:
            value->rValue = model->B4SOIlkgisl;
            return(OK);
        case B4SOI_MOD_LFGISL:
            value->rValue = model->B4SOIlfgisl;
            return(OK);        
        case B4SOI_MOD_LNTUNS:        /* v4.0 */
            value->rValue = model->B4SOIlntun;
            return(OK);
        case B4SOI_MOD_LNTUND:        /* v4.0 */
            value->rValue = model->B4SOIlntund;
            return(OK);
        case B4SOI_MOD_LNDIODES: /* v4.0 */
            value->rValue = model->B4SOIlndiode;
            return(OK);
        case B4SOI_MOD_LNDIODED: /* v4.0 */
            value->rValue = model->B4SOIlndioded;
            return(OK);
        case B4SOI_MOD_LNRECF0S:        /* v4.0 */
            value->rValue = model->B4SOIlnrecf0;
            return(OK);
        case B4SOI_MOD_LNRECF0D:        /* v4.0 */
            value->rValue = model->B4SOIlnrecf0d;
            return(OK);
        case B4SOI_MOD_LNRECR0S:        /* v4.0 */
            value->rValue = model->B4SOIlnrecr0;
            return(OK);
        case B4SOI_MOD_LNRECR0D:        /* v4.0 */
            value->rValue = model->B4SOIlnrecr0d;
            return(OK);
        case B4SOI_MOD_LISBJT:
            value->rValue = model->B4SOIlisbjt;
            return(OK);
        case B4SOI_MOD_LIDBJT:        /* v4.0 */
            value->rValue = model->B4SOIlidbjt;
            return(OK);
        case B4SOI_MOD_LISDIF:
            value->rValue = model->B4SOIlisdif;
            return(OK);
        case B4SOI_MOD_LIDDIF:        /* v4.0 */
            value->rValue = model->B4SOIliddif;
            return(OK);
        case B4SOI_MOD_LISREC:
            value->rValue = model->B4SOIlisrec;
            return(OK);
        case B4SOI_MOD_LIDREC:        /* v4.0 */
            value->rValue = model->B4SOIlidrec;
            return(OK);
        case B4SOI_MOD_LISTUN:
            value->rValue = model->B4SOIlistun;
            return(OK);
        case B4SOI_MOD_LIDTUN:        /* v4.0 */
            value->rValue = model->B4SOIlidtun;
            return(OK);
        case B4SOI_MOD_LVREC0S:  /* v4.0 */
            value->rValue = model->B4SOIlvrec0;
            return(OK);
        case B4SOI_MOD_LVREC0D:  /* v4.0 */
            value->rValue = model->B4SOIlvrec0d;
            return(OK);
        case B4SOI_MOD_LVTUN0S:  /* v4.0 */
            value->rValue = model->B4SOIlvtun0;
            return(OK);
        case B4SOI_MOD_LVTUN0D:  /* v4.0 */
            value->rValue = model->B4SOIlvtun0d;
            return(OK);
        case B4SOI_MOD_LNBJT:
            value->rValue = model->B4SOIlnbjt;
            return(OK);
        case B4SOI_MOD_LLBJT0:
            value->rValue = model->B4SOIllbjt0;
            return(OK);
        case B4SOI_MOD_LVABJT:
            value->rValue = model->B4SOIlvabjt;
            return(OK);
        case B4SOI_MOD_LAELY:
            value->rValue = model->B4SOIlaely;
            return(OK);
        case B4SOI_MOD_LAHLIS:        /* v4.0 */ 
            value->rValue = model->B4SOIlahli;
            return(OK);
        case B4SOI_MOD_LAHLID:        /* v4.0 */ 
            value->rValue = model->B4SOIlahlid;
            return(OK);
        /* CV Model */
        case B4SOI_MOD_LVSDFB:
            value->rValue = model->B4SOIlvsdfb;
            return(OK);
        case B4SOI_MOD_LVSDTH:
            value->rValue = model->B4SOIlvsdth;
            return(OK);
        case B4SOI_MOD_LDELVT:
            value->rValue = model->B4SOIldelvt;
            return(OK);
        case B4SOI_MOD_LACDE:
            value->rValue = model->B4SOIlacde;
            return(OK);
        case B4SOI_MOD_LMOIN:
            value->rValue = model->B4SOIlmoin;
            return(OK);
        case B4SOI_MOD_LNOFF:
            value->rValue = model->B4SOIlnoff;
            return(OK); /* v3.2 */

        /* Width Dependence */
/* v3.1 */
        case B4SOI_MOD_WXJ:
            value->rValue = model->B4SOIwxj;
            return(OK);
        case B4SOI_MOD_WALPHAGB1:
            value->rValue = model->B4SOIwalphaGB1;
            return(OK);
        case B4SOI_MOD_WALPHAGB2:
            value->rValue = model->B4SOIwalphaGB2;
            return(OK);
        case B4SOI_MOD_WBETAGB1:
            value->rValue = model->B4SOIwbetaGB1;
            return(OK);
        case B4SOI_MOD_WBETAGB2:
            value->rValue = model->B4SOIwbetaGB2;
            return(OK);
        case B4SOI_MOD_WAIGBCP2:
            value->rValue = model->B4SOIwaigbcp2;
            return(OK);
        case B4SOI_MOD_WBIGBCP2:
            value->rValue = model->B4SOIwbigbcp2;
            return(OK);
        case B4SOI_MOD_WCIGBCP2:
            value->rValue = model->B4SOIwcigbcp2;
            return(OK);
        case B4SOI_MOD_WNDIF:
            value->rValue = model->B4SOIwndif;
            return(OK);
        case B4SOI_MOD_WNTRECF:
            value->rValue = model->B4SOIwntrecf;
            return(OK);
        case B4SOI_MOD_WNTRECR:
            value->rValue = model->B4SOIwntrecr;
            return(OK);
        case B4SOI_MOD_WXBJT:
            value->rValue = model->B4SOIwxbjt;
            return(OK);
        case B4SOI_MOD_WXDIFS:
            value->rValue = model->B4SOIwxdif;
            return(OK);
        case B4SOI_MOD_WXRECS:
            value->rValue = model->B4SOIwxrec;
            return(OK);
        case B4SOI_MOD_WXTUNS:
            value->rValue = model->B4SOIwxtun;
            return(OK);
        case B4SOI_MOD_WXDIFD:
            value->rValue = model->B4SOIwxdifd;
            return(OK);
        case B4SOI_MOD_WXRECD:
            value->rValue = model->B4SOIwxrecd;
            return(OK);
        case B4SOI_MOD_WXTUND:
            value->rValue = model->B4SOIwxtund;
            return(OK);
        case B4SOI_MOD_WCGDL:
            value->rValue = model->B4SOIwcgdl;
            return(OK);
        case B4SOI_MOD_WCGSL:
            value->rValue = model->B4SOIwcgsl;
            return(OK);
        case B4SOI_MOD_WCKAPPA:
            value->rValue = model->B4SOIwckappa;
            return(OK);
        case B4SOI_MOD_WUTE:
            value->rValue = model->B4SOIwute;
            return(OK);
        case B4SOI_MOD_WKT1:
            value->rValue = model->B4SOIwkt1;
            return(OK);
        case B4SOI_MOD_WKT2:
            value->rValue = model->B4SOIwkt2;
            return(OK);
        case B4SOI_MOD_WKT1L:
            value->rValue = model->B4SOIwkt1l;
            return(OK);
        case B4SOI_MOD_WUA1:
            value->rValue = model->B4SOIwua1;
            return(OK);
        case B4SOI_MOD_WUB1:
            value->rValue = model->B4SOIwub1;
            return(OK);
        case B4SOI_MOD_WUC1:
            value->rValue = model->B4SOIwuc1;
            return(OK);
        case B4SOI_MOD_WAT:
            value->rValue = model->B4SOIwat;
            return(OK);
        case B4SOI_MOD_WPRT:
            value->rValue = model->B4SOIwprt;
            return(OK);

/* v3.0 */
        case B4SOI_MOD_WAIGC:
            value->rValue = model->B4SOIwaigc;
            return(OK);
        case B4SOI_MOD_WBIGC:
            value->rValue = model->B4SOIwbigc;
            return(OK);
        case B4SOI_MOD_WCIGC:
            value->rValue = model->B4SOIwcigc;
            return(OK);
        case B4SOI_MOD_WAIGSD:
            value->rValue = model->B4SOIwaigsd;
            return(OK);
        case B4SOI_MOD_WBIGSD:
            value->rValue = model->B4SOIwbigsd;
            return(OK);
        case B4SOI_MOD_WCIGSD:
            value->rValue = model->B4SOIwcigsd;
            return(OK);
        case B4SOI_MOD_WNIGC:
            value->rValue = model->B4SOIwnigc;
            return(OK);
        case B4SOI_MOD_WPIGCD:
            value->rValue = model->B4SOIwpigcd;
            return(OK);
        case B4SOI_MOD_WPOXEDGE:
            value->rValue = model->B4SOIwpoxedge;
            return(OK);

        case B4SOI_MOD_WNPEAK:
            value->rValue = model->B4SOIwnpeak;
            return(OK);
        case B4SOI_MOD_WNSUB:
            value->rValue = model->B4SOIwnsub;
            return(OK);
        case B4SOI_MOD_WNGATE:
            value->rValue = model->B4SOIwngate;
            return(OK);
        case B4SOI_MOD_WNSD:
            value->rValue = model->B4SOIwnsd;
            return(OK);
        case B4SOI_MOD_WVTH0:
            value->rValue = model->B4SOIwvth0;
            return(OK);
       case B4SOI_MOD_WVFB: 
            value->rValue = model->B4SOIwvfb; 
            return(OK);  /* v4.1 */
        case  B4SOI_MOD_WK1:
          value->rValue = model->B4SOIwk1;
            return(OK);
        case  B4SOI_MOD_WK1W1:
          value->rValue = model->B4SOIwk1w1;
            return(OK);
        case  B4SOI_MOD_WK1W2:
          value->rValue = model->B4SOIwk1w2;
            return(OK);
        case  B4SOI_MOD_WK2:
          value->rValue = model->B4SOIwk2;
            return(OK);
        case  B4SOI_MOD_WK3:
          value->rValue = model->B4SOIwk3;
            return(OK);
        case  B4SOI_MOD_WK3B:
          value->rValue = model->B4SOIwk3b;
            return(OK);
        case  B4SOI_MOD_WKB1:
            value->rValue = model->B4SOIwkb1;
            return(OK);
        case  B4SOI_MOD_WW0:
          value->rValue = model->B4SOIww0;
            return(OK);
        case  B4SOI_MOD_WLPE0:
          value->rValue = model->B4SOIwlpe0;
            return(OK);
        case  B4SOI_MOD_WLPEB: /* v4.0 for Vth */
          value->rValue = model->B4SOIwlpeb;
            return(OK);
        case  B4SOI_MOD_WDVT0 :
          value->rValue = model->B4SOIwdvt0;
            return(OK);
        case  B4SOI_MOD_WDVT1 :
          value->rValue = model->B4SOIwdvt1;
            return(OK);
        case  B4SOI_MOD_WDVT2 :
          value->rValue = model->B4SOIwdvt2;
            return(OK);
        case  B4SOI_MOD_WDVT0W :
          value->rValue = model->B4SOIwdvt0w;
            return(OK);
        case  B4SOI_MOD_WDVT1W :
          value->rValue = model->B4SOIwdvt1w;
            return(OK);
        case  B4SOI_MOD_WDVT2W :
          value->rValue = model->B4SOIwdvt2w;
            return(OK);
        case B4SOI_MOD_WU0:
            value->rValue = model->B4SOIwu0;
            return(OK);
        case B4SOI_MOD_WUA:
            value->rValue = model->B4SOIwua;
            return(OK);
        case B4SOI_MOD_WUB:
            value->rValue = model->B4SOIwub;
            return(OK);
        case B4SOI_MOD_WUC:
            value->rValue = model->B4SOIwuc;
            return(OK);
        case B4SOI_MOD_WVSAT:
            value->rValue = model->B4SOIwvsat;
            return(OK);
        case B4SOI_MOD_WA0:
            value->rValue = model->B4SOIwa0;
            return(OK);
        case B4SOI_MOD_WAGS:
            value->rValue = model->B4SOIwags;
            return(OK);
        case B4SOI_MOD_WB0:
            value->rValue = model->B4SOIwb0;
            return(OK);
        case B4SOI_MOD_WB1:
            value->rValue = model->B4SOIwb1;
            return(OK);
        case B4SOI_MOD_WKETA:
            value->rValue = model->B4SOIwketa;
            return(OK);
        case B4SOI_MOD_WKETAS:
            value->rValue = model->B4SOIwketas;
            return(OK);
        case B4SOI_MOD_WA1:
            value->rValue = model->B4SOIwa1;
            return(OK);
        case B4SOI_MOD_WA2:
            value->rValue = model->B4SOIwa2;
            return(OK);
        case B4SOI_MOD_WRDSW:
            value->rValue = model->B4SOIwrdsw;
            return(OK);
        case B4SOI_MOD_WRDW:
            value->rValue = model->B4SOIwrdw;
            return(OK);
        case B4SOI_MOD_WRSW:
            value->rValue = model->B4SOIwrsw;
            return(OK);
        case B4SOI_MOD_WPRWB:
            value->rValue = model->B4SOIwprwb;
            return(OK);
        case B4SOI_MOD_WPRWG:
            value->rValue = model->B4SOIwprwg;
            return(OK);
        case B4SOI_MOD_WWR:
            value->rValue = model->B4SOIwwr;
            return(OK);
        case  B4SOI_MOD_WNFACTOR :
          value->rValue = model->B4SOIwnfactor;
            return(OK);
        case B4SOI_MOD_WDWG:
            value->rValue = model->B4SOIwdwg;
            return(OK);
        case B4SOI_MOD_WDWB:
            value->rValue = model->B4SOIwdwb;
            return(OK);
        case B4SOI_MOD_WVOFF:
            value->rValue = model->B4SOIwvoff;
            return(OK);
        case B4SOI_MOD_WETA0:
            value->rValue = model->B4SOIweta0;
            return(OK);
        case B4SOI_MOD_WETAB:
            value->rValue = model->B4SOIwetab;
            return(OK);
        case  B4SOI_MOD_WDSUB :
          value->rValue = model->B4SOIwdsub;
            return(OK);
        case  B4SOI_MOD_WCIT :
          value->rValue = model->B4SOIwcit;
            return(OK);
        case  B4SOI_MOD_WCDSC :
          value->rValue = model->B4SOIwcdsc;
            return(OK);
        case  B4SOI_MOD_WCDSCB :
          value->rValue = model->B4SOIwcdscb;
            return(OK);
        case  B4SOI_MOD_WCDSCD :
          value->rValue = model->B4SOIwcdscd;
            return(OK);
        case B4SOI_MOD_WPCLM:
            value->rValue = model->B4SOIwpclm;
            return(OK);
        case B4SOI_MOD_WPDIBL1:
            value->rValue = model->B4SOIwpdibl1;
            return(OK);
        case B4SOI_MOD_WPDIBL2:
            value->rValue = model->B4SOIwpdibl2;
            return(OK);
        case B4SOI_MOD_WPDIBLB:
            value->rValue = model->B4SOIwpdiblb;
            return(OK);
        case  B4SOI_MOD_WDROUT :
          value->rValue = model->B4SOIwdrout;
            return(OK);
        case B4SOI_MOD_WPVAG:
            value->rValue = model->B4SOIwpvag;
            return(OK);
        case B4SOI_MOD_WDELTA:
            value->rValue = model->B4SOIwdelta;
            return(OK);
        case B4SOI_MOD_WALPHA0:
            value->rValue = model->B4SOIwalpha0;
            return(OK);
        case B4SOI_MOD_WFBJTII:
            value->rValue = model->B4SOIwfbjtii;
            return(OK);
        /*4.1 Iii model*/
         case B4SOI_MOD_WEBJTII:
            value->rValue = model->B4SOIwebjtii;
            return(OK);
        case B4SOI_MOD_WCBJTII:
            value->rValue = model->B4SOIwcbjtii;
            return(OK);
        case B4SOI_MOD_WVBCI:
            value->rValue = model->B4SOIwvbci;
            return(OK);
        case B4SOI_MOD_WABJTII:
            value->rValue = model->B4SOIwabjtii;
            return(OK);
        case B4SOI_MOD_WMBJTII:
            value->rValue = model->B4SOIwmbjtii;
            return(OK);
        case B4SOI_MOD_WBETA0:
            value->rValue = model->B4SOIwbeta0;
            return(OK);
        case B4SOI_MOD_WBETA1:
            value->rValue = model->B4SOIwbeta1;
            return(OK);
        case B4SOI_MOD_WBETA2:
            value->rValue = model->B4SOIwbeta2;
            return(OK);
        case B4SOI_MOD_WVDSATII0:
            value->rValue = model->B4SOIwvdsatii0;
            return(OK);
        case B4SOI_MOD_WLII:
            value->rValue = model->B4SOIwlii;
            return(OK);
        case B4SOI_MOD_WESATII:
            value->rValue = model->B4SOIwesatii;
            return(OK);
        case B4SOI_MOD_WSII0:
            value->rValue = model->B4SOIwsii0;
            return(OK);
        case B4SOI_MOD_WSII1:
            value->rValue = model->B4SOIwsii1;
            return(OK);
        case B4SOI_MOD_WSII2:
            value->rValue = model->B4SOIwsii2;
            return(OK);
        case B4SOI_MOD_WSIID:
            value->rValue = model->B4SOIwsiid;
            return(OK);
        case B4SOI_MOD_WAGIDL:
            value->rValue = model->B4SOIwagidl;
            return(OK);
        case B4SOI_MOD_WBGIDL:
            value->rValue = model->B4SOIwbgidl;
            return(OK);
        case B4SOI_MOD_WCGIDL:
            value->rValue = model->B4SOIwcgidl;
            return(OK);
        case B4SOI_MOD_WEGIDL:
            value->rValue = model->B4SOIwegidl;
            return(OK);
        case B4SOI_MOD_WRGIDL:
            value->rValue = model->B4SOIwrgidl;
            return(OK);        
        case B4SOI_MOD_WKGIDL:
            value->rValue = model->B4SOIwkgidl;
            return(OK);
        case B4SOI_MOD_WFGIDL:
            value->rValue = model->B4SOIwfgidl;
            return(OK);
                
                case B4SOI_MOD_WAGISL:
            value->rValue = model->B4SOIwagisl;
            return(OK);
                        
        case B4SOI_MOD_WBGISL:
            value->rValue = model->B4SOIwbgisl;
            return(OK);
                        
        case B4SOI_MOD_WCGISL:
            value->rValue = model->B4SOIwcgisl;
            return(OK);
                        
        case B4SOI_MOD_WEGISL:
            value->rValue = model->B4SOIwegisl;
            return(OK);
                        
        case B4SOI_MOD_WRGISL:
            value->rValue = model->B4SOIwrgisl;
            return(OK);        
                        
        case B4SOI_MOD_WKGISL:
            value->rValue = model->B4SOIwkgisl;
            return(OK);
                        
        case B4SOI_MOD_WFGISL:
            value->rValue = model->B4SOIwfgisl;
            return(OK);
                        
                        
        case B4SOI_MOD_WNTUNS:        /* v4.0 */
            value->rValue = model->B4SOIwntun;
            return(OK);
        case B4SOI_MOD_WNTUND:        /* v4.0 */
            value->rValue = model->B4SOIwntund;
            return(OK);
        case B4SOI_MOD_WNDIODES: /* v4.0 */
            value->rValue = model->B4SOIwndiode;
            return(OK);
        case B4SOI_MOD_WNDIODED: /* v4.0 */
            value->rValue = model->B4SOIwndioded;
            return(OK);
        case B4SOI_MOD_WNRECF0S:        /* v4.0 */
            value->rValue = model->B4SOIwnrecf0;
            return(OK);
        case B4SOI_MOD_WNRECF0D:        /* v4.0 */
            value->rValue = model->B4SOIwnrecf0d;
            return(OK);
        case B4SOI_MOD_WNRECR0S:        /* v4.0 */
            value->rValue = model->B4SOIwnrecr0;
            return(OK);
        case B4SOI_MOD_WNRECR0D:        /* v4.0 */
            value->rValue = model->B4SOIwnrecr0d;
            return(OK);
        case B4SOI_MOD_WISBJT:
            value->rValue = model->B4SOIwisbjt;
            return(OK);
        case B4SOI_MOD_WIDBJT:        /* v4.0 */
            value->rValue = model->B4SOIwidbjt;
            return(OK);
        case B4SOI_MOD_WISDIF:
            value->rValue = model->B4SOIwisdif;
            return(OK);
        case B4SOI_MOD_WIDDIF:        /* v4.0 */
            value->rValue = model->B4SOIwiddif;
            return(OK);
        case B4SOI_MOD_WISREC:
            value->rValue = model->B4SOIwisrec;
            return(OK);
        case B4SOI_MOD_WIDREC:        /* v4.0 */
            value->rValue = model->B4SOIwidrec;
            return(OK);
        case B4SOI_MOD_WISTUN:
            value->rValue = model->B4SOIwistun;
            return(OK);
        case B4SOI_MOD_WIDTUN:        /* v4.0 */
            value->rValue = model->B4SOIwidtun;
            return(OK);
        case B4SOI_MOD_WVREC0S:  /* v4.0 */
            value->rValue = model->B4SOIwvrec0;
            return(OK);
        case B4SOI_MOD_WVREC0D:  /* v4.0 */
            value->rValue = model->B4SOIwvrec0d;
            return(OK);
        case B4SOI_MOD_WVTUN0S:  /* v4.0 */
            value->rValue = model->B4SOIwvtun0;
            return(OK);
        case B4SOI_MOD_WVTUN0D:  /* v4.0 */
            value->rValue = model->B4SOIwvtun0d;
            return(OK);
        case B4SOI_MOD_WNBJT:
            value->rValue = model->B4SOIwnbjt;
            return(OK);
        case B4SOI_MOD_WLBJT0:
            value->rValue = model->B4SOIwlbjt0;
            return(OK);
        case B4SOI_MOD_WVABJT:
            value->rValue = model->B4SOIwvabjt;
            return(OK);
        case B4SOI_MOD_WAELY:
            value->rValue = model->B4SOIwaely;
            return(OK);
        case B4SOI_MOD_WAHLIS:        /* v4.0 */ 
            value->rValue = model->B4SOIwahli;
            return(OK);
        case B4SOI_MOD_WAHLID:        /* v4.0 */ 
            value->rValue = model->B4SOIwahlid;
            return(OK);
        /* CV Model */
        case B4SOI_MOD_WVSDFB:
            value->rValue = model->B4SOIwvsdfb;
            return(OK);
        case B4SOI_MOD_WVSDTH:
            value->rValue = model->B4SOIwvsdth;
            return(OK);
        case B4SOI_MOD_WDELVT:
            value->rValue = model->B4SOIwdelvt;
            return(OK);
        case B4SOI_MOD_WACDE:
            value->rValue = model->B4SOIwacde;
            return(OK);
        case B4SOI_MOD_WMOIN:
            value->rValue = model->B4SOIwmoin;
            return(OK);
        case B4SOI_MOD_WNOFF:
            value->rValue = model->B4SOIwnoff;
            return(OK); /* v3.2 */

        /* Cross-term Dependence */
/* v3.1 */
        case B4SOI_MOD_PXJ:
            value->rValue = model->B4SOIpxj;
            return(OK);
        case B4SOI_MOD_PALPHAGB1:
            value->rValue = model->B4SOIpalphaGB1;
            return(OK);
        case B4SOI_MOD_PALPHAGB2:
            value->rValue = model->B4SOIpalphaGB2;
            return(OK);
        case B4SOI_MOD_PBETAGB1:
            value->rValue = model->B4SOIpbetaGB1;
            return(OK);
        case B4SOI_MOD_PBETAGB2:
            value->rValue = model->B4SOIpbetaGB2;
            return(OK);
        case B4SOI_MOD_PAIGBCP2:
            value->rValue = model->B4SOIpaigbcp2;
            return(OK);
        case B4SOI_MOD_PBIGBCP2:
            value->rValue = model->B4SOIpbigbcp2;
            return(OK);
        case B4SOI_MOD_PCIGBCP2:
            value->rValue = model->B4SOIpcigbcp2;
            return(OK);
        case B4SOI_MOD_PNDIF:
            value->rValue = model->B4SOIpndif;
            return(OK);
        case B4SOI_MOD_PNTRECF:
            value->rValue = model->B4SOIpntrecf;
            return(OK);
        case B4SOI_MOD_PNTRECR:
            value->rValue = model->B4SOIpntrecr;
            return(OK);
        case B4SOI_MOD_PXBJT:
            value->rValue = model->B4SOIpxbjt;
            return(OK);
        case B4SOI_MOD_PXDIFS:
            value->rValue = model->B4SOIpxdif;
            return(OK);
        case B4SOI_MOD_PXRECS:
            value->rValue = model->B4SOIpxrec;
            return(OK);
        case B4SOI_MOD_PXTUNS:
            value->rValue = model->B4SOIpxtun;
            return(OK);
        case B4SOI_MOD_PXDIFD:
            value->rValue = model->B4SOIpxdifd;
            return(OK);
        case B4SOI_MOD_PXRECD:
            value->rValue = model->B4SOIpxrecd;
            return(OK);
        case B4SOI_MOD_PXTUND:
            value->rValue = model->B4SOIpxtund;
            return(OK);
        case B4SOI_MOD_PCGDL:
            value->rValue = model->B4SOIpcgdl;
            return(OK);
        case B4SOI_MOD_PCGSL:
            value->rValue = model->B4SOIpcgsl;
            return(OK);
        case B4SOI_MOD_PCKAPPA:
            value->rValue = model->B4SOIpckappa;
            return(OK);
        case B4SOI_MOD_PUTE:
            value->rValue = model->B4SOIpute;
            return(OK);
        case B4SOI_MOD_PKT1:
            value->rValue = model->B4SOIpkt1;
            return(OK);
        case B4SOI_MOD_PKT2:
            value->rValue = model->B4SOIpkt2;
            return(OK);
        case B4SOI_MOD_PKT1L:
            value->rValue = model->B4SOIpkt1l;
            return(OK);
        case B4SOI_MOD_PUA1:
            value->rValue = model->B4SOIpua1;
            return(OK);
        case B4SOI_MOD_PUB1:
            value->rValue = model->B4SOIpub1;
            return(OK);
        case B4SOI_MOD_PUC1:
            value->rValue = model->B4SOIpuc1;
            return(OK);
        case B4SOI_MOD_PAT:
            value->rValue = model->B4SOIpat;
            return(OK);
        case B4SOI_MOD_PPRT:
            value->rValue = model->B4SOIpprt;
            return(OK);


/* v3.0 */
        case B4SOI_MOD_PAIGC:
            value->rValue = model->B4SOIpaigc;
            return(OK);
        case B4SOI_MOD_PBIGC:
            value->rValue = model->B4SOIpbigc;
            return(OK);
        case B4SOI_MOD_PCIGC:
            value->rValue = model->B4SOIpcigc;
            return(OK);
        case B4SOI_MOD_PAIGSD:
            value->rValue = model->B4SOIpaigsd;
            return(OK);
        case B4SOI_MOD_PBIGSD:
            value->rValue = model->B4SOIpbigsd;
            return(OK);
        case B4SOI_MOD_PCIGSD:
            value->rValue = model->B4SOIpcigsd;
            return(OK);
        case B4SOI_MOD_PNIGC:
            value->rValue = model->B4SOIpnigc;
            return(OK);
        case B4SOI_MOD_PPIGCD:
            value->rValue = model->B4SOIppigcd;
            return(OK);
        case B4SOI_MOD_PPOXEDGE:
            value->rValue = model->B4SOIppoxedge;
            return(OK);

        case B4SOI_MOD_PNPEAK:
            value->rValue = model->B4SOIpnpeak;
            return(OK);
        case B4SOI_MOD_PNSUB:
            value->rValue = model->B4SOIpnsub;
            return(OK);
        case B4SOI_MOD_PNGATE:
            value->rValue = model->B4SOIpngate;
            return(OK);
        case B4SOI_MOD_PNSD:
            value->rValue = model->B4SOIpnsd;
            return(OK);
        case B4SOI_MOD_PVTH0:
            value->rValue = model->B4SOIpvth0;
            return(OK);
        case B4SOI_MOD_PVFB: 
            value->rValue = model->B4SOIpvfb; 
            return(OK); /* v4.1 */
        case  B4SOI_MOD_PK1:
          value->rValue = model->B4SOIpk1;
            return(OK);
        case  B4SOI_MOD_PK1W1:
          value->rValue = model->B4SOIpk1w1;
            return(OK);
        case  B4SOI_MOD_PK1W2:
          value->rValue = model->B4SOIpk1w2;
            return(OK);
        case  B4SOI_MOD_PK2:
          value->rValue = model->B4SOIpk2;
            return(OK);
        case  B4SOI_MOD_PK3:
          value->rValue = model->B4SOIpk3;
            return(OK);
        case  B4SOI_MOD_PK3B:
          value->rValue = model->B4SOIpk3b;
            return(OK);
        case  B4SOI_MOD_PKB1:
            value->rValue = model->B4SOIpkb1;
            return(OK);
        case  B4SOI_MOD_PW0:
          value->rValue = model->B4SOIpw0;
            return(OK);
        case  B4SOI_MOD_PLPE0:
          value->rValue = model->B4SOIplpe0;
            return(OK);
        case  B4SOI_MOD_PLPEB: /* v4.0 for Vth */
          value->rValue = model->B4SOIplpeb;
            return(OK);
        case  B4SOI_MOD_PDVT0 :
          value->rValue = model->B4SOIpdvt0;
            return(OK);
        case  B4SOI_MOD_PDVT1 :
          value->rValue = model->B4SOIpdvt1;
            return(OK);
        case  B4SOI_MOD_PDVT2 :
          value->rValue = model->B4SOIpdvt2;
            return(OK);
        case  B4SOI_MOD_PDVT0W :
          value->rValue = model->B4SOIpdvt0w;
            return(OK);
        case  B4SOI_MOD_PDVT1W :
          value->rValue = model->B4SOIpdvt1w;
            return(OK);
        case  B4SOI_MOD_PDVT2W :
          value->rValue = model->B4SOIpdvt2w;
            return(OK);
        case B4SOI_MOD_PU0:
            value->rValue = model->B4SOIpu0;
            return(OK);
        case B4SOI_MOD_PUA:
            value->rValue = model->B4SOIpua;
            return(OK);
        case B4SOI_MOD_PUB:
            value->rValue = model->B4SOIpub;
            return(OK);
        case B4SOI_MOD_PUC:
            value->rValue = model->B4SOIpuc;
            return(OK);
        case B4SOI_MOD_PVSAT:
            value->rValue = model->B4SOIpvsat;
            return(OK);
        case B4SOI_MOD_PA0:
            value->rValue = model->B4SOIpa0;
            return(OK);
        case B4SOI_MOD_PAGS:
            value->rValue = model->B4SOIpags;
            return(OK);
        case B4SOI_MOD_PB0:
            value->rValue = model->B4SOIpb0;
            return(OK);
        case B4SOI_MOD_PB1:
            value->rValue = model->B4SOIpb1;
            return(OK);
        case B4SOI_MOD_PKETA:
            value->rValue = model->B4SOIpketa;
            return(OK);
        case B4SOI_MOD_PKETAS:
            value->rValue = model->B4SOIpketas;
            return(OK);
        case B4SOI_MOD_PA1:
            value->rValue = model->B4SOIpa1;
            return(OK);
        case B4SOI_MOD_PA2:
            value->rValue = model->B4SOIpa2;
            return(OK);
        case B4SOI_MOD_PRDSW:
            value->rValue = model->B4SOIprdsw;
            return(OK);
        case B4SOI_MOD_PRDW:
            value->rValue = model->B4SOIprdw;
            return(OK);
        case B4SOI_MOD_PRSW:
            value->rValue = model->B4SOIprsw;
            return(OK);
        case B4SOI_MOD_PPRWB:
            value->rValue = model->B4SOIpprwb;
            return(OK);
        case B4SOI_MOD_PPRWG:
            value->rValue = model->B4SOIpprwg;
            return(OK);
        case B4SOI_MOD_PWR:
            value->rValue = model->B4SOIpwr;
            return(OK);
        case  B4SOI_MOD_PNFACTOR :
          value->rValue = model->B4SOIpnfactor;
            return(OK);
        case B4SOI_MOD_PDWG:
            value->rValue = model->B4SOIpdwg;
            return(OK);
        case B4SOI_MOD_PDWB:
            value->rValue = model->B4SOIpdwb;
            return(OK);
        case B4SOI_MOD_PVOFF:
            value->rValue = model->B4SOIpvoff;
            return(OK);
        case B4SOI_MOD_PETA0:
            value->rValue = model->B4SOIpeta0;
            return(OK);
        case B4SOI_MOD_PETAB:
            value->rValue = model->B4SOIpetab;
            return(OK);
        case  B4SOI_MOD_PDSUB :
          value->rValue = model->B4SOIpdsub;
            return(OK);
        case  B4SOI_MOD_PCIT :
          value->rValue = model->B4SOIpcit;
            return(OK);
        case  B4SOI_MOD_PCDSC :
          value->rValue = model->B4SOIpcdsc;
            return(OK);
        case  B4SOI_MOD_PCDSCB :
          value->rValue = model->B4SOIpcdscb;
            return(OK);
        case  B4SOI_MOD_PCDSCD :
          value->rValue = model->B4SOIpcdscd;
            return(OK);
        case B4SOI_MOD_PPCLM:
            value->rValue = model->B4SOIppclm;
            return(OK);
        case B4SOI_MOD_PPDIBL1:
            value->rValue = model->B4SOIppdibl1;
            return(OK);
        case B4SOI_MOD_PPDIBL2:
            value->rValue = model->B4SOIppdibl2;
            return(OK);
        case B4SOI_MOD_PPDIBLB:
            value->rValue = model->B4SOIppdiblb;
            return(OK);
        case  B4SOI_MOD_PDROUT :
          value->rValue = model->B4SOIpdrout;
            return(OK);
        case B4SOI_MOD_PPVAG:
            value->rValue = model->B4SOIppvag;
            return(OK);
        case B4SOI_MOD_PDELTA:
            value->rValue = model->B4SOIpdelta;
            return(OK);
        case B4SOI_MOD_PALPHA0:
            value->rValue = model->B4SOIpalpha0;
            return(OK);
        case B4SOI_MOD_PFBJTII:
            value->rValue = model->B4SOIpfbjtii;
            return(OK);
        /*4.1 Iii model*/
            case B4SOI_MOD_PEBJTII:
            value->rValue = model->B4SOIpebjtii;
            return(OK);
        case B4SOI_MOD_PCBJTII:
            value->rValue = model->B4SOIpcbjtii;
            return(OK);
        case B4SOI_MOD_PVBCI:
            value->rValue = model->B4SOIpvbci;
            return(OK);
        case B4SOI_MOD_PABJTII:
            value->rValue = model->B4SOIpabjtii;
            return(OK);
        case B4SOI_MOD_PMBJTII:
            value->rValue = model->B4SOIpmbjtii;
            return(OK);
                        
        case B4SOI_MOD_PBETA0:
            value->rValue = model->B4SOIpbeta0;
            return(OK);
        case B4SOI_MOD_PBETA1:
            value->rValue = model->B4SOIpbeta1;
            return(OK);
        case B4SOI_MOD_PBETA2:
            value->rValue = model->B4SOIpbeta2;
            return(OK);
        case B4SOI_MOD_PVDSATII0:
            value->rValue = model->B4SOIpvdsatii0;
            return(OK);
        case B4SOI_MOD_PLII:
            value->rValue = model->B4SOIplii;
            return(OK);
        case B4SOI_MOD_PESATII:
            value->rValue = model->B4SOIpesatii;
            return(OK);
        case B4SOI_MOD_PSII0:
            value->rValue = model->B4SOIpsii0;
            return(OK);
        case B4SOI_MOD_PSII1:
            value->rValue = model->B4SOIpsii1;
            return(OK);
        case B4SOI_MOD_PSII2:
            value->rValue = model->B4SOIpsii2;
            return(OK);
        case B4SOI_MOD_PSIID:
            value->rValue = model->B4SOIpsiid;
            return(OK);
        case B4SOI_MOD_PAGIDL:
            value->rValue = model->B4SOIpagidl;
            return(OK);
        case B4SOI_MOD_PBGIDL:
            value->rValue = model->B4SOIpbgidl;
            return(OK);
        case B4SOI_MOD_PCGIDL:
            value->rValue = model->B4SOIpcgidl;
            return(OK);
        case B4SOI_MOD_PEGIDL:
            value->rValue = model->B4SOIpegidl;
            return(OK);
        case B4SOI_MOD_PRGIDL:
            value->rValue = model->B4SOIprgidl;
            return(OK);
        case B4SOI_MOD_PKGIDL:
            value->rValue = model->B4SOIpkgidl;
            return(OK);
        case B4SOI_MOD_PFGIDL:
            value->rValue = model->B4SOIpfgidl;
            return(OK);
                        
                case B4SOI_MOD_PAGISL:
            value->rValue = model->B4SOIpagisl;
            return(OK);
        case B4SOI_MOD_PBGISL:
            value->rValue = model->B4SOIpbgisl;
            return(OK);
        case B4SOI_MOD_PCGISL:
            value->rValue = model->B4SOIpcgisl;
            return(OK);
        case B4SOI_MOD_PEGISL:
            value->rValue = model->B4SOIpegisl;
            return(OK);
        case B4SOI_MOD_PRGISL:
            value->rValue = model->B4SOIprgisl;
            return(OK);
        case B4SOI_MOD_PKGISL:
            value->rValue = model->B4SOIpkgisl;
            return(OK);
        case B4SOI_MOD_PFGISL:
            value->rValue = model->B4SOIpfgisl;
            return(OK);        
                        
        case B4SOI_MOD_PNTUNS:                /* v4.0 */
            value->rValue = model->B4SOIpntun;
            return(OK);
        case B4SOI_MOD_PNTUND:                /* v4.0 */
            value->rValue = model->B4SOIpntund;
            return(OK);
        case B4SOI_MOD_PNDIODES:        /* v4.0 */
            value->rValue = model->B4SOIpndiode;
            return(OK);
        case B4SOI_MOD_PNDIODED:        /* v4.0 */
            value->rValue = model->B4SOIpndioded;
            return(OK);
        case B4SOI_MOD_PNRECF0S:        /* v4.0 */
            value->rValue = model->B4SOIpnrecf0;
            return(OK);
        case B4SOI_MOD_PNRECF0D:        /* v4.0 */
            value->rValue = model->B4SOIpnrecf0d;
            return(OK);
        case B4SOI_MOD_PNRECR0S:        /* v4.0 */
            value->rValue = model->B4SOIpnrecr0;
            return(OK);
        case B4SOI_MOD_PNRECR0D:        /* v4.0 */
            value->rValue = model->B4SOIpnrecr0d;
            return(OK);
        case B4SOI_MOD_PISBJT:
            value->rValue = model->B4SOIpisbjt;
            return(OK);
        case B4SOI_MOD_PIDBJT:        /* v4.0 */
            value->rValue = model->B4SOIpidbjt;
            return(OK);
        case B4SOI_MOD_PISDIF:
            value->rValue = model->B4SOIpisdif;
            return(OK);
        case B4SOI_MOD_PIDDIF:        /* v4.0 */
            value->rValue = model->B4SOIpiddif;
            return(OK);
        case B4SOI_MOD_PISREC:
            value->rValue = model->B4SOIpisrec;
            return(OK);
        case B4SOI_MOD_PIDREC: /* v4.0 */
            value->rValue = model->B4SOIpidrec;
            return(OK);
        case B4SOI_MOD_PISTUN:
            value->rValue = model->B4SOIpistun;
            return(OK);
        case B4SOI_MOD_PIDTUN:        /* v4.0 */ 
            value->rValue = model->B4SOIpidtun;
            return(OK);
        case B4SOI_MOD_PVREC0S:  /* v4.0 */
            value->rValue = model->B4SOIpvrec0;
            return(OK);
        case B4SOI_MOD_PVREC0D:  /* v4.0 */
            value->rValue = model->B4SOIpvrec0d;
            return(OK);
        case B4SOI_MOD_PVTUN0S:  /* v4.0 */
            value->rValue = model->B4SOIpvtun0;
            return(OK);
        case B4SOI_MOD_PVTUN0D:  /* v4.0 */
            value->rValue = model->B4SOIpvtun0d;
            return(OK);
        case B4SOI_MOD_PNBJT:
            value->rValue = model->B4SOIpnbjt;
            return(OK);
        case B4SOI_MOD_PLBJT0:
            value->rValue = model->B4SOIplbjt0;
            return(OK);
        case B4SOI_MOD_PVABJT:
            value->rValue = model->B4SOIpvabjt;
            return(OK);
        case B4SOI_MOD_PAELY:
            value->rValue = model->B4SOIpaely;
            return(OK);
        case B4SOI_MOD_PAHLIS:        /* v4.0 */
            value->rValue = model->B4SOIpahli;
            return(OK);
        case B4SOI_MOD_PAHLID:        /* v4.0 */
            value->rValue = model->B4SOIpahlid;
            return(OK);
        /* CV Model */
        case B4SOI_MOD_PVSDFB:
            value->rValue = model->B4SOIpvsdfb;
            return(OK);
        case B4SOI_MOD_PVSDTH:
            value->rValue = model->B4SOIpvsdth;
            return(OK);
        case B4SOI_MOD_PDELVT:
            value->rValue = model->B4SOIpdelvt;
            return(OK);
        case B4SOI_MOD_PACDE:
            value->rValue = model->B4SOIpacde;
            return(OK);
        case B4SOI_MOD_PMOIN:
            value->rValue = model->B4SOIpmoin;
            return(OK);
        case B4SOI_MOD_PNOFF:
            value->rValue = model->B4SOIpnoff;
            return(OK); /* v3.2 */
/* Added for binning - END */

        case B4SOI_MOD_VGS_MAX:
            value->rValue = model->B4SOIvgsMax;
            return(OK);
        case B4SOI_MOD_VGD_MAX:
            value->rValue = model->B4SOIvgdMax;
            return(OK);
        case B4SOI_MOD_VGB_MAX:
            value->rValue = model->B4SOIvgbMax;
            return(OK);
        case B4SOI_MOD_VDS_MAX:
            value->rValue = model->B4SOIvdsMax;
            return(OK);
        case B4SOI_MOD_VBS_MAX:
            value->rValue = model->B4SOIvbsMax;
            return(OK);
        case B4SOI_MOD_VBD_MAX:
            value->rValue = model->B4SOIvbdMax;
            return(OK);
        case B4SOI_MOD_VGSR_MAX:
            value->rValue = model->B4SOIvgsrMax;
            return(OK);
        case B4SOI_MOD_VGDR_MAX:
            value->rValue = model->B4SOIvgdrMax;
            return(OK);
        case B4SOI_MOD_VGBR_MAX:
            value->rValue = model->B4SOIvgbrMax;
            return(OK);
        case B4SOI_MOD_VBSR_MAX:
            value->rValue = model->B4SOIvbsrMax;
            return(OK);
        case B4SOI_MOD_VBDR_MAX:
            value->rValue = model->B4SOIvbdrMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



