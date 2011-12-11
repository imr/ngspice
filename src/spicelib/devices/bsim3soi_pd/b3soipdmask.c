/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdmask.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su 02/3/5
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIPDmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case B3SOIPD_MOD_MOBMOD:
            value->iValue = model->B3SOIPDmobMod; 
            return(OK);
        case B3SOIPD_MOD_PARAMCHK:
            value->iValue = model->B3SOIPDparamChk; 
            return(OK);
        case B3SOIPD_MOD_BINUNIT:
            value->iValue = model->B3SOIPDbinUnit; 
            return(OK);
        case B3SOIPD_MOD_CAPMOD:
            value->iValue = model->B3SOIPDcapMod; 
            return(OK);
        case B3SOIPD_MOD_SHMOD:
            value->iValue = model->B3SOIPDshMod; 
            return(OK);
        case B3SOIPD_MOD_NOIMOD:
            value->iValue = model->B3SOIPDnoiMod; 
            return(OK);
        case  B3SOIPD_MOD_VERSION :
          value->rValue = model->B3SOIPDversion;
            return(OK);
        case  B3SOIPD_MOD_TOX :
          value->rValue = model->B3SOIPDtox;
            return(OK);
/* v2.2.3 */
        case  B3SOIPD_MOD_DTOXCV :
          value->rValue = model->B3SOIPDdtoxcv;
            return(OK);

        case  B3SOIPD_MOD_CDSC :
          value->rValue = model->B3SOIPDcdsc;
            return(OK);
        case  B3SOIPD_MOD_CDSCB :
          value->rValue = model->B3SOIPDcdscb;
            return(OK);

        case  B3SOIPD_MOD_CDSCD :
          value->rValue = model->B3SOIPDcdscd;
            return(OK);

        case  B3SOIPD_MOD_CIT :
          value->rValue = model->B3SOIPDcit;
            return(OK);
        case  B3SOIPD_MOD_NFACTOR :
          value->rValue = model->B3SOIPDnfactor;
            return(OK);
        case B3SOIPD_MOD_VSAT:
            value->rValue = model->B3SOIPDvsat;
            return(OK);
        case B3SOIPD_MOD_AT:
            value->rValue = model->B3SOIPDat;
            return(OK);
        case B3SOIPD_MOD_A0:
            value->rValue = model->B3SOIPDa0;
            return(OK);

        case B3SOIPD_MOD_AGS:
            value->rValue = model->B3SOIPDags;
            return(OK);

        case B3SOIPD_MOD_A1:
            value->rValue = model->B3SOIPDa1;
            return(OK);
        case B3SOIPD_MOD_A2:
            value->rValue = model->B3SOIPDa2;
            return(OK);
        case B3SOIPD_MOD_KETA:
            value->rValue = model->B3SOIPDketa;
            return(OK);   
        case B3SOIPD_MOD_NSUB:
            value->rValue = model->B3SOIPDnsub;
            return(OK);
        case B3SOIPD_MOD_NPEAK:
            value->rValue = model->B3SOIPDnpeak;
            return(OK);
        case B3SOIPD_MOD_NGATE:
            value->rValue = model->B3SOIPDngate;
            return(OK);
        case B3SOIPD_MOD_GAMMA1:
            value->rValue = model->B3SOIPDgamma1;
            return(OK);
        case B3SOIPD_MOD_GAMMA2:
            value->rValue = model->B3SOIPDgamma2;
            return(OK);
        case B3SOIPD_MOD_VBX:
            value->rValue = model->B3SOIPDvbx;
            return(OK);
        case B3SOIPD_MOD_VBM:
            value->rValue = model->B3SOIPDvbm;
            return(OK);
        case B3SOIPD_MOD_XT:
            value->rValue = model->B3SOIPDxt;
            return(OK);
        case  B3SOIPD_MOD_K1:
          value->rValue = model->B3SOIPDk1;
            return(OK);
        case  B3SOIPD_MOD_KT1:
          value->rValue = model->B3SOIPDkt1;
            return(OK);
        case  B3SOIPD_MOD_KT1L:
          value->rValue = model->B3SOIPDkt1l;
            return(OK);
        case  B3SOIPD_MOD_KT2 :
          value->rValue = model->B3SOIPDkt2;
            return(OK);
        case  B3SOIPD_MOD_K2 :
          value->rValue = model->B3SOIPDk2;
            return(OK);
        case  B3SOIPD_MOD_K3:
          value->rValue = model->B3SOIPDk3;
            return(OK);
        case  B3SOIPD_MOD_K3B:
          value->rValue = model->B3SOIPDk3b;
            return(OK);
        case  B3SOIPD_MOD_W0:
          value->rValue = model->B3SOIPDw0;
            return(OK);
        case  B3SOIPD_MOD_NLX:
          value->rValue = model->B3SOIPDnlx;
            return(OK);
        case  B3SOIPD_MOD_DVT0 :                
          value->rValue = model->B3SOIPDdvt0;
            return(OK);
        case  B3SOIPD_MOD_DVT1 :             
          value->rValue = model->B3SOIPDdvt1;
            return(OK);
        case  B3SOIPD_MOD_DVT2 :             
          value->rValue = model->B3SOIPDdvt2;
            return(OK);
        case  B3SOIPD_MOD_DVT0W :                
          value->rValue = model->B3SOIPDdvt0w;
            return(OK);
        case  B3SOIPD_MOD_DVT1W :             
          value->rValue = model->B3SOIPDdvt1w;
            return(OK);
        case  B3SOIPD_MOD_DVT2W :             
          value->rValue = model->B3SOIPDdvt2w;
            return(OK);
        case  B3SOIPD_MOD_DROUT :           
          value->rValue = model->B3SOIPDdrout;
            return(OK);
        case  B3SOIPD_MOD_DSUB :           
          value->rValue = model->B3SOIPDdsub;
            return(OK);
        case B3SOIPD_MOD_VTH0:
            value->rValue = model->B3SOIPDvth0; 
            return(OK);
        case B3SOIPD_MOD_UA:
            value->rValue = model->B3SOIPDua; 
            return(OK);
        case B3SOIPD_MOD_UA1:
            value->rValue = model->B3SOIPDua1; 
            return(OK);
        case B3SOIPD_MOD_UB:
            value->rValue = model->B3SOIPDub;  
            return(OK);
        case B3SOIPD_MOD_UB1:
            value->rValue = model->B3SOIPDub1;  
            return(OK);
        case B3SOIPD_MOD_UC:
            value->rValue = model->B3SOIPDuc; 
            return(OK);
        case B3SOIPD_MOD_UC1:
            value->rValue = model->B3SOIPDuc1; 
            return(OK);
        case B3SOIPD_MOD_U0:
            value->rValue = model->B3SOIPDu0;
            return(OK);
        case B3SOIPD_MOD_UTE:
            value->rValue = model->B3SOIPDute;
            return(OK);
        case B3SOIPD_MOD_VOFF:
            value->rValue = model->B3SOIPDvoff;
            return(OK);
        case B3SOIPD_MOD_DELTA:
            value->rValue = model->B3SOIPDdelta;
            return(OK);
        case B3SOIPD_MOD_RDSW:
            value->rValue = model->B3SOIPDrdsw; 
            return(OK);             
        case B3SOIPD_MOD_PRWG:
            value->rValue = model->B3SOIPDprwg; 
            return(OK);             
        case B3SOIPD_MOD_PRWB:
            value->rValue = model->B3SOIPDprwb; 
            return(OK);             
        case B3SOIPD_MOD_PRT:
            value->rValue = model->B3SOIPDprt; 
            return(OK);              
        case B3SOIPD_MOD_ETA0:
            value->rValue = model->B3SOIPDeta0; 
            return(OK);               
        case B3SOIPD_MOD_ETAB:
            value->rValue = model->B3SOIPDetab; 
            return(OK);               
        case B3SOIPD_MOD_PCLM:
            value->rValue = model->B3SOIPDpclm; 
            return(OK);               
        case B3SOIPD_MOD_PDIBL1:
            value->rValue = model->B3SOIPDpdibl1; 
            return(OK);               
        case B3SOIPD_MOD_PDIBL2:
            value->rValue = model->B3SOIPDpdibl2; 
            return(OK);               
        case B3SOIPD_MOD_PDIBLB:
            value->rValue = model->B3SOIPDpdiblb; 
            return(OK);               
        case B3SOIPD_MOD_PVAG:
            value->rValue = model->B3SOIPDpvag; 
            return(OK);               
        case B3SOIPD_MOD_WR:
            value->rValue = model->B3SOIPDwr;
            return(OK);
        case B3SOIPD_MOD_DWG:
            value->rValue = model->B3SOIPDdwg;
            return(OK);
        case B3SOIPD_MOD_DWB:
            value->rValue = model->B3SOIPDdwb;
            return(OK);
        case B3SOIPD_MOD_B0:
            value->rValue = model->B3SOIPDb0;
            return(OK);
        case B3SOIPD_MOD_B1:
            value->rValue = model->B3SOIPDb1;
            return(OK);
        case B3SOIPD_MOD_ALPHA0:
            value->rValue = model->B3SOIPDalpha0;
            return(OK);

        case B3SOIPD_MOD_CGSL:
            value->rValue = model->B3SOIPDcgsl;
            return(OK);
        case B3SOIPD_MOD_CGDL:
            value->rValue = model->B3SOIPDcgdl;
            return(OK);
        case B3SOIPD_MOD_CKAPPA:
            value->rValue = model->B3SOIPDckappa;
            return(OK);
        case B3SOIPD_MOD_CF:
            value->rValue = model->B3SOIPDcf;
            return(OK);
        case B3SOIPD_MOD_CLC:
            value->rValue = model->B3SOIPDclc;
            return(OK);
        case B3SOIPD_MOD_CLE:
            value->rValue = model->B3SOIPDcle;
            return(OK);
        case B3SOIPD_MOD_DWC:
            value->rValue = model->B3SOIPDdwc;
            return(OK);
        case B3SOIPD_MOD_DLC:
            value->rValue = model->B3SOIPDdlc;
            return(OK);

        case B3SOIPD_MOD_TBOX:
            value->rValue = model->B3SOIPDtbox; 
            return(OK);
        case B3SOIPD_MOD_TSI:
            value->rValue = model->B3SOIPDtsi; 
            return(OK);
        case B3SOIPD_MOD_RTH0:
            value->rValue = model->B3SOIPDrth0; 
            return(OK);
        case B3SOIPD_MOD_CTH0:
            value->rValue = model->B3SOIPDcth0; 
            return(OK);
        case B3SOIPD_MOD_NDIODE:
            value->rValue = model->B3SOIPDndiode; 
            return(OK);
        case B3SOIPD_MOD_XBJT:
            value->rValue = model->B3SOIPDxbjt; 
            return(OK);

        case B3SOIPD_MOD_XDIF:
            value->rValue = model->B3SOIPDxdif;
            return(OK);

        case B3SOIPD_MOD_XREC:
            value->rValue = model->B3SOIPDxrec; 
            return(OK);
        case B3SOIPD_MOD_XTUN:
            value->rValue = model->B3SOIPDxtun; 
            return(OK);
        case B3SOIPD_MOD_TT:
            value->rValue = model->B3SOIPDtt; 
            return(OK);
        case B3SOIPD_MOD_VSDTH:
            value->rValue = model->B3SOIPDvsdth; 
            return(OK);
        case B3SOIPD_MOD_VSDFB:
            value->rValue = model->B3SOIPDvsdfb; 
            return(OK);
        case B3SOIPD_MOD_CSDMIN:
            value->rValue = model->B3SOIPDcsdmin; 
            return(OK);
        case B3SOIPD_MOD_ASD:
            value->rValue = model->B3SOIPDasd; 
            return(OK);

        case  B3SOIPD_MOD_TNOM :
          value->rValue = model->B3SOIPDtnom;
            return(OK);
        case B3SOIPD_MOD_CGSO:
            value->rValue = model->B3SOIPDcgso; 
            return(OK);
        case B3SOIPD_MOD_CGDO:
            value->rValue = model->B3SOIPDcgdo; 
            return(OK);
        case B3SOIPD_MOD_CGEO:
            value->rValue = model->B3SOIPDcgeo; 
            return(OK);
        case B3SOIPD_MOD_XPART:
            value->rValue = model->B3SOIPDxpart; 
            return(OK);
        case B3SOIPD_MOD_RSH:
            value->rValue = model->B3SOIPDsheetResistance; 
            return(OK);
        case B3SOIPD_MOD_PBSWG:
            value->rValue = model->B3SOIPDGatesidewallJctPotential; 
            return(OK);
        case B3SOIPD_MOD_MJSWG:
            value->rValue = model->B3SOIPDbodyJctGateSideGradingCoeff; 
            return(OK);
        case B3SOIPD_MOD_CJSWG:
            value->rValue = model->B3SOIPDunitLengthGateSidewallJctCap; 
            return(OK);
        case B3SOIPD_MOD_CSDESW:
            value->rValue = model->B3SOIPDcsdesw; 
            return(OK);
        case B3SOIPD_MOD_LINT:
            value->rValue = model->B3SOIPDLint; 
            return(OK);
        case B3SOIPD_MOD_LL:
            value->rValue = model->B3SOIPDLl;
            return(OK);
/* v2.2.3 */
        case B3SOIPD_MOD_LLC:
            value->rValue = model->B3SOIPDLlc;
            return(OK);

        case B3SOIPD_MOD_LLN:
            value->rValue = model->B3SOIPDLln;
            return(OK);
        case B3SOIPD_MOD_LW:
            value->rValue = model->B3SOIPDLw;
            return(OK);
/* v2.2.3 */
        case B3SOIPD_MOD_LWC:
            value->rValue = model->B3SOIPDLwc;
            return(OK);

        case B3SOIPD_MOD_LWN:
            value->rValue = model->B3SOIPDLwn;
            return(OK);
        case B3SOIPD_MOD_LWL:
            value->rValue = model->B3SOIPDLwl;
            return(OK);
/* v2.2.3 */
        case B3SOIPD_MOD_LWLC:
            value->rValue = model->B3SOIPDLwlc;
            return(OK);

        case B3SOIPD_MOD_WINT:
            value->rValue = model->B3SOIPDWint;
            return(OK);
        case B3SOIPD_MOD_WL:
            value->rValue = model->B3SOIPDWl;
            return(OK);
/* v2.2.3 */
        case B3SOIPD_MOD_WLC:
            value->rValue = model->B3SOIPDWlc;
            return(OK);

        case B3SOIPD_MOD_WLN:
            value->rValue = model->B3SOIPDWln;
            return(OK);
        case B3SOIPD_MOD_WW:
            value->rValue = model->B3SOIPDWw;
            return(OK);
/* v2.2.3 */
        case B3SOIPD_MOD_WWC:
            value->rValue = model->B3SOIPDWwc;
            return(OK);

        case B3SOIPD_MOD_WWN:
            value->rValue = model->B3SOIPDWwn;
            return(OK);
        case B3SOIPD_MOD_WWL:
            value->rValue = model->B3SOIPDWwl;
            return(OK);
/* v2.2.3 */
        case B3SOIPD_MOD_WWLC:
            value->rValue = model->B3SOIPDWwlc;
            return(OK);

        case B3SOIPD_MOD_NOIA:
            value->rValue = model->B3SOIPDoxideTrapDensityA;
            return(OK);
        case B3SOIPD_MOD_NOIB:
            value->rValue = model->B3SOIPDoxideTrapDensityB;
            return(OK);
        case B3SOIPD_MOD_NOIC:
            value->rValue = model->B3SOIPDoxideTrapDensityC;
            return(OK);
        case B3SOIPD_MOD_NOIF:
            value->rValue = model->B3SOIPDnoif;
            return(OK);
        case B3SOIPD_MOD_EM:
            value->rValue = model->B3SOIPDem;
            return(OK);
        case B3SOIPD_MOD_EF:
            value->rValue = model->B3SOIPDef;
            return(OK);
        case B3SOIPD_MOD_AF:
            value->rValue = model->B3SOIPDaf;
            return(OK);
        case B3SOIPD_MOD_KF:
            value->rValue = model->B3SOIPDkf;
            return(OK);


/* v2.0 release */
        case B3SOIPD_MOD_K1W1:                    
            value->rValue = model->B3SOIPDk1w1;
            return(OK);
        case B3SOIPD_MOD_K1W2:
            value->rValue = model->B3SOIPDk1w2;
            return(OK);
        case B3SOIPD_MOD_KETAS:
            value->rValue = model->B3SOIPDketas;
            return(OK);
        case B3SOIPD_MOD_DWBC:
            value->rValue = model->B3SOIPDdwbc;
            return(OK);
        case B3SOIPD_MOD_BETA0:
            value->rValue = model->B3SOIPDbeta0;
            return(OK);
        case B3SOIPD_MOD_BETA1:
            value->rValue = model->B3SOIPDbeta1;
            return(OK);
        case B3SOIPD_MOD_BETA2:
            value->rValue = model->B3SOIPDbeta2;
            return(OK);
        case B3SOIPD_MOD_VDSATII0:
            value->rValue = model->B3SOIPDvdsatii0;
            return(OK);
        case B3SOIPD_MOD_TII:
            value->rValue = model->B3SOIPDtii;
            return(OK);
        case B3SOIPD_MOD_LII:
            value->rValue = model->B3SOIPDlii;
            return(OK);
        case B3SOIPD_MOD_SII0:
            value->rValue = model->B3SOIPDsii0;
            return(OK);
        case B3SOIPD_MOD_SII1:
            value->rValue = model->B3SOIPDsii1;
            return(OK);
        case B3SOIPD_MOD_SII2:
            value->rValue = model->B3SOIPDsii2;
            return(OK);
        case B3SOIPD_MOD_SIID:
            value->rValue = model->B3SOIPDsiid;
            return(OK);
        case B3SOIPD_MOD_FBJTII:
            value->rValue = model->B3SOIPDfbjtii;
            return(OK);
        case B3SOIPD_MOD_ESATII:
            value->rValue = model->B3SOIPDesatii;
            return(OK);
        case B3SOIPD_MOD_NTUN:
            value->rValue = model->B3SOIPDntun;
            return(OK);
        case B3SOIPD_MOD_NRECF0:
            value->rValue = model->B3SOIPDnrecf0;
            return(OK);
        case B3SOIPD_MOD_NRECR0:
            value->rValue = model->B3SOIPDnrecr0;
            return(OK);
        case B3SOIPD_MOD_ISBJT:
            value->rValue = model->B3SOIPDisbjt;
            return(OK);
        case B3SOIPD_MOD_ISDIF:
            value->rValue = model->B3SOIPDisdif;
            return(OK);
        case B3SOIPD_MOD_ISREC:
            value->rValue = model->B3SOIPDisrec;
            return(OK);
        case B3SOIPD_MOD_ISTUN:
            value->rValue = model->B3SOIPDistun;
            return(OK);
        case B3SOIPD_MOD_LN:
            value->rValue = model->B3SOIPDln;
            return(OK);
        case B3SOIPD_MOD_VREC0:
            value->rValue = model->B3SOIPDvrec0;
            return(OK);
        case B3SOIPD_MOD_VTUN0:
            value->rValue = model->B3SOIPDvtun0;
            return(OK);
        case B3SOIPD_MOD_NBJT:
            value->rValue = model->B3SOIPDnbjt;
            return(OK);
        case B3SOIPD_MOD_LBJT0:
            value->rValue = model->B3SOIPDlbjt0;
            return(OK);
        case B3SOIPD_MOD_LDIF0:
            value->rValue = model->B3SOIPDldif0;
            return(OK);
        case B3SOIPD_MOD_VABJT:
            value->rValue = model->B3SOIPDvabjt;
            return(OK);
        case B3SOIPD_MOD_AELY:
            value->rValue = model->B3SOIPDaely;
            return(OK);
        case B3SOIPD_MOD_AHLI:
            value->rValue = model->B3SOIPDahli;
            return(OK);
        case B3SOIPD_MOD_RBODY:
            value->rValue = model->B3SOIPDrbody;
            return(OK);
        case B3SOIPD_MOD_RBSH:
            value->rValue = model->B3SOIPDrbsh;
            return(OK);
        case B3SOIPD_MOD_NTRECF:
            value->rValue = model->B3SOIPDntrecf;
            return(OK);
        case B3SOIPD_MOD_NTRECR:
            value->rValue = model->B3SOIPDntrecr;
            return(OK);
        case B3SOIPD_MOD_NDIF:
            value->rValue = model->B3SOIPDndif;
            return(OK);
        case B3SOIPD_MOD_DLCB:
            value->rValue = model->B3SOIPDdlcb;
            return(OK);
        case B3SOIPD_MOD_FBODY:
            value->rValue = model->B3SOIPDfbody;
            return(OK);
        case B3SOIPD_MOD_TCJSWG:
            value->rValue = model->B3SOIPDtcjswg;
            return(OK);
        case B3SOIPD_MOD_TPBSWG:
            value->rValue = model->B3SOIPDtpbswg;
            return(OK);
        case B3SOIPD_MOD_ACDE:
            value->rValue = model->B3SOIPDacde;
            return(OK);
        case B3SOIPD_MOD_MOIN:
            value->rValue = model->B3SOIPDmoin;
            return(OK);
        case B3SOIPD_MOD_DELVT:
            value->rValue = model->B3SOIPDdelvt;
            return(OK);
        case  B3SOIPD_MOD_KB1:
            value->rValue = model->B3SOIPDkb1;
            return(OK);
        case B3SOIPD_MOD_DLBG:
            value->rValue = model->B3SOIPDdlbg;
            return(OK);

        case B3SOIPD_MOD_NGIDL:
            value->rValue = model->B3SOIPDngidl;
            return(OK);
        case B3SOIPD_MOD_AGIDL:
            value->rValue = model->B3SOIPDagidl;
            return(OK);
        case B3SOIPD_MOD_BGIDL:
            value->rValue = model->B3SOIPDbgidl;
            return(OK);


/* v2.2 release */
        case B3SOIPD_MOD_WTH0:
            value->rValue = model->B3SOIPDwth0;
            return(OK);
        case B3SOIPD_MOD_RHALO:
            value->rValue = model->B3SOIPDrhalo;
            return(OK);
        case B3SOIPD_MOD_NTOX:
            value->rValue = model->B3SOIPDntox;
            return(OK);
        case B3SOIPD_MOD_TOXREF:
            value->rValue = model->B3SOIPDtoxref;
            return(OK);
        case B3SOIPD_MOD_EBG:
            value->rValue = model->B3SOIPDebg;
            return(OK);
        case B3SOIPD_MOD_VEVB:
            value->rValue = model->B3SOIPDvevb;
            return(OK);
        case B3SOIPD_MOD_ALPHAGB1:
            value->rValue = model->B3SOIPDalphaGB1;
            return(OK);
        case B3SOIPD_MOD_BETAGB1:
            value->rValue = model->B3SOIPDbetaGB1;
            return(OK);
        case B3SOIPD_MOD_VGB1:
            value->rValue = model->B3SOIPDvgb1;
            return(OK);
        case B3SOIPD_MOD_VECB:
            value->rValue = model->B3SOIPDvecb;
            return(OK);
        case B3SOIPD_MOD_ALPHAGB2:
            value->rValue = model->B3SOIPDalphaGB2;
            return(OK);
        case B3SOIPD_MOD_BETAGB2:
            value->rValue = model->B3SOIPDbetaGB2;
            return(OK);
        case B3SOIPD_MOD_VGB2:
            value->rValue = model->B3SOIPDvgb2;
            return(OK);
        case B3SOIPD_MOD_TOXQM:
            value->rValue = model->B3SOIPDtoxqm;
            return(OK);
        case B3SOIPD_MOD_VOXH:
            value->rValue = model->B3SOIPDvoxh;
            return(OK);
        case B3SOIPD_MOD_DELTAVOX:
            value->rValue = model->B3SOIPDdeltavox;
            return(OK);
        case B3SOIPD_MOD_IGMOD:
            value->iValue = model->B3SOIPDigMod;
            return(OK);


/* Added for binning - START */
        /* Length Dependence */
        case B3SOIPD_MOD_LNPEAK:
            value->rValue = model->B3SOIPDlnpeak;
            return(OK);
        case B3SOIPD_MOD_LNSUB:
            value->rValue = model->B3SOIPDlnsub;
            return(OK);
        case B3SOIPD_MOD_LNGATE:
            value->rValue = model->B3SOIPDlngate;
            return(OK);
        case B3SOIPD_MOD_LVTH0:
            value->rValue = model->B3SOIPDlvth0;
            return(OK);
        case  B3SOIPD_MOD_LK1:
          value->rValue = model->B3SOIPDlk1;
            return(OK);
        case  B3SOIPD_MOD_LK1W1:
          value->rValue = model->B3SOIPDlk1w1;
            return(OK);
        case  B3SOIPD_MOD_LK1W2:
          value->rValue = model->B3SOIPDlk1w2;
            return(OK);
        case  B3SOIPD_MOD_LK2:
          value->rValue = model->B3SOIPDlk2;
            return(OK);
        case  B3SOIPD_MOD_LK3:
          value->rValue = model->B3SOIPDlk3;
            return(OK);
        case  B3SOIPD_MOD_LK3B:
          value->rValue = model->B3SOIPDlk3b;
            return(OK);
        case  B3SOIPD_MOD_LKB1:
            value->rValue = model->B3SOIPDlkb1;
            return(OK);
        case  B3SOIPD_MOD_LW0:
          value->rValue = model->B3SOIPDlw0;
            return(OK);
        case  B3SOIPD_MOD_LNLX:
          value->rValue = model->B3SOIPDlnlx;
            return(OK);
        case  B3SOIPD_MOD_LDVT0 :
          value->rValue = model->B3SOIPDldvt0;
            return(OK);
        case  B3SOIPD_MOD_LDVT1 :
          value->rValue = model->B3SOIPDldvt1;
            return(OK);
        case  B3SOIPD_MOD_LDVT2 :
          value->rValue = model->B3SOIPDldvt2;
            return(OK);
        case  B3SOIPD_MOD_LDVT0W :
          value->rValue = model->B3SOIPDldvt0w;
            return(OK);
        case  B3SOIPD_MOD_LDVT1W :
          value->rValue = model->B3SOIPDldvt1w;
            return(OK);
        case  B3SOIPD_MOD_LDVT2W :
          value->rValue = model->B3SOIPDldvt2w;
            return(OK);
        case B3SOIPD_MOD_LU0:
            value->rValue = model->B3SOIPDlu0;
            return(OK);
        case B3SOIPD_MOD_LUA:
            value->rValue = model->B3SOIPDlua;
            return(OK);
        case B3SOIPD_MOD_LUB:
            value->rValue = model->B3SOIPDlub;
            return(OK);
        case B3SOIPD_MOD_LUC:
            value->rValue = model->B3SOIPDluc;
            return(OK);
        case B3SOIPD_MOD_LVSAT:
            value->rValue = model->B3SOIPDlvsat;
            return(OK);
        case B3SOIPD_MOD_LA0:
            value->rValue = model->B3SOIPDla0;
            return(OK);
        case B3SOIPD_MOD_LAGS:
            value->rValue = model->B3SOIPDlags;
            return(OK);
        case B3SOIPD_MOD_LB0:
            value->rValue = model->B3SOIPDlb0;
            return(OK);
        case B3SOIPD_MOD_LB1:
            value->rValue = model->B3SOIPDlb1;
            return(OK);
        case B3SOIPD_MOD_LKETA:
            value->rValue = model->B3SOIPDlketa;
            return(OK);
        case B3SOIPD_MOD_LKETAS:
            value->rValue = model->B3SOIPDlketas;
            return(OK);
        case B3SOIPD_MOD_LA1:
            value->rValue = model->B3SOIPDla1;
            return(OK);
        case B3SOIPD_MOD_LA2:
            value->rValue = model->B3SOIPDla2;
            return(OK);
        case B3SOIPD_MOD_LRDSW:
            value->rValue = model->B3SOIPDlrdsw;
            return(OK);
        case B3SOIPD_MOD_LPRWB:
            value->rValue = model->B3SOIPDlprwb;
            return(OK);
        case B3SOIPD_MOD_LPRWG:
            value->rValue = model->B3SOIPDlprwg;
            return(OK);
        case B3SOIPD_MOD_LWR:
            value->rValue = model->B3SOIPDlwr;
            return(OK);
        case  B3SOIPD_MOD_LNFACTOR :
          value->rValue = model->B3SOIPDlnfactor;
            return(OK);
        case B3SOIPD_MOD_LDWG:
            value->rValue = model->B3SOIPDldwg;
            return(OK);
        case B3SOIPD_MOD_LDWB:
            value->rValue = model->B3SOIPDldwb;
            return(OK);
        case B3SOIPD_MOD_LVOFF:
            value->rValue = model->B3SOIPDlvoff;
            return(OK);
        case B3SOIPD_MOD_LETA0:
            value->rValue = model->B3SOIPDleta0;
            return(OK);
        case B3SOIPD_MOD_LETAB:
            value->rValue = model->B3SOIPDletab;
            return(OK);
        case  B3SOIPD_MOD_LDSUB :
          value->rValue = model->B3SOIPDldsub;
            return(OK);
        case  B3SOIPD_MOD_LCIT :
          value->rValue = model->B3SOIPDlcit;
            return(OK);
        case  B3SOIPD_MOD_LCDSC :
          value->rValue = model->B3SOIPDlcdsc;
            return(OK);
        case  B3SOIPD_MOD_LCDSCB :
          value->rValue = model->B3SOIPDlcdscb;
            return(OK);
        case  B3SOIPD_MOD_LCDSCD :
          value->rValue = model->B3SOIPDlcdscd;
            return(OK);
        case B3SOIPD_MOD_LPCLM:
            value->rValue = model->B3SOIPDlpclm;
            return(OK);
        case B3SOIPD_MOD_LPDIBL1:
            value->rValue = model->B3SOIPDlpdibl1;
            return(OK);
        case B3SOIPD_MOD_LPDIBL2:
            value->rValue = model->B3SOIPDlpdibl2;
            return(OK);
        case B3SOIPD_MOD_LPDIBLB:
            value->rValue = model->B3SOIPDlpdiblb;
            return(OK);
        case  B3SOIPD_MOD_LDROUT :
          value->rValue = model->B3SOIPDldrout;
            return(OK);
        case B3SOIPD_MOD_LPVAG:
            value->rValue = model->B3SOIPDlpvag;
            return(OK);
        case B3SOIPD_MOD_LDELTA:
            value->rValue = model->B3SOIPDldelta;
            return(OK);
        case B3SOIPD_MOD_LALPHA0:
            value->rValue = model->B3SOIPDlalpha0;
            return(OK);
        case B3SOIPD_MOD_LFBJTII:
            value->rValue = model->B3SOIPDlfbjtii;
            return(OK);
        case B3SOIPD_MOD_LBETA0:
            value->rValue = model->B3SOIPDlbeta0;
            return(OK);
        case B3SOIPD_MOD_LBETA1:
            value->rValue = model->B3SOIPDlbeta1;
            return(OK);
        case B3SOIPD_MOD_LBETA2:
            value->rValue = model->B3SOIPDlbeta2;
            return(OK);
        case B3SOIPD_MOD_LVDSATII0:
            value->rValue = model->B3SOIPDlvdsatii0;
            return(OK);
        case B3SOIPD_MOD_LLII:
            value->rValue = model->B3SOIPDllii;
            return(OK);
        case B3SOIPD_MOD_LESATII:
            value->rValue = model->B3SOIPDlesatii;
            return(OK);
        case B3SOIPD_MOD_LSII0:
            value->rValue = model->B3SOIPDlsii0;
            return(OK);
        case B3SOIPD_MOD_LSII1:
            value->rValue = model->B3SOIPDlsii1;
            return(OK);
        case B3SOIPD_MOD_LSII2:
            value->rValue = model->B3SOIPDlsii2;
            return(OK);
        case B3SOIPD_MOD_LSIID:
            value->rValue = model->B3SOIPDlsiid;
            return(OK);
        case B3SOIPD_MOD_LAGIDL:
            value->rValue = model->B3SOIPDlagidl;
            return(OK);
        case B3SOIPD_MOD_LBGIDL:
            value->rValue = model->B3SOIPDlbgidl;
            return(OK);
        case B3SOIPD_MOD_LNGIDL:
            value->rValue = model->B3SOIPDlngidl;
            return(OK);
        case B3SOIPD_MOD_LNTUN:
            value->rValue = model->B3SOIPDlntun;
            return(OK);
        case B3SOIPD_MOD_LNDIODE:
            value->rValue = model->B3SOIPDlndiode;
            return(OK);
        case B3SOIPD_MOD_LNRECF0:
            value->rValue = model->B3SOIPDlnrecf0;
            return(OK);
        case B3SOIPD_MOD_LNRECR0:
            value->rValue = model->B3SOIPDlnrecr0;
            return(OK);
        case B3SOIPD_MOD_LISBJT:
            value->rValue = model->B3SOIPDlisbjt;
            return(OK);
        case B3SOIPD_MOD_LISDIF:
            value->rValue = model->B3SOIPDlisdif;
            return(OK);
        case B3SOIPD_MOD_LISREC:
            value->rValue = model->B3SOIPDlisrec;
            return(OK);
        case B3SOIPD_MOD_LISTUN:
            value->rValue = model->B3SOIPDlistun;
            return(OK);
        case B3SOIPD_MOD_LVREC0:
            value->rValue = model->B3SOIPDlvrec0;
            return(OK);
        case B3SOIPD_MOD_LVTUN0:
            value->rValue = model->B3SOIPDlvtun0;
            return(OK);
        case B3SOIPD_MOD_LNBJT:
            value->rValue = model->B3SOIPDlnbjt;
            return(OK);
        case B3SOIPD_MOD_LLBJT0:
            value->rValue = model->B3SOIPDllbjt0;
            return(OK);
        case B3SOIPD_MOD_LVABJT:
            value->rValue = model->B3SOIPDlvabjt;
            return(OK);
        case B3SOIPD_MOD_LAELY:
            value->rValue = model->B3SOIPDlaely;
            return(OK);
        case B3SOIPD_MOD_LAHLI:
            value->rValue = model->B3SOIPDlahli;
            return(OK);
	/* CV Model */
        case B3SOIPD_MOD_LVSDFB:
            value->rValue = model->B3SOIPDlvsdfb;
            return(OK);
        case B3SOIPD_MOD_LVSDTH:
            value->rValue = model->B3SOIPDlvsdth;
            return(OK);
        case B3SOIPD_MOD_LDELVT:
            value->rValue = model->B3SOIPDldelvt;
            return(OK);
        case B3SOIPD_MOD_LACDE:
            value->rValue = model->B3SOIPDlacde;
            return(OK);
        case B3SOIPD_MOD_LMOIN:
            value->rValue = model->B3SOIPDlmoin;
            return(OK);

        /* Width Dependence */
        case B3SOIPD_MOD_WNPEAK:
            value->rValue = model->B3SOIPDwnpeak;
            return(OK);
        case B3SOIPD_MOD_WNSUB:
            value->rValue = model->B3SOIPDwnsub;
            return(OK);
        case B3SOIPD_MOD_WNGATE:
            value->rValue = model->B3SOIPDwngate;
            return(OK);
        case B3SOIPD_MOD_WVTH0:
            value->rValue = model->B3SOIPDwvth0;
            return(OK);
        case  B3SOIPD_MOD_WK1:
          value->rValue = model->B3SOIPDwk1;
            return(OK);
        case  B3SOIPD_MOD_WK1W1:
          value->rValue = model->B3SOIPDwk1w1;
            return(OK);
        case  B3SOIPD_MOD_WK1W2:
          value->rValue = model->B3SOIPDwk1w2;
            return(OK);
        case  B3SOIPD_MOD_WK2:
          value->rValue = model->B3SOIPDwk2;
            return(OK);
        case  B3SOIPD_MOD_WK3:
          value->rValue = model->B3SOIPDwk3;
            return(OK);
        case  B3SOIPD_MOD_WK3B:
          value->rValue = model->B3SOIPDwk3b;
            return(OK);
        case  B3SOIPD_MOD_WKB1:
            value->rValue = model->B3SOIPDwkb1;
            return(OK);
        case  B3SOIPD_MOD_WW0:
          value->rValue = model->B3SOIPDww0;
            return(OK);
        case  B3SOIPD_MOD_WNLX:
          value->rValue = model->B3SOIPDwnlx;
            return(OK);
        case  B3SOIPD_MOD_WDVT0 :
          value->rValue = model->B3SOIPDwdvt0;
            return(OK);
        case  B3SOIPD_MOD_WDVT1 :
          value->rValue = model->B3SOIPDwdvt1;
            return(OK);
        case  B3SOIPD_MOD_WDVT2 :
          value->rValue = model->B3SOIPDwdvt2;
            return(OK);
        case  B3SOIPD_MOD_WDVT0W :
          value->rValue = model->B3SOIPDwdvt0w;
            return(OK);
        case  B3SOIPD_MOD_WDVT1W :
          value->rValue = model->B3SOIPDwdvt1w;
            return(OK);
        case  B3SOIPD_MOD_WDVT2W :
          value->rValue = model->B3SOIPDwdvt2w;
            return(OK);
        case B3SOIPD_MOD_WU0:
            value->rValue = model->B3SOIPDwu0;
            return(OK);
        case B3SOIPD_MOD_WUA:
            value->rValue = model->B3SOIPDwua;
            return(OK);
        case B3SOIPD_MOD_WUB:
            value->rValue = model->B3SOIPDwub;
            return(OK);
        case B3SOIPD_MOD_WUC:
            value->rValue = model->B3SOIPDwuc;
            return(OK);
        case B3SOIPD_MOD_WVSAT:
            value->rValue = model->B3SOIPDwvsat;
            return(OK);
        case B3SOIPD_MOD_WA0:
            value->rValue = model->B3SOIPDwa0;
            return(OK);
        case B3SOIPD_MOD_WAGS:
            value->rValue = model->B3SOIPDwags;
            return(OK);
        case B3SOIPD_MOD_WB0:
            value->rValue = model->B3SOIPDwb0;
            return(OK);
        case B3SOIPD_MOD_WB1:
            value->rValue = model->B3SOIPDwb1;
            return(OK);
        case B3SOIPD_MOD_WKETA:
            value->rValue = model->B3SOIPDwketa;
            return(OK);
        case B3SOIPD_MOD_WKETAS:
            value->rValue = model->B3SOIPDwketas;
            return(OK);
        case B3SOIPD_MOD_WA1:
            value->rValue = model->B3SOIPDwa1;
            return(OK);
        case B3SOIPD_MOD_WA2:
            value->rValue = model->B3SOIPDwa2;
            return(OK);
        case B3SOIPD_MOD_WRDSW:
            value->rValue = model->B3SOIPDwrdsw;
            return(OK);
        case B3SOIPD_MOD_WPRWB:
            value->rValue = model->B3SOIPDwprwb;
            return(OK);
        case B3SOIPD_MOD_WPRWG:
            value->rValue = model->B3SOIPDwprwg;
            return(OK);
        case B3SOIPD_MOD_WWR:
            value->rValue = model->B3SOIPDwwr;
            return(OK);
        case  B3SOIPD_MOD_WNFACTOR :
          value->rValue = model->B3SOIPDwnfactor;
            return(OK);
        case B3SOIPD_MOD_WDWG:
            value->rValue = model->B3SOIPDwdwg;
            return(OK);
        case B3SOIPD_MOD_WDWB:
            value->rValue = model->B3SOIPDwdwb;
            return(OK);
        case B3SOIPD_MOD_WVOFF:
            value->rValue = model->B3SOIPDwvoff;
            return(OK);
        case B3SOIPD_MOD_WETA0:
            value->rValue = model->B3SOIPDweta0;
            return(OK);
        case B3SOIPD_MOD_WETAB:
            value->rValue = model->B3SOIPDwetab;
            return(OK);
        case  B3SOIPD_MOD_WDSUB :
          value->rValue = model->B3SOIPDwdsub;
            return(OK);
        case  B3SOIPD_MOD_WCIT :
          value->rValue = model->B3SOIPDwcit;
            return(OK);
        case  B3SOIPD_MOD_WCDSC :
          value->rValue = model->B3SOIPDwcdsc;
            return(OK);
        case  B3SOIPD_MOD_WCDSCB :
          value->rValue = model->B3SOIPDwcdscb;
            return(OK);
        case  B3SOIPD_MOD_WCDSCD :
          value->rValue = model->B3SOIPDwcdscd;
            return(OK);
        case B3SOIPD_MOD_WPCLM:
            value->rValue = model->B3SOIPDwpclm;
            return(OK);
        case B3SOIPD_MOD_WPDIBL1:
            value->rValue = model->B3SOIPDwpdibl1;
            return(OK);
        case B3SOIPD_MOD_WPDIBL2:
            value->rValue = model->B3SOIPDwpdibl2;
            return(OK);
        case B3SOIPD_MOD_WPDIBLB:
            value->rValue = model->B3SOIPDwpdiblb;
            return(OK);
        case  B3SOIPD_MOD_WDROUT :
          value->rValue = model->B3SOIPDwdrout;
            return(OK);
        case B3SOIPD_MOD_WPVAG:
            value->rValue = model->B3SOIPDwpvag;
            return(OK);
        case B3SOIPD_MOD_WDELTA:
            value->rValue = model->B3SOIPDwdelta;
            return(OK);
        case B3SOIPD_MOD_WALPHA0:
            value->rValue = model->B3SOIPDwalpha0;
            return(OK);
        case B3SOIPD_MOD_WFBJTII:
            value->rValue = model->B3SOIPDwfbjtii;
            return(OK);
        case B3SOIPD_MOD_WBETA0:
            value->rValue = model->B3SOIPDwbeta0;
            return(OK);
        case B3SOIPD_MOD_WBETA1:
            value->rValue = model->B3SOIPDwbeta1;
            return(OK);
        case B3SOIPD_MOD_WBETA2:
            value->rValue = model->B3SOIPDwbeta2;
            return(OK);
        case B3SOIPD_MOD_WVDSATII0:
            value->rValue = model->B3SOIPDwvdsatii0;
            return(OK);
        case B3SOIPD_MOD_WLII:
            value->rValue = model->B3SOIPDwlii;
            return(OK);
        case B3SOIPD_MOD_WESATII:
            value->rValue = model->B3SOIPDwesatii;
            return(OK);
        case B3SOIPD_MOD_WSII0:
            value->rValue = model->B3SOIPDwsii0;
            return(OK);
        case B3SOIPD_MOD_WSII1:
            value->rValue = model->B3SOIPDwsii1;
            return(OK);
        case B3SOIPD_MOD_WSII2:
            value->rValue = model->B3SOIPDwsii2;
            return(OK);
        case B3SOIPD_MOD_WSIID:
            value->rValue = model->B3SOIPDwsiid;
            return(OK);
        case B3SOIPD_MOD_WAGIDL:
            value->rValue = model->B3SOIPDwagidl;
            return(OK);
        case B3SOIPD_MOD_WBGIDL:
            value->rValue = model->B3SOIPDwbgidl;
            return(OK);
        case B3SOIPD_MOD_WNGIDL:
            value->rValue = model->B3SOIPDwngidl;
            return(OK);
        case B3SOIPD_MOD_WNTUN:
            value->rValue = model->B3SOIPDwntun;
            return(OK);
        case B3SOIPD_MOD_WNDIODE:
            value->rValue = model->B3SOIPDwndiode;
            return(OK);
        case B3SOIPD_MOD_WNRECF0:
            value->rValue = model->B3SOIPDwnrecf0;
            return(OK);
        case B3SOIPD_MOD_WNRECR0:
            value->rValue = model->B3SOIPDwnrecr0;
            return(OK);
        case B3SOIPD_MOD_WISBJT:
            value->rValue = model->B3SOIPDwisbjt;
            return(OK);
        case B3SOIPD_MOD_WISDIF:
            value->rValue = model->B3SOIPDwisdif;
            return(OK);
        case B3SOIPD_MOD_WISREC:
            value->rValue = model->B3SOIPDwisrec;
            return(OK);
        case B3SOIPD_MOD_WISTUN:
            value->rValue = model->B3SOIPDwistun;
            return(OK);
        case B3SOIPD_MOD_WVREC0:
            value->rValue = model->B3SOIPDwvrec0;
            return(OK);
        case B3SOIPD_MOD_WVTUN0:
            value->rValue = model->B3SOIPDwvtun0;
            return(OK);
        case B3SOIPD_MOD_WNBJT:
            value->rValue = model->B3SOIPDwnbjt;
            return(OK);
        case B3SOIPD_MOD_WLBJT0:
            value->rValue = model->B3SOIPDwlbjt0;
            return(OK);
        case B3SOIPD_MOD_WVABJT:
            value->rValue = model->B3SOIPDwvabjt;
            return(OK);
        case B3SOIPD_MOD_WAELY:
            value->rValue = model->B3SOIPDwaely;
            return(OK);
        case B3SOIPD_MOD_WAHLI:
            value->rValue = model->B3SOIPDwahli;
            return(OK);
	/* CV Model */
        case B3SOIPD_MOD_WVSDFB:
            value->rValue = model->B3SOIPDwvsdfb;
            return(OK);
        case B3SOIPD_MOD_WVSDTH:
            value->rValue = model->B3SOIPDwvsdth;
            return(OK);
        case B3SOIPD_MOD_WDELVT:
            value->rValue = model->B3SOIPDwdelvt;
            return(OK);
        case B3SOIPD_MOD_WACDE:
            value->rValue = model->B3SOIPDwacde;
            return(OK);
        case B3SOIPD_MOD_WMOIN:
            value->rValue = model->B3SOIPDwmoin;
            return(OK);

        /* Cross-term Dependence */
        case B3SOIPD_MOD_PNPEAK:
            value->rValue = model->B3SOIPDpnpeak;
            return(OK);
        case B3SOIPD_MOD_PNSUB:
            value->rValue = model->B3SOIPDpnsub;
            return(OK);
        case B3SOIPD_MOD_PNGATE:
            value->rValue = model->B3SOIPDpngate;
            return(OK);
        case B3SOIPD_MOD_PVTH0:
            value->rValue = model->B3SOIPDpvth0;
            return(OK);
        case  B3SOIPD_MOD_PK1:
          value->rValue = model->B3SOIPDpk1;
            return(OK);
        case  B3SOIPD_MOD_PK1W1:
          value->rValue = model->B3SOIPDpk1w1;
            return(OK);
        case  B3SOIPD_MOD_PK1W2:
          value->rValue = model->B3SOIPDpk1w2;
            return(OK);
        case  B3SOIPD_MOD_PK2:
          value->rValue = model->B3SOIPDpk2;
            return(OK);
        case  B3SOIPD_MOD_PK3:
          value->rValue = model->B3SOIPDpk3;
            return(OK);
        case  B3SOIPD_MOD_PK3B:
          value->rValue = model->B3SOIPDpk3b;
            return(OK);
        case  B3SOIPD_MOD_PKB1:
            value->rValue = model->B3SOIPDpkb1;
            return(OK);
        case  B3SOIPD_MOD_PW0:
          value->rValue = model->B3SOIPDpw0;
            return(OK);
        case  B3SOIPD_MOD_PNLX:
          value->rValue = model->B3SOIPDpnlx;
            return(OK);
        case  B3SOIPD_MOD_PDVT0 :
          value->rValue = model->B3SOIPDpdvt0;
            return(OK);
        case  B3SOIPD_MOD_PDVT1 :
          value->rValue = model->B3SOIPDpdvt1;
            return(OK);
        case  B3SOIPD_MOD_PDVT2 :
          value->rValue = model->B3SOIPDpdvt2;
            return(OK);
        case  B3SOIPD_MOD_PDVT0W :
          value->rValue = model->B3SOIPDpdvt0w;
            return(OK);
        case  B3SOIPD_MOD_PDVT1W :
          value->rValue = model->B3SOIPDpdvt1w;
            return(OK);
        case  B3SOIPD_MOD_PDVT2W :
          value->rValue = model->B3SOIPDpdvt2w;
            return(OK);
        case B3SOIPD_MOD_PU0:
            value->rValue = model->B3SOIPDpu0;
            return(OK);
        case B3SOIPD_MOD_PUA:
            value->rValue = model->B3SOIPDpua;
            return(OK);
        case B3SOIPD_MOD_PUB:
            value->rValue = model->B3SOIPDpub;
            return(OK);
        case B3SOIPD_MOD_PUC:
            value->rValue = model->B3SOIPDpuc;
            return(OK);
        case B3SOIPD_MOD_PVSAT:
            value->rValue = model->B3SOIPDpvsat;
            return(OK);
        case B3SOIPD_MOD_PA0:
            value->rValue = model->B3SOIPDpa0;
            return(OK);
        case B3SOIPD_MOD_PAGS:
            value->rValue = model->B3SOIPDpags;
            return(OK);
        case B3SOIPD_MOD_PB0:
            value->rValue = model->B3SOIPDpb0;
            return(OK);
        case B3SOIPD_MOD_PB1:
            value->rValue = model->B3SOIPDpb1;
            return(OK);
        case B3SOIPD_MOD_PKETA:
            value->rValue = model->B3SOIPDpketa;
            return(OK);
        case B3SOIPD_MOD_PKETAS:
            value->rValue = model->B3SOIPDpketas;
            return(OK);
        case B3SOIPD_MOD_PA1:
            value->rValue = model->B3SOIPDpa1;
            return(OK);
        case B3SOIPD_MOD_PA2:
            value->rValue = model->B3SOIPDpa2;
            return(OK);
        case B3SOIPD_MOD_PRDSW:
            value->rValue = model->B3SOIPDprdsw;
            return(OK);
        case B3SOIPD_MOD_PPRWB:
            value->rValue = model->B3SOIPDpprwb;
            return(OK);
        case B3SOIPD_MOD_PPRWG:
            value->rValue = model->B3SOIPDpprwg;
            return(OK);
        case B3SOIPD_MOD_PWR:
            value->rValue = model->B3SOIPDpwr;
            return(OK);
        case  B3SOIPD_MOD_PNFACTOR :
          value->rValue = model->B3SOIPDpnfactor;
            return(OK);
        case B3SOIPD_MOD_PDWG:
            value->rValue = model->B3SOIPDpdwg;
            return(OK);
        case B3SOIPD_MOD_PDWB:
            value->rValue = model->B3SOIPDpdwb;
            return(OK);
        case B3SOIPD_MOD_PVOFF:
            value->rValue = model->B3SOIPDpvoff;
            return(OK);
        case B3SOIPD_MOD_PETA0:
            value->rValue = model->B3SOIPDpeta0;
            return(OK);
        case B3SOIPD_MOD_PETAB:
            value->rValue = model->B3SOIPDpetab;
            return(OK);
        case  B3SOIPD_MOD_PDSUB :
          value->rValue = model->B3SOIPDpdsub;
            return(OK);
        case  B3SOIPD_MOD_PCIT :
          value->rValue = model->B3SOIPDpcit;
            return(OK);
        case  B3SOIPD_MOD_PCDSC :
          value->rValue = model->B3SOIPDpcdsc;
            return(OK);
        case  B3SOIPD_MOD_PCDSCB :
          value->rValue = model->B3SOIPDpcdscb;
            return(OK);
        case  B3SOIPD_MOD_PCDSCD :
          value->rValue = model->B3SOIPDpcdscd;
            return(OK);
        case B3SOIPD_MOD_PPCLM:
            value->rValue = model->B3SOIPDppclm;
            return(OK);
        case B3SOIPD_MOD_PPDIBL1:
            value->rValue = model->B3SOIPDppdibl1;
            return(OK);
        case B3SOIPD_MOD_PPDIBL2:
            value->rValue = model->B3SOIPDppdibl2;
            return(OK);
        case B3SOIPD_MOD_PPDIBLB:
            value->rValue = model->B3SOIPDppdiblb;
            return(OK);
        case  B3SOIPD_MOD_PDROUT :
          value->rValue = model->B3SOIPDpdrout;
            return(OK);
        case B3SOIPD_MOD_PPVAG:
            value->rValue = model->B3SOIPDppvag;
            return(OK);
        case B3SOIPD_MOD_PDELTA:
            value->rValue = model->B3SOIPDpdelta;
            return(OK);
        case B3SOIPD_MOD_PALPHA0:
            value->rValue = model->B3SOIPDpalpha0;
            return(OK);
        case B3SOIPD_MOD_PFBJTII:
            value->rValue = model->B3SOIPDpfbjtii;
            return(OK);
        case B3SOIPD_MOD_PBETA0:
            value->rValue = model->B3SOIPDpbeta0;
            return(OK);
        case B3SOIPD_MOD_PBETA1:
            value->rValue = model->B3SOIPDpbeta1;
            return(OK);
        case B3SOIPD_MOD_PBETA2:
            value->rValue = model->B3SOIPDpbeta2;
            return(OK);
        case B3SOIPD_MOD_PVDSATII0:
            value->rValue = model->B3SOIPDpvdsatii0;
            return(OK);
        case B3SOIPD_MOD_PLII:
            value->rValue = model->B3SOIPDplii;
            return(OK);
        case B3SOIPD_MOD_PESATII:
            value->rValue = model->B3SOIPDpesatii;
            return(OK);
        case B3SOIPD_MOD_PSII0:
            value->rValue = model->B3SOIPDpsii0;
            return(OK);
        case B3SOIPD_MOD_PSII1:
            value->rValue = model->B3SOIPDpsii1;
            return(OK);
        case B3SOIPD_MOD_PSII2:
            value->rValue = model->B3SOIPDpsii2;
            return(OK);
        case B3SOIPD_MOD_PSIID:
            value->rValue = model->B3SOIPDpsiid;
            return(OK);
        case B3SOIPD_MOD_PAGIDL:
            value->rValue = model->B3SOIPDpagidl;
            return(OK);
        case B3SOIPD_MOD_PBGIDL:
            value->rValue = model->B3SOIPDpbgidl;
            return(OK);
        case B3SOIPD_MOD_PNGIDL:
            value->rValue = model->B3SOIPDpngidl;
            return(OK);
        case B3SOIPD_MOD_PNTUN:
            value->rValue = model->B3SOIPDpntun;
            return(OK);
        case B3SOIPD_MOD_PNDIODE:
            value->rValue = model->B3SOIPDpndiode;
            return(OK);
        case B3SOIPD_MOD_PNRECF0:
            value->rValue = model->B3SOIPDpnrecf0;
            return(OK);
        case B3SOIPD_MOD_PNRECR0:
            value->rValue = model->B3SOIPDpnrecr0;
            return(OK);
        case B3SOIPD_MOD_PISBJT:
            value->rValue = model->B3SOIPDpisbjt;
            return(OK);
        case B3SOIPD_MOD_PISDIF:
            value->rValue = model->B3SOIPDpisdif;
            return(OK);
        case B3SOIPD_MOD_PISREC:
            value->rValue = model->B3SOIPDpisrec;
            return(OK);
        case B3SOIPD_MOD_PISTUN:
            value->rValue = model->B3SOIPDpistun;
            return(OK);
        case B3SOIPD_MOD_PVREC0:
            value->rValue = model->B3SOIPDpvrec0;
            return(OK);
        case B3SOIPD_MOD_PVTUN0:
            value->rValue = model->B3SOIPDpvtun0;
            return(OK);
        case B3SOIPD_MOD_PNBJT:
            value->rValue = model->B3SOIPDpnbjt;
            return(OK);
        case B3SOIPD_MOD_PLBJT0:
            value->rValue = model->B3SOIPDplbjt0;
            return(OK);
        case B3SOIPD_MOD_PVABJT:
            value->rValue = model->B3SOIPDpvabjt;
            return(OK);
        case B3SOIPD_MOD_PAELY:
            value->rValue = model->B3SOIPDpaely;
            return(OK);
        case B3SOIPD_MOD_PAHLI:
            value->rValue = model->B3SOIPDpahli;
            return(OK);
	/* CV Model */
        case B3SOIPD_MOD_PVSDFB:
            value->rValue = model->B3SOIPDpvsdfb;
            return(OK);
        case B3SOIPD_MOD_PVSDTH:
            value->rValue = model->B3SOIPDpvsdth;
            return(OK);
        case B3SOIPD_MOD_PDELVT:
            value->rValue = model->B3SOIPDpdelvt;
            return(OK);
        case B3SOIPD_MOD_PACDE:
            value->rValue = model->B3SOIPDpacde;
            return(OK);
        case B3SOIPD_MOD_PMOIN:
            value->rValue = model->B3SOIPDpmoin;
            return(OK);
/* Added for binning - END */

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



