/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdcheck.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su 00/3/1
Modified by Pin Su and Hui Wan 02/3/5
Modifies by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
B3SOIPDcheckModel(B3SOIPDmodel *model, B3SOIPDinstance *here, CKTcircuit *ckt)
{
struct b3soipdSizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;

    NG_IGNORE(ckt);
    
    if ((fplog = fopen("b3soipdv223check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "B3SOIPDV223 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->B3SOIPDmodName);
	fprintf(fplog, "W = %g, L = %g, M = %g\n", here->B3SOIPDw, 
	here->B3SOIPDl, here->B3SOIPDm);
   	    

            if (pParam->B3SOIPDnlx < -pParam->B3SOIPDleff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOIPDnlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOIPDnlx);
		Fatal_Flag = 1;
            }

	if (model->B3SOIPDtox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->B3SOIPDtox);
	    printf("Fatal: Tox = %g is not positive.\n", model->B3SOIPDtox);
	    Fatal_Flag = 1;
	}

/* v2.2.3 */
        if (model->B3SOIPDtox - model->B3SOIPDdtoxcv <= 0.0)
        {   fprintf(fplog, "Fatal: Tox - dtoxcv = %g is not positive.\n",
                    model->B3SOIPDtox - model->B3SOIPDdtoxcv);
            printf("Fatal: Tox - dtoxcv = %g is not positive.\n", model->B3SOIPDtox - model->B3SOIPDdtoxcv);
            Fatal_Flag = 1;
        }


	if (model->B3SOIPDtbox <= 0.0)
	{   fprintf(fplog, "Fatal: Tbox = %g is not positive.\n",
		    model->B3SOIPDtbox);
	    printf("Fatal: Tbox = %g is not positive.\n", model->B3SOIPDtbox);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIPDnpeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->B3SOIPDnpeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->B3SOIPDnpeak);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIPDngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->B3SOIPDngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->B3SOIPDngate);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIPDngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->B3SOIPDngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->B3SOIPDngate);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIPDdvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->B3SOIPDdvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->B3SOIPDdvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIPDdvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->B3SOIPDdvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->B3SOIPDdvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIPDw0 == -pParam->B3SOIPDweff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->B3SOIPDdsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->B3SOIPDdsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->B3SOIPDdsub);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIPDb1 == -pParam->B3SOIPDweff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->B3SOIPDu0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->B3SOIPDu0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->B3SOIPDu0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->B3SOIPDdelta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->B3SOIPDdelta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->B3SOIPDdelta);
	    Fatal_Flag = 1;
        }      

	if (pParam->B3SOIPDvsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->B3SOIPDvsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->B3SOIPDvsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->B3SOIPDpclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->B3SOIPDpclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->B3SOIPDpclm);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIPDdrout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->B3SOIPDdrout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->B3SOIPDdrout);
	    Fatal_Flag = 1;
	}
      if ( model->B3SOIPDunitLengthGateSidewallJctCap > 0.0)
      {
	if (here->B3SOIPDdrainPerimeter < pParam->B3SOIPDweff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->B3SOIPDdrainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->B3SOIPDdrainPerimeter);
            here->B3SOIPDdrainPerimeter =pParam->B3SOIPDweff; 
	}
	if (here->B3SOIPDsourcePerimeter < pParam->B3SOIPDweff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->B3SOIPDsourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->B3SOIPDsourcePerimeter);
            here->B3SOIPDsourcePerimeter =pParam->B3SOIPDweff;
	}
      }
/* Check capacitance parameters */
        if (pParam->B3SOIPDclc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->B3SOIPDclc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->B3SOIPDclc);
	    Fatal_Flag = 1;
        } 

/* v2.2.3 */
        if (pParam->B3SOIPDmoin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->B3SOIPDmoin);
            printf("Warning: Moin = %g is too small.\n", pParam->B3SOIPDmoin);
        }
        if (pParam->B3SOIPDmoin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->B3SOIPDmoin);
            printf("Warning: Moin = %g is too large.\n", pParam->B3SOIPDmoin);
        }


	if (model->B3SOIPDcapMod == 3) {
		if (pParam->B3SOIPDacde < 0.4)
		{  fprintf (fplog, "Warning: Acde = %g is too small.\n",
		   	    pParam->B3SOIPDacde);
 		   printf ("Warning: Acde = %g is too small.\n",
                    	    pParam->B3SOIPDacde);
		}
		if (pParam->B3SOIPDacde > 1.6)
		{  fprintf (fplog, "Warning: Acde = %g is too large.\n",
		   	    pParam->B3SOIPDacde);
 		   printf ("Warning: Acde = %g is too large.\n",
                    	    pParam->B3SOIPDacde);
		}
	}
/* v2.2.3 */
     
      if (model->B3SOIPDparamChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->B3SOIPDleff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->B3SOIPDleff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->B3SOIPDleff);
	}    
	
	if (pParam->B3SOIPDleffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->B3SOIPDleffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->B3SOIPDleffCV);
	}  
	
        if (pParam->B3SOIPDweff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->B3SOIPDweff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->B3SOIPDweff);
	}             
	
	if (pParam->B3SOIPDweffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->B3SOIPDweffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->B3SOIPDweffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->B3SOIPDnlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->B3SOIPDnlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->B3SOIPDnlx);
        }
	if (model->B3SOIPDtox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->B3SOIPDtox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->B3SOIPDtox);
        }

        if (pParam->B3SOIPDnpeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->B3SOIPDnpeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->B3SOIPDnpeak);
	}
	else if (pParam->B3SOIPDnpeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->B3SOIPDnpeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->B3SOIPDnpeak);
	}

	if (fabs(pParam->B3SOIPDnsub) >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->B3SOIPDnsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->B3SOIPDnsub);
	}

	if ((pParam->B3SOIPDngate > 0.0) &&
	    (pParam->B3SOIPDngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->B3SOIPDngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->B3SOIPDngate);
	}
       
        if (pParam->B3SOIPDdvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->B3SOIPDdvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->B3SOIPDdvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->B3SOIPDw0 + pParam->B3SOIPDweff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->B3SOIPDnfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->B3SOIPDnfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->B3SOIPDnfactor);
	}
	if (pParam->B3SOIPDcdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->B3SOIPDcdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->B3SOIPDcdsc);
	}
	if (pParam->B3SOIPDcdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->B3SOIPDcdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->B3SOIPDcdscd);
	}
/* Check DIBL parameters */
	if (pParam->B3SOIPDeta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->B3SOIPDeta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->B3SOIPDeta0); 
	}
	      
/* Check Abulk parameters */	    
        if (fabs(1.0e-6 / (pParam->B3SOIPDb1 + pParam->B3SOIPDweff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    

/* Check Saturation parameters */
     	if (pParam->B3SOIPDa2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->B3SOIPDa2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->B3SOIPDa2);
	    pParam->B3SOIPDa2 = 0.01;
	}
	else if (pParam->B3SOIPDa2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->B3SOIPDa2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->B3SOIPDa2);
	    pParam->B3SOIPDa2 = 1.0;
	    pParam->B3SOIPDa1 = 0.0;

	}

	if (pParam->B3SOIPDrdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->B3SOIPDrdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->B3SOIPDrdsw);
	    pParam->B3SOIPDrdsw = 0.0;
	    pParam->B3SOIPDrds0 = 0.0;
	}
	else if ((pParam->B3SOIPDrds0 > 0.0) && (pParam->B3SOIPDrds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->B3SOIPDrds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->B3SOIPDrds0);
	    pParam->B3SOIPDrds0 = 0.0;
	}
	 if (pParam->B3SOIPDvsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIPDvsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIPDvsattemp);
	}

	if (pParam->B3SOIPDpdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->B3SOIPDpdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->B3SOIPDpdibl1);
	}
	if (pParam->B3SOIPDpdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->B3SOIPDpdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->B3SOIPDpdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->B3SOIPDcgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIPDcgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIPDcgdo);
	    model->B3SOIPDcgdo = 0.0;
        }      
        if (model->B3SOIPDcgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIPDcgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIPDcgso);
	    model->B3SOIPDcgso = 0.0;
        }      
        if (model->B3SOIPDcgeo < 0.0)
	{   fprintf(fplog, "Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIPDcgeo);
	    printf("Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIPDcgeo);
	    model->B3SOIPDcgeo = 0.0;
        }

	if (model->B3SOIPDntun < 0.0)
	{   fprintf(fplog, "Warning: Ntun = %g is negative.\n",
		    model->B3SOIPDntun); 
	    printf("Warning: Ntun = %g is negative.\n", model->B3SOIPDntun); 
	}

	if (model->B3SOIPDndiode < 0.0)
	{   fprintf(fplog, "Warning: Ndiode = %g is negative.\n",
		    model->B3SOIPDndiode); 
	    printf("Warning: Ndiode = %g is negative.\n", model->B3SOIPDndiode); 
	}

	if (model->B3SOIPDisbjt < 0.0)
	{   fprintf(fplog, "Warning: Isbjt = %g is negative.\n",
		    model->B3SOIPDisbjt); 
	    printf("Warning: Isbjt = %g is negative.\n", model->B3SOIPDisbjt); 
	}

	if (model->B3SOIPDisdif < 0.0)
	{   fprintf(fplog, "Warning: Isdif = %g is negative.\n",
		    model->B3SOIPDisdif); 
	    printf("Warning: Isdif = %g is negative.\n", model->B3SOIPDisdif); 
	}

	if (model->B3SOIPDisrec < 0.0)
	{   fprintf(fplog, "Warning: Isrec = %g is negative.\n",
		    model->B3SOIPDisrec); 
	    printf("Warning: Isrec = %g is negative.\n", model->B3SOIPDisrec); 
	}

	if (model->B3SOIPDistun < 0.0)
	{   fprintf(fplog, "Warning: Istun = %g is negative.\n",
		    model->B3SOIPDistun); 
	    printf("Warning: Istun = %g is negative.\n", model->B3SOIPDistun); 
	}

	if (model->B3SOIPDtt < 0.0)
	{   fprintf(fplog, "Warning: Tt = %g is negative.\n",
		    model->B3SOIPDtt); 
	    printf("Warning: Tt = %g is negative.\n", model->B3SOIPDtt); 
	}

	if (model->B3SOIPDcsdmin < 0.0)
	{   fprintf(fplog, "Warning: Csdmin = %g is negative.\n",
		    model->B3SOIPDcsdmin); 
	    printf("Warning: Csdmin = %g is negative.\n", model->B3SOIPDcsdmin); 
	}

	if (model->B3SOIPDcsdesw < 0.0)
	{   fprintf(fplog, "Warning: Csdesw = %g is negative.\n",
		    model->B3SOIPDcsdesw); 
	    printf("Warning: Csdesw = %g is negative.\n", model->B3SOIPDcsdesw); 
	}

	if (model->B3SOIPDasd < 0.0)
	{   fprintf(fplog, "Warning: Asd = %g should be within (0, 1).\n",
		    model->B3SOIPDasd); 
	    printf("Warning: Asd = %g should be within (0, 1).\n", model->B3SOIPDasd); 
	}

	if (model->B3SOIPDrth0 < 0.0)
	{   fprintf(fplog, "Warning: Rth0 = %g is negative.\n",
		    model->B3SOIPDrth0); 
	    printf("Warning: Rth0 = %g is negative.\n", model->B3SOIPDrth0); 
	}

	if (model->B3SOIPDcth0 < 0.0)
	{   fprintf(fplog, "Warning: Cth0 = %g is negative.\n",
		    model->B3SOIPDcth0); 
	    printf("Warning: Cth0 = %g is negative.\n", model->B3SOIPDcth0); 
	}

	if (model->B3SOIPDrbody < 0.0)
	{   fprintf(fplog, "Warning: Rbody = %g is negative.\n",
		    model->B3SOIPDrbody); 
	    printf("Warning: Rbody = %g is negative.\n", model->B3SOIPDrbody); 
	}

	if (model->B3SOIPDrbsh < 0.0)
	{   fprintf(fplog, "Warning: Rbsh = %g is negative.\n",
		    model->B3SOIPDrbsh); 
	    printf("Warning: Rbsh = %g is negative.\n", model->B3SOIPDrbsh); 
	}


/* v2.2 release */
        if (model->B3SOIPDwth0 < 0.0)
        {   fprintf(fplog, "Warning: WTH0 = %g is negative.\n",
                    model->B3SOIPDwth0);
            printf("Warning:  Wth0 = %g is negative.\n", model->B3SOIPDwth0);
        }
        if (model->B3SOIPDrhalo < 0.0)
        {   fprintf(fplog, "Warning: RHALO = %g is negative.\n",
                    model->B3SOIPDrhalo);
            printf("Warning:  Rhalo = %g is negative.\n", model->B3SOIPDrhalo);
        }
        if (model->B3SOIPDntox < 0.0)
        {   fprintf(fplog, "Warning: NTOX = %g is negative.\n",
                    model->B3SOIPDntox);
            printf("Warning:  Ntox = %g is negative.\n", model->B3SOIPDntox);
        }
        if (model->B3SOIPDtoxref < 0.0)
        {   fprintf(fplog, "Warning: TOXREF = %g is negative.\n",
                    model->B3SOIPDtoxref);
            printf("Warning:  Toxref = %g is negative.\n", model->B3SOIPDtoxref);
            Fatal_Flag = 1;
        }
        if (model->B3SOIPDebg < 0.0)
        {   fprintf(fplog, "Warning: EBG = %g is negative.\n",
                    model->B3SOIPDebg);
            printf("Warning:  Ebg = %g is negative.\n", model->B3SOIPDebg);
        }
        if (model->B3SOIPDvevb < 0.0)
        {   fprintf(fplog, "Warning: VEVB = %g is negative.\n",
                    model->B3SOIPDvevb);
            printf("Warning:  Vevb = %g is negative.\n", model->B3SOIPDvevb);
        }
        if (model->B3SOIPDalphaGB1 < 0.0)
        {   fprintf(fplog, "Warning: ALPHAGB1 = %g is negative.\n",
                    model->B3SOIPDalphaGB1);
            printf("Warning:  AlphaGB1 = %g is negative.\n", model->B3SOIPDalphaGB1);
        }
        if (model->B3SOIPDbetaGB1 < 0.0)
        {   fprintf(fplog, "Warning: BETAGB1 = %g is negative.\n",
                    model->B3SOIPDbetaGB1);
            printf("Warning:  BetaGB1 = %g is negative.\n", model->B3SOIPDbetaGB1);
        }
        if (model->B3SOIPDvgb1 < 0.0)
        {   fprintf(fplog, "Warning: VGB1 = %g is negative.\n",
                    model->B3SOIPDvgb1);
            printf("Warning:  Vgb1 = %g is negative.\n", model->B3SOIPDvgb1);
        }
        if (model->B3SOIPDvecb < 0.0)
        {   fprintf(fplog, "Warning: VECB = %g is negative.\n",
                    model->B3SOIPDvecb);
            printf("Warning:  Vecb = %g is negative.\n", model->B3SOIPDvecb);
        }
        if (model->B3SOIPDalphaGB2 < 0.0)
        {   fprintf(fplog, "Warning: ALPHAGB2 = %g is negative.\n",
                    model->B3SOIPDalphaGB2);
            printf("Warning:  AlphaGB2 = %g is negative.\n", model->B3SOIPDalphaGB2);
        }
        if (model->B3SOIPDbetaGB2 < 0.0)
        {   fprintf(fplog, "Warning: BETAGB2 = %g is negative.\n",
                    model->B3SOIPDbetaGB2);
            printf("Warning:  BetaGB2 = %g is negative.\n", model->B3SOIPDbetaGB2);
        }
        if (model->B3SOIPDvgb2 < 0.0)
        {   fprintf(fplog, "Warning: VGB2 = %g is negative.\n",
                    model->B3SOIPDvgb2);
            printf("Warning:  Vgb2 = %g is negative.\n", model->B3SOIPDvgb2);
        }
        if (model->B3SOIPDtoxqm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxqm = %g is not positive.\n",
                    model->B3SOIPDtoxqm);
            printf("Fatal: Toxqm = %g is not positive.\n", model->B3SOIPDtoxqm);
            Fatal_Flag = 1;
        }
        if (model->B3SOIPDvoxh < 0.0)
        {   fprintf(fplog, "Warning: Voxh = %g is negative.\n",
                    model->B3SOIPDvoxh);
            printf("Warning:  Voxh = %g is negative.\n", model->B3SOIPDvoxh);
        }
        if (model->B3SOIPDdeltavox <= 0.0)
        {   fprintf(fplog, "Fatal: Deltavox = %g is not positive.\n",
                    model->B3SOIPDdeltavox);
            printf("Fatal: Deltavox = %g is not positive.\n", model->B3SOIPDdeltavox);
        }


/* v2.0 release */
        if (model->B3SOIPDk1w1 < 0.0)                    
        {   fprintf(fplog, "Warning: K1W1 = %g is negative.\n",
                    model->B3SOIPDk1w1);
            printf("Warning:  K1w1 = %g is negative.\n", model->B3SOIPDk1w1);
        }
        if (model->B3SOIPDk1w2 < 0.0)
        {   fprintf(fplog, "Warning: K1W2 = %g is negative.\n",
                    model->B3SOIPDk1w2);
            printf("Warning:  K1w2 = %g is negative.\n", model->B3SOIPDk1w2);
        }
        if (model->B3SOIPDketas < 0.0)
        {   fprintf(fplog, "Warning: KETAS = %g is negative.\n",
                    model->B3SOIPDketas);
            printf("Warning:  Ketas = %g is negative.\n", model->B3SOIPDketas);
        }
        if (model->B3SOIPDdwbc < 0.0)
        {   fprintf(fplog, "Warning: DWBC = %g is negative.\n",
                    model->B3SOIPDdwbc);
            printf("Warning:  Dwbc = %g is negative.\n", model->B3SOIPDdwbc);
        }
        if (model->B3SOIPDbeta0 < 0.0)
        {   fprintf(fplog, "Warning: BETA0 = %g is negative.\n",
                    model->B3SOIPDbeta0);
            printf("Warning:  Beta0 = %g is negative.\n", model->B3SOIPDbeta0);
        }
        if (model->B3SOIPDbeta1 < 0.0)
        {   fprintf(fplog, "Warning: BETA1 = %g is negative.\n",
                    model->B3SOIPDbeta1);
            printf("Warning:  Beta1 = %g is negative.\n", model->B3SOIPDbeta1);
        }
        if (model->B3SOIPDbeta2 < 0.0)
        {   fprintf(fplog, "Warning: BETA2 = %g is negative.\n",
                    model->B3SOIPDbeta2);
            printf("Warning:  Beta2 = %g is negative.\n", model->B3SOIPDbeta2);
        }
        if (model->B3SOIPDtii < 0.0)
        {   fprintf(fplog, "Warning: TII = %g is negative.\n",
                    model->B3SOIPDtii);
            printf("Warning:  Tii = %g is negative.\n", model->B3SOIPDtii);
        }
        if (model->B3SOIPDlii < 0.0)
        {   fprintf(fplog, "Warning: LII = %g is negative.\n",
                    model->B3SOIPDlii);
            printf("Warning:  Lii = %g is negative.\n", model->B3SOIPDlii);
        }
        if (model->B3SOIPDsii1 < 0.0)
        {   fprintf(fplog, "Warning: SII1 = %g is negative.\n",
                    model->B3SOIPDsii1);
            printf("Warning:  Sii1 = %g is negative.\n", model->B3SOIPDsii1);
        }
        if (model->B3SOIPDsii2 < 0.0)
        {   fprintf(fplog, "Warning: SII2 = %g is negative.\n", 
                    model->B3SOIPDsii2);
            printf("Warning:  Sii2 = %g is negative.\n", model->B3SOIPDsii1);
        }
        if (model->B3SOIPDsiid < 0.0)
        {   fprintf(fplog, "Warning: SIID = %g is negative.\n",
                    model->B3SOIPDsiid);
            printf("Warning:  Siid = %g is negative.\n", model->B3SOIPDsiid);
        }
        if (model->B3SOIPDfbjtii < 0.0)
        {   fprintf(fplog, "Warning: FBJTII = %g is negative.\n",
                    model->B3SOIPDfbjtii);
            printf("Warning:  fbjtii = %g is negative.\n", model->B3SOIPDfbjtii);
        }
        if (model->B3SOIPDvrec0 < 0.0)
        {   fprintf(fplog, "Warning: VREC0 = %g is negative.\n",
                    model->B3SOIPDvrec0);
            printf("Warning:  Vrec0 = %g is negative.\n", model->B3SOIPDvrec0);
        }
        if (model->B3SOIPDvtun0 < 0.0)
        {   fprintf(fplog, "Warning: VTUN0 = %g is negative.\n",
                    model->B3SOIPDvtun0);
            printf("Warning:  Vtun0 = %g is negative.\n", model->B3SOIPDvtun0);
        }
        if (model->B3SOIPDnbjt < 0.0)
        {   fprintf(fplog, "Warning: NBJT = %g is negative.\n",
                    model->B3SOIPDnbjt);
            printf("Warning:  Nbjt = %g is negative.\n", model->B3SOIPDnbjt);
        }
        if (model->B3SOIPDaely < 0.0)
        {   fprintf(fplog, "Warning: AELY = %g is negative.\n",
                    model->B3SOIPDaely);
            printf("Warning:  Aely = %g is negative.\n", model->B3SOIPDaely);
        }
        if (model->B3SOIPDahli < 0.0)
        {   fprintf(fplog, "Warning: AHLI = %g is negative.\n",
                    model->B3SOIPDahli);
            printf("Warning:  Ahli = %g is negative.\n", model->B3SOIPDahli);
        }
        if (model->B3SOIPDrbody < 0.0)
        {   fprintf(fplog, "Warning: RBODY = %g is negative.\n",
                    model->B3SOIPDrbody);
            printf("Warning:  Rbody = %g is negative.\n", model->B3SOIPDrbody);
        }
        if (model->B3SOIPDrbsh < 0.0)
        {   fprintf(fplog, "Warning: RBSH = %g is negative.\n",
                    model->B3SOIPDrbsh);
            printf("Warning:  Rbsh = %g is negative.\n", model->B3SOIPDrbsh);
        }
        if (model->B3SOIPDntrecf < 0.0)
        {   fprintf(fplog, "Warning: NTRECF = %g is negative.\n",
                    model->B3SOIPDntrecf);
            printf("Warning:  Ntrecf = %g is negative.\n", model->B3SOIPDntrecf);
        }
        if (model->B3SOIPDntrecr < 0.0)
        {   fprintf(fplog, "Warning: NTRECR = %g is negative.\n",
                    model->B3SOIPDntrecr);
            printf("Warning:  Ntrecr = %g is negative.\n", model->B3SOIPDntrecr);
        }
        if (model->B3SOIPDndif < 0.0)
        {   fprintf(fplog, "Warning: NDIF = %g is negative.\n",
                    model->B3SOIPDndif);
            printf("Warning:  Ndif = %g is negative.\n", model->B3SOIPDndif);
        }
        if (model->B3SOIPDtcjswg < 0.0)
        {   fprintf(fplog, "Warning: TCJSWG = %g is negative.\n",
                    model->B3SOIPDtcjswg);
            printf("Warning:  Tcjswg = %g is negative.\n", model->B3SOIPDtcjswg);
        }
        if (model->B3SOIPDtpbswg < 0.0)
        {   fprintf(fplog, "Warning: TPBSWG = %g is negative.\n",
                    model->B3SOIPDtpbswg);
            printf("Warning:  Tpbswg = %g is negative.\n", model->B3SOIPDtpbswg);
        }
        if ((model->B3SOIPDacde < 0.4) || (model->B3SOIPDacde > 1.6))
        {   fprintf(fplog, "Warning: ACDE = %g is out of range.\n",
                    model->B3SOIPDacde);
            printf("Warning:  Acde = %g is out of range.\n", model->B3SOIPDacde);
        }
        if ((model->B3SOIPDmoin < 5.0)||(model->B3SOIPDmoin > 25.0))
        {   fprintf(fplog, "Warning: MOIN = %g is out of range.\n",
                    model->B3SOIPDmoin);
            printf("Warning:  Moin = %g is out of range.\n", model->B3SOIPDmoin);
        }
        if (model->B3SOIPDdlbg < 0.0)
        {   fprintf(fplog, "Warning: DLBG = %g is negative.\n",
                    model->B3SOIPDdlbg);
            printf("Warning:  dlbg = %g is negative.\n", model->B3SOIPDdlbg);
        }


        if (model->B3SOIPDagidl < 0.0)
        {   fprintf(fplog, "Warning: AGIDL = %g is negative.\n",
                    model->B3SOIPDagidl);
            printf("Warning:  Agidl = %g is negative.\n", model->B3SOIPDagidl);
        }
        if (model->B3SOIPDbgidl < 0.0)
        {   fprintf(fplog, "Warning: BGIDL = %g is negative.\n",
                    model->B3SOIPDbgidl);
            printf("Warning:  Bgidl = %g is negative.\n", model->B3SOIPDbgidl);
        }
        if (model->B3SOIPDngidl < 0.0)
        {   fprintf(fplog, "Warning: NGIDL = %g is negative.\n",
                    model->B3SOIPDngidl);
            printf("Warning:  Ngidl = %g is negative.\n", model->B3SOIPDngidl);
        }
        if (model->B3SOIPDesatii < 0.0)
        {   fprintf(fplog, "Warning: Esatii = %g should be within positive.\n",
                    model->B3SOIPDesatii);
            printf("Warning: Esatii = %g should be within (0, 1).\n", model->B3SOIPDesatii);
        }


	if (model->B3SOIPDxj > model->B3SOIPDtsi)
	{   fprintf(fplog, "Warning: Xj = %g is thicker than Tsi = %g.\n",
		    model->B3SOIPDxj, model->B3SOIPDtsi); 
	    printf("Warning: Xj = %g is thicker than Tsi = %g.\n", 
		    model->B3SOIPDxj, model->B3SOIPDtsi); 
	}

        if (model->B3SOIPDcapMod < 2)
	{   fprintf(fplog, "Warning: capMod < 2 is not supported by BSIM3SOI.\n");
	    printf("Warning: Warning: capMod < 2 is not supported by BSIM3SOI.\n");
	}

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

