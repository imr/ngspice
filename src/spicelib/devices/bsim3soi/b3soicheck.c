/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soicheck.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Pin Su 00/3/1
Modified by Pin Su and Hui Wan 02/3/5
Modified by Pin Su 02/5/20
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "b3soidef.h"
#include "trandefs.h"
#include "const.h"
#include "sperror.h"
#include "devdefs.h"
#include "suffix.h"

int
B3SOIcheckModel(B3SOImodel *model, B3SOIinstance *here, CKTcircuit *ckt)
{
struct b3soiSizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    if ((fplog = fopen("b3soiV3check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "B3SOIV3 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->B3SOImodName);
	fprintf(fplog, "W = %g, L = %g\n", here->B3SOIw, here->B3SOIl);
   	    

            if (pParam->B3SOInlx < -pParam->B3SOIleff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOInlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOInlx);
		Fatal_Flag = 1;
            }

	if (model->B3SOItox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->B3SOItox);
	    printf("Fatal: Tox = %g is not positive.\n", model->B3SOItox);
	    Fatal_Flag = 1;
	}

/* v2.2.3 */
        if (model->B3SOItox - model->B3SOIdtoxcv <= 0.0)
        {   fprintf(fplog, "Fatal: Tox - dtoxcv = %g is not positive.\n",
                    model->B3SOItox - model->B3SOIdtoxcv);
            printf("Fatal: Tox - dtoxcv = %g is not positive.\n", model->B3SOItox - model->B3SOIdtoxcv);
            Fatal_Flag = 1;
        }


	if (model->B3SOItbox <= 0.0)
	{   fprintf(fplog, "Fatal: Tbox = %g is not positive.\n",
		    model->B3SOItbox);
	    printf("Fatal: Tbox = %g is not positive.\n", model->B3SOItbox);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOInpeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->B3SOInpeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->B3SOInpeak);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->B3SOIngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->B3SOIngate);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->B3SOIngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->B3SOIngate);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIdvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->B3SOIdvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->B3SOIdvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIdvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->B3SOIdvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->B3SOIdvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIw0 == -pParam->B3SOIweff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->B3SOIdsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->B3SOIdsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->B3SOIdsub);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIb1 == -pParam->B3SOIweff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->B3SOIu0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->B3SOIu0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->B3SOIu0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->B3SOIdelta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->B3SOIdelta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->B3SOIdelta);
	    Fatal_Flag = 1;
        }      

	if (pParam->B3SOIvsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->B3SOIvsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->B3SOIvsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->B3SOIpclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->B3SOIpclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->B3SOIpclm);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIdrout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->B3SOIdrout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->B3SOIdrout);
	    Fatal_Flag = 1;
	}
      if ( model->B3SOIunitLengthGateSidewallJctCap > 0.0)
      {
	if (here->B3SOIdrainPerimeter < pParam->B3SOIweff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->B3SOIdrainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->B3SOIdrainPerimeter);
            here->B3SOIdrainPerimeter =pParam->B3SOIweff; 
	}
	if (here->B3SOIsourcePerimeter < pParam->B3SOIweff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->B3SOIsourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->B3SOIsourcePerimeter);
            here->B3SOIsourcePerimeter =pParam->B3SOIweff;
	}
      }
/* Check capacitance parameters */
        if (pParam->B3SOIclc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->B3SOIclc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->B3SOIclc);
	    Fatal_Flag = 1;
        } 

/* v2.2.3 */
        if (pParam->B3SOImoin < 5.0)
        {   fprintf(fplog, "Warning: Moin = %g is too small.\n",
                    pParam->B3SOImoin);
            printf("Warning: Moin = %g is too small.\n", pParam->B3SOImoin);
        }
        if (pParam->B3SOImoin > 25.0)
        {   fprintf(fplog, "Warning: Moin = %g is too large.\n",
                    pParam->B3SOImoin);
            printf("Warning: Moin = %g is too large.\n", pParam->B3SOImoin);
        }

/* v3.0 */
        if (model->B3SOImoinFD < 5.0)
        {   fprintf(fplog, "Warning: MoinFD = %g is too small.\n",
                    model->B3SOImoinFD);
            printf("Warning: MoinFD = %g is too small.\n", model->B3SOImoinFD);
        }


	if (model->B3SOIcapMod == 3) {
		if (pParam->B3SOIacde < 0.4)
		{  fprintf (fplog, "Warning: Acde = %g is too small.\n",
		   	    pParam->B3SOIacde);
 		   printf ("Warning: Acde = %g is too small.\n",
                    	    pParam->B3SOIacde);
		}
		if (pParam->B3SOIacde > 1.6)
		{  fprintf (fplog, "Warning: Acde = %g is too large.\n",
		   	    pParam->B3SOIacde);
 		   printf ("Warning: Acde = %g is too large.\n",
                    	    pParam->B3SOIacde);
		}
	}
/* v2.2.3 */
     
      if (model->B3SOIparamChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->B3SOIleff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->B3SOIleff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->B3SOIleff);
	}    
	
	if (pParam->B3SOIleffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->B3SOIleffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->B3SOIleffCV);
	}  
	
        if (pParam->B3SOIweff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->B3SOIweff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->B3SOIweff);
	}             
	
	if (pParam->B3SOIweffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->B3SOIweffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->B3SOIweffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->B3SOInlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->B3SOInlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->B3SOInlx);
        }
	if (model->B3SOItox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->B3SOItox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->B3SOItox);
        }

        if (pParam->B3SOInpeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->B3SOInpeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->B3SOInpeak);
	}
	else if (pParam->B3SOInpeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->B3SOInpeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->B3SOInpeak);
	}

	if (fabs(pParam->B3SOInsub) >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->B3SOInsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->B3SOInsub);
	}

	if ((pParam->B3SOIngate > 0.0) &&
	    (pParam->B3SOIngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->B3SOIngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->B3SOIngate);
	}
       
        if (pParam->B3SOIdvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->B3SOIdvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->B3SOIdvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->B3SOIw0 + pParam->B3SOIweff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->B3SOInfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->B3SOInfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->B3SOInfactor);
	}
	if (pParam->B3SOIcdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->B3SOIcdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->B3SOIcdsc);
	}
	if (pParam->B3SOIcdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->B3SOIcdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->B3SOIcdscd);
	}
/* Check DIBL parameters */
	if (pParam->B3SOIeta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->B3SOIeta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->B3SOIeta0); 
	}
	      
/* Check Abulk parameters */	    
        if (fabs(1.0e-6 / (pParam->B3SOIb1 + pParam->B3SOIweff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    

/* Check Saturation parameters */
     	if (pParam->B3SOIa2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->B3SOIa2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->B3SOIa2);
	    pParam->B3SOIa2 = 0.01;
	}
	else if (pParam->B3SOIa2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->B3SOIa2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->B3SOIa2);
	    pParam->B3SOIa2 = 1.0;
	    pParam->B3SOIa1 = 0.0;

	}

	if (pParam->B3SOIrdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->B3SOIrdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->B3SOIrdsw);
	    pParam->B3SOIrdsw = 0.0;
	    pParam->B3SOIrds0 = 0.0;
	}
	else if ((pParam->B3SOIrds0 > 0.0) && (pParam->B3SOIrds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->B3SOIrds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->B3SOIrds0);
	    pParam->B3SOIrds0 = 0.0;
	}
	 if (pParam->B3SOIvsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIvsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIvsattemp);
	}

	if (pParam->B3SOIpdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->B3SOIpdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->B3SOIpdibl1);
	}
	if (pParam->B3SOIpdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->B3SOIpdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->B3SOIpdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->B3SOIcgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIcgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIcgdo);
	    model->B3SOIcgdo = 0.0;
        }      
        if (model->B3SOIcgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIcgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIcgso);
	    model->B3SOIcgso = 0.0;
        }      
        if (model->B3SOIcgeo < 0.0)
	{   fprintf(fplog, "Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIcgeo);
	    printf("Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIcgeo);
	    model->B3SOIcgeo = 0.0;
        }

	if (model->B3SOIntun < 0.0)
	{   fprintf(fplog, "Warning: Ntun = %g is negative.\n",
		    model->B3SOIntun); 
	    printf("Warning: Ntun = %g is negative.\n", model->B3SOIntun); 
	}

	if (model->B3SOIndiode < 0.0)
	{   fprintf(fplog, "Warning: Ndiode = %g is negative.\n",
		    model->B3SOIndiode); 
	    printf("Warning: Ndiode = %g is negative.\n", model->B3SOIndiode); 
	}

	if (model->B3SOIisbjt < 0.0)
	{   fprintf(fplog, "Warning: Isbjt = %g is negative.\n",
		    model->B3SOIisbjt); 
	    printf("Warning: Isbjt = %g is negative.\n", model->B3SOIisbjt); 
	}

	if (model->B3SOIisdif < 0.0)
	{   fprintf(fplog, "Warning: Isdif = %g is negative.\n",
		    model->B3SOIisdif); 
	    printf("Warning: Isdif = %g is negative.\n", model->B3SOIisdif); 
	}

	if (model->B3SOIisrec < 0.0)
	{   fprintf(fplog, "Warning: Isrec = %g is negative.\n",
		    model->B3SOIisrec); 
	    printf("Warning: Isrec = %g is negative.\n", model->B3SOIisrec); 
	}

	if (model->B3SOIistun < 0.0)
	{   fprintf(fplog, "Warning: Istun = %g is negative.\n",
		    model->B3SOIistun); 
	    printf("Warning: Istun = %g is negative.\n", model->B3SOIistun); 
	}

	if (model->B3SOItt < 0.0)
	{   fprintf(fplog, "Warning: Tt = %g is negative.\n",
		    model->B3SOItt); 
	    printf("Warning: Tt = %g is negative.\n", model->B3SOItt); 
	}

	if (model->B3SOIcsdmin < 0.0)
	{   fprintf(fplog, "Warning: Csdmin = %g is negative.\n",
		    model->B3SOIcsdmin); 
	    printf("Warning: Csdmin = %g is negative.\n", model->B3SOIcsdmin); 
	}

	if (model->B3SOIcsdesw < 0.0)
	{   fprintf(fplog, "Warning: Csdesw = %g is negative.\n",
		    model->B3SOIcsdesw); 
	    printf("Warning: Csdesw = %g is negative.\n", model->B3SOIcsdesw); 
	}

	if (model->B3SOIasd < 0.0)
	{   fprintf(fplog, "Warning: Asd = %g should be within (0, 1).\n",
		    model->B3SOIasd); 
	    printf("Warning: Asd = %g should be within (0, 1).\n", model->B3SOIasd); 
	}

	if (model->B3SOIrth0 < 0.0)
	{   fprintf(fplog, "Warning: Rth0 = %g is negative.\n",
		    model->B3SOIrth0); 
	    printf("Warning: Rth0 = %g is negative.\n", model->B3SOIrth0); 
	}

	if (model->B3SOIcth0 < 0.0)
	{   fprintf(fplog, "Warning: Cth0 = %g is negative.\n",
		    model->B3SOIcth0); 
	    printf("Warning: Cth0 = %g is negative.\n", model->B3SOIcth0); 
	}

	if (model->B3SOIrbody < 0.0)
	{   fprintf(fplog, "Warning: Rbody = %g is negative.\n",
		    model->B3SOIrbody); 
	    printf("Warning: Rbody = %g is negative.\n", model->B3SOIrbody); 
	}

	if (model->B3SOIrbsh < 0.0)
	{   fprintf(fplog, "Warning: Rbsh = %g is negative.\n",
		    model->B3SOIrbsh); 
	    printf("Warning: Rbsh = %g is negative.\n", model->B3SOIrbsh); 
	}


/* v3.0 */
        if (pParam->B3SOInigc <= 0.0)
        {   fprintf(fplog, "Fatal: nigc = %g is non-positive.\n",
                    pParam->B3SOInigc);
            printf("Fatal: nigc = %g is non-positive.\n", pParam->B3SOInigc);
            Fatal_Flag = 1;
        }
        if (pParam->B3SOIpoxedge <= 0.0)
        {   fprintf(fplog, "Fatal: poxedge = %g is non-positive.\n",
                    pParam->B3SOIpoxedge);
            printf("Fatal: poxedge = %g is non-positive.\n", pParam->B3SOIpoxedge);
            Fatal_Flag = 1;
        }
        if (pParam->B3SOIpigcd <= 0.0)
        {   fprintf(fplog, "Fatal: pigcd = %g is non-positive.\n",
                    pParam->B3SOIpigcd);
            printf("Fatal: pigcd = %g is non-positive.\n", pParam->B3SOIpigcd);
            Fatal_Flag = 1;
        }


/* v2.2 release */
        if (model->B3SOIwth0 < 0.0)
        {   fprintf(fplog, "Warning: WTH0 = %g is negative.\n",
                    model->B3SOIwth0);
            printf("Warning:  Wth0 = %g is negative.\n", model->B3SOIwth0);
        }
        if (model->B3SOIrhalo < 0.0)
        {   fprintf(fplog, "Warning: RHALO = %g is negative.\n",
                    model->B3SOIrhalo);
            printf("Warning:  Rhalo = %g is negative.\n", model->B3SOIrhalo);
        }
        if (model->B3SOIntox < 0.0)
        {   fprintf(fplog, "Warning: NTOX = %g is negative.\n",
                    model->B3SOIntox);
            printf("Warning:  Ntox = %g is negative.\n", model->B3SOIntox);
        }
        if (model->B3SOItoxref < 0.0)
        {   fprintf(fplog, "Warning: TOXREF = %g is negative.\n",
                    model->B3SOItoxref);
            printf("Warning:  Toxref = %g is negative.\n", model->B3SOItoxref);
            Fatal_Flag = 1;
        }
        if (model->B3SOIebg < 0.0)
        {   fprintf(fplog, "Warning: EBG = %g is negative.\n",
                    model->B3SOIebg);
            printf("Warning:  Ebg = %g is negative.\n", model->B3SOIebg);
        }
        if (model->B3SOIvevb < 0.0)
        {   fprintf(fplog, "Warning: VEVB = %g is negative.\n",
                    model->B3SOIvevb);
            printf("Warning:  Vevb = %g is negative.\n", model->B3SOIvevb);
        }
        if (model->B3SOIalphaGB1 < 0.0)
        {   fprintf(fplog, "Warning: ALPHAGB1 = %g is negative.\n",
                    model->B3SOIalphaGB1);
            printf("Warning:  AlphaGB1 = %g is negative.\n", model->B3SOIalphaGB1);
        }
        if (model->B3SOIbetaGB1 < 0.0)
        {   fprintf(fplog, "Warning: BETAGB1 = %g is negative.\n",
                    model->B3SOIbetaGB1);
            printf("Warning:  BetaGB1 = %g is negative.\n", model->B3SOIbetaGB1);
        }
        if (model->B3SOIvgb1 < 0.0)
        {   fprintf(fplog, "Warning: VGB1 = %g is negative.\n",
                    model->B3SOIvgb1);
            printf("Warning:  Vgb1 = %g is negative.\n", model->B3SOIvgb1);
        }
        if (model->B3SOIvecb < 0.0)
        {   fprintf(fplog, "Warning: VECB = %g is negative.\n",
                    model->B3SOIvecb);
            printf("Warning:  Vecb = %g is negative.\n", model->B3SOIvecb);
        }
        if (model->B3SOIalphaGB2 < 0.0)
        {   fprintf(fplog, "Warning: ALPHAGB2 = %g is negative.\n",
                    model->B3SOIalphaGB2);
            printf("Warning:  AlphaGB2 = %g is negative.\n", model->B3SOIalphaGB2);
        }
        if (model->B3SOIbetaGB2 < 0.0)
        {   fprintf(fplog, "Warning: BETAGB2 = %g is negative.\n",
                    model->B3SOIbetaGB2);
            printf("Warning:  BetaGB2 = %g is negative.\n", model->B3SOIbetaGB2);
        }
        if (model->B3SOIvgb2 < 0.0)
        {   fprintf(fplog, "Warning: VGB2 = %g is negative.\n",
                    model->B3SOIvgb2);
            printf("Warning:  Vgb2 = %g is negative.\n", model->B3SOIvgb2);
        }
        if (model->B3SOItoxqm <= 0.0)
        {   fprintf(fplog, "Fatal: Toxqm = %g is not positive.\n",
                    model->B3SOItoxqm);
            printf("Fatal: Toxqm = %g is not positive.\n", model->B3SOItoxqm);
            Fatal_Flag = 1;
        }
        if (model->B3SOIvoxh < 0.0)
        {   fprintf(fplog, "Warning: Voxh = %g is negative.\n",
                    model->B3SOIvoxh);
            printf("Warning:  Voxh = %g is negative.\n", model->B3SOIvoxh);
        }
        if (model->B3SOIdeltavox <= 0.0)
        {   fprintf(fplog, "Fatal: Deltavox = %g is not positive.\n",
                    model->B3SOIdeltavox);
            printf("Fatal: Deltavox = %g is not positive.\n", model->B3SOIdeltavox);
        }


/* v2.0 release */
        if (model->B3SOIk1w1 < 0.0)                    
        {   fprintf(fplog, "Warning: K1W1 = %g is negative.\n",
                    model->B3SOIk1w1);
            printf("Warning:  K1w1 = %g is negative.\n", model->B3SOIk1w1);
        }
        if (model->B3SOIk1w2 < 0.0)
        {   fprintf(fplog, "Warning: K1W2 = %g is negative.\n",
                    model->B3SOIk1w2);
            printf("Warning:  K1w2 = %g is negative.\n", model->B3SOIk1w2);
        }
        if (model->B3SOIketas < 0.0)
        {   fprintf(fplog, "Warning: KETAS = %g is negative.\n",
                    model->B3SOIketas);
            printf("Warning:  Ketas = %g is negative.\n", model->B3SOIketas);
        }
        if (model->B3SOIdwbc < 0.0)
        {   fprintf(fplog, "Warning: DWBC = %g is negative.\n",
                    model->B3SOIdwbc);
            printf("Warning:  Dwbc = %g is negative.\n", model->B3SOIdwbc);
        }
        if (model->B3SOIbeta0 < 0.0)
        {   fprintf(fplog, "Warning: BETA0 = %g is negative.\n",
                    model->B3SOIbeta0);
            printf("Warning:  Beta0 = %g is negative.\n", model->B3SOIbeta0);
        }
        if (model->B3SOIbeta1 < 0.0)
        {   fprintf(fplog, "Warning: BETA1 = %g is negative.\n",
                    model->B3SOIbeta1);
            printf("Warning:  Beta1 = %g is negative.\n", model->B3SOIbeta1);
        }
        if (model->B3SOIbeta2 < 0.0)
        {   fprintf(fplog, "Warning: BETA2 = %g is negative.\n",
                    model->B3SOIbeta2);
            printf("Warning:  Beta2 = %g is negative.\n", model->B3SOIbeta2);
        }
        if (model->B3SOItii < 0.0)
        {   fprintf(fplog, "Warning: TII = %g is negative.\n",
                    model->B3SOItii);
            printf("Warning:  Tii = %g is negative.\n", model->B3SOItii);
        }
        if (model->B3SOIlii < 0.0)
        {   fprintf(fplog, "Warning: LII = %g is negative.\n",
                    model->B3SOIlii);
            printf("Warning:  Lii = %g is negative.\n", model->B3SOIlii);
        }
        if (model->B3SOIsii1 < 0.0)
        {   fprintf(fplog, "Warning: SII1 = %g is negative.\n",
                    model->B3SOIsii1);
            printf("Warning:  Sii1 = %g is negative.\n", model->B3SOIsii1);
        }
        if (model->B3SOIsii2 < 0.0)
        {   fprintf(fplog, "Warning: SII2 = %g is negative.\n", 
                    model->B3SOIsii2);
            printf("Warning:  Sii2 = %g is negative.\n", model->B3SOIsii1);
        }
        if (model->B3SOIsiid < 0.0)
        {   fprintf(fplog, "Warning: SIID = %g is negative.\n",
                    model->B3SOIsiid);
            printf("Warning:  Siid = %g is negative.\n", model->B3SOIsiid);
        }
        if (model->B3SOIfbjtii < 0.0)
        {   fprintf(fplog, "Warning: FBJTII = %g is negative.\n",
                    model->B3SOIfbjtii);
            printf("Warning:  fbjtii = %g is negative.\n", model->B3SOIfbjtii);
        }
        if (model->B3SOIvrec0 < 0.0)
        {   fprintf(fplog, "Warning: VREC0 = %g is negative.\n",
                    model->B3SOIvrec0);
            printf("Warning:  Vrec0 = %g is negative.\n", model->B3SOIvrec0);
        }
        if (model->B3SOIvtun0 < 0.0)
        {   fprintf(fplog, "Warning: VTUN0 = %g is negative.\n",
                    model->B3SOIvtun0);
            printf("Warning:  Vtun0 = %g is negative.\n", model->B3SOIvtun0);
        }
        if (model->B3SOInbjt < 0.0)
        {   fprintf(fplog, "Warning: NBJT = %g is negative.\n",
                    model->B3SOInbjt);
            printf("Warning:  Nbjt = %g is negative.\n", model->B3SOInbjt);
        }
        if (model->B3SOIaely < 0.0)
        {   fprintf(fplog, "Warning: AELY = %g is negative.\n",
                    model->B3SOIaely);
            printf("Warning:  Aely = %g is negative.\n", model->B3SOIaely);
        }
        if (model->B3SOIahli < 0.0)
        {   fprintf(fplog, "Warning: AHLI = %g is negative.\n",
                    model->B3SOIahli);
            printf("Warning:  Ahli = %g is negative.\n", model->B3SOIahli);
        }
        if (model->B3SOIrbody < 0.0)
        {   fprintf(fplog, "Warning: RBODY = %g is negative.\n",
                    model->B3SOIrbody);
            printf("Warning:  Rbody = %g is negative.\n", model->B3SOIrbody);
        }
        if (model->B3SOIrbsh < 0.0)
        {   fprintf(fplog, "Warning: RBSH = %g is negative.\n",
                    model->B3SOIrbsh);
            printf("Warning:  Rbsh = %g is negative.\n", model->B3SOIrbsh);
        }
        if (model->B3SOIntrecf < 0.0)
        {   fprintf(fplog, "Warning: NTRECF = %g is negative.\n",
                    model->B3SOIntrecf);
            printf("Warning:  Ntrecf = %g is negative.\n", model->B3SOIntrecf);
        }
        if (model->B3SOIntrecr < 0.0)
        {   fprintf(fplog, "Warning: NTRECR = %g is negative.\n",
                    model->B3SOIntrecr);
            printf("Warning:  Ntrecr = %g is negative.\n", model->B3SOIntrecr);
        }

/* v3.0 bug fix */
/*
        if (model->B3SOIndif < 0.0)
        {   fprintf(fplog, "Warning: NDIF = %g is negative.\n",
                    model->B3SOIndif);
            printf("Warning:  Ndif = %g is negative.\n", model->B3SOIndif);
        }
*/

        if (model->B3SOItcjswg < 0.0)
        {   fprintf(fplog, "Warning: TCJSWG = %g is negative.\n",
                    model->B3SOItcjswg);
            printf("Warning:  Tcjswg = %g is negative.\n", model->B3SOItcjswg);
        }
        if (model->B3SOItpbswg < 0.0)
        {   fprintf(fplog, "Warning: TPBSWG = %g is negative.\n",
                    model->B3SOItpbswg);
            printf("Warning:  Tpbswg = %g is negative.\n", model->B3SOItpbswg);
        }
        if ((model->B3SOIacde < 0.4) || (model->B3SOIacde > 1.6))
        {   fprintf(fplog, "Warning: ACDE = %g is out of range.\n",
                    model->B3SOIacde);
            printf("Warning:  Acde = %g is out of range.\n", model->B3SOIacde);
        }
        if ((model->B3SOImoin < 5.0)||(model->B3SOImoin > 25.0))
        {   fprintf(fplog, "Warning: MOIN = %g is out of range.\n",
                    model->B3SOImoin);
            printf("Warning:  Moin = %g is out of range.\n", model->B3SOImoin);
        }
        if (model->B3SOIdlbg < 0.0)
        {   fprintf(fplog, "Warning: DLBG = %g is negative.\n",
                    model->B3SOIdlbg);
            printf("Warning:  dlbg = %g is negative.\n", model->B3SOIdlbg);
        }


        if (model->B3SOIagidl < 0.0)
        {   fprintf(fplog, "Warning: AGIDL = %g is negative.\n",
                    model->B3SOIagidl);
            printf("Warning:  Agidl = %g is negative.\n", model->B3SOIagidl);
        }
        if (model->B3SOIbgidl < 0.0)
        {   fprintf(fplog, "Warning: BGIDL = %g is negative.\n",
                    model->B3SOIbgidl);
            printf("Warning:  Bgidl = %g is negative.\n", model->B3SOIbgidl);
        }
        if (model->B3SOIngidl < 0.0)
        {   fprintf(fplog, "Warning: NGIDL = %g is negative.\n",
                    model->B3SOIngidl);
            printf("Warning:  Ngidl = %g is negative.\n", model->B3SOIngidl);
        }
        if (model->B3SOIesatii < 0.0)
        {   fprintf(fplog, "Warning: Esatii = %g should be within positive.\n",
                    model->B3SOIesatii);
            printf("Warning: Esatii = %g should be within (0, 1).\n", model->B3SOIesatii);
        }


	if (model->B3SOIxj > model->B3SOItsi)
	{   fprintf(fplog, "Warning: Xj = %g is thicker than Tsi = %g.\n",
		    model->B3SOIxj, model->B3SOItsi); 
	    printf("Warning: Xj = %g is thicker than Tsi = %g.\n", 
		    model->B3SOIxj, model->B3SOItsi); 
	}

        if (model->B3SOIcapMod < 2)
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

