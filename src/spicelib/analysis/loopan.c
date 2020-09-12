/*******************************************************************************
 * Copyright 2020 Florian Ballenegger, Anamosic Ballenegger Design
 *******************************************************************************
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/acdefs.h"
#include "ngspice/loopdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/enh.h"
/* gtri - add - wbk - 12/19/90 - Add headers */ 
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk */
#endif


#define INIT_STATS() \
do { \
    startTime  = SPfrontEnd->IFseconds();       \
    startdTime = ckt->CKTstat->STATdecompTime;  \
    startsTime = ckt->CKTstat->STATsolveTime;   \
    startlTime = ckt->CKTstat->STATloadTime;    \
    startkTime = ckt->CKTstat->STATsyncTime;    \
} while(0)

#define UPDATE_STATS(analysis) \
do { \
    ckt->CKTcurrentAnalysis = analysis; \
    ckt->CKTstat->STATacTime += SPfrontEnd->IFseconds() - startTime; \
    ckt->CKTstat->STATacDecompTime += ckt->CKTstat->STATdecompTime - startdTime; \
    ckt->CKTstat->STATacSolveTime += ckt->CKTstat->STATsolveTime - startsTime; \
    ckt->CKTstat->STATacLoadTime += ckt->CKTstat->STATloadTime - startlTime; \
    ckt->CKTstat->STATacSyncTime += ckt->CKTstat->STATsyncTime - startkTime; \
} while(0)

extern SPICEdev **DEVices;

static IFcomplex
caddf(IFcomplex a, IFcomplex b)
{
    IFcomplex r;
    r.real = a.real + b.real;
    r.imag = a.imag + b.imag;
    return r;
}

static IFcomplex
csubf(IFcomplex a, IFcomplex b)
{
    IFcomplex r;
    r.real = a.real - b.real;
    r.imag = a.imag - b.imag;
    return r;
}

static IFcomplex
cmulf(IFcomplex a, IFcomplex b)
{
    IFcomplex r;
    r.real = a.real*b.real - a.imag*b.imag;
    r.imag = a.real*b.imag + a.imag*b.real;
    return r;
}

static IFcomplex
cdivf(IFcomplex a, IFcomplex b)
{
    IFcomplex r;
    double denom;
    denom = b.real*b.real + b.imag*b.imag;
    r.real = (a.real*b.real + a.imag*b.imag)/denom;
    r.imag = (a.imag*b.real - a.real*b.imag)/denom;
    return r;
}

int
LOOPpreset(CKTcircuit *ckt, JOB *anal)
{
    LOOPAN *job = (LOOPAN *) anal;
    GENinstance* inst;
    GENinstance* probesrc;
    GENmodel*    modvsrc;
    CKTnode *nodeinj, *node;
    IFdevice* dev;
    IFuid    eltUid;
    IFuid    modUid;
    IFvalue  ptemp;
    int      termidx;
    int      error;
    int      vtype;
        
    if(!job->LOOPrefNodeGiven)
       job->LOOPrefNode = NULL;
       
    if(job->LOOPportnameGiven || job->LOOPportnumGiven)
    {
       inst = CKTfndDev(ckt, job->LOOPprobeSrc);
       if (!inst || inst->GENmodPtr->GENmodType < 0) {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Loop probe source '%s' not in circuit",
                             job->LOOPprobeSrc);
        return E_NOTFOUND;
       }
       dev = &DEVices[inst->GENmodPtr->GENmodType]->DEVpublic;
       if (inst->GENmodPtr->GENmodType == CKTtypelook("Vsource"))
          printf("Consider to specify the whole Vsource '%s' as loop probe\n", job->LOOPprobeSrc);
			     
       if(job->LOOPportnameGiven) {
	   for(termidx=0;termidx<(*(dev->numNames));termidx++) {
	     if(0) printf("cmp %s with %s\n", job->LOOPportname, dev->termNames[termidx]);
	     if(strcasecmp(job->LOOPportname, dev->termNames[termidx])==0)
	       break;
	   }
       } else
	   termidx = job->LOOPportnum-1; /* LOOPportnum counts from 1, termidx counts from 0 */
       if(termidx<0 || termidx>=*(dev->numNames)) {
           SPfrontEnd->IFerrorf (ERR_WARNING, "No such terminal %d", termidx);
           return E_NOTFOUND;
       }
       /* now break the loop at terminal termidx */
       {
       char probename[32];
       sprintf(probename, "probe_%d", termidx);
       error = SPfrontEnd->IFnewUid (ckt, &eltUid, inst->GENname,
                        probename, UID_INSTANCE, NULL);
       }
       if(error) return(error);
       probesrc = CKTfndDev(ckt, eltUid);
       if (probesrc && probesrc->GENmodPtr->GENmodType >= 0)
       {
         /* probe already inserted, nothing to do except to free eltUid ? */
	 printf("Loop analysis: The probe was already in place\n");
       }
       else
       {
       printf("Loop analysis: Break the loop at device '%s' terminal '%s'\n", inst->GENname, dev->termNames[termidx]);
       node = CKTnum2nod(ckt, GENnode(inst)[termidx]);
       error = CKTmkVolt(ckt, &nodeinj, inst->GENname, "loopinj");
       if(error) return(error);
       error = SPfrontEnd->IFnewUid (ckt, &modUid, inst->GENname,
                        "probemod", UID_MODEL, NULL);
       if(error) return(error);
       vtype = CKTtypelook("Vsource");
       modvsrc = NULL;
       error = CKTmodCrt(ckt,vtype,&modvsrc, modUid);
       if(error) return(error);
       error = CKTcrtElt(ckt, modvsrc, &probesrc, eltUid);
       if(error) return(error);
       ptemp.rValue = 0;
       error = CKTpName("dc",&ptemp,ckt,vtype,"probe",&probesrc);
       if(error) return(error);
       /* LOOPdirection is set later */
       error = CKTbindNode(ckt,probesrc,job->LOOPdirection==2 ? 1 : 1,nodeinj);
       if(error) return(error);
       error = CKTbindNode(ckt,probesrc,job->LOOPdirection==2 ? 2 : 2 ,node);
       if(error) return(error);
       error = CKTbindNode(ckt,inst,termidx+1,nodeinj); /* bindNode counts from 1 ! */
       if(error) return(error);
       }
       job->LOOPprobeSrc = probesrc->GENname;
       job->LOOPportnameGiven = 0; /* don't mess if LOOPpreset is called a second time */
       job->LOOPportnumGiven = 0;  /* idem */
       
       job->LOOPinIV = LOOP_IV_UNSET;
       if(job->LOOPinSrcGiven) {
            GENinstance *insrc;
	    insrc = CKTfndDev(ckt, job->LOOPinSrc);
	    if (!insrc || insrc->GENmodPtr->GENmodType < 0) {
	        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Transfer function source %s not in circuit",
                             job->LOOPinSrc);
                return E_NOTFOUND;
	    }
	    if (insrc->GENmodPtr->GENmodType == CKTtypelook("Vsource"))
               job->LOOPinIV = LOOP_IV_VOLTAGE;
            else if (insrc->GENmodPtr->GENmodType == CKTtypelook("Isource")) {
               job->LOOPinIV = LOOP_IV_CURRENT;
            }
       }
       
       job->LOOPoutIV = LOOP_IV_UNSET;
       if(job->LOOPoutSrcGiven) {
	    GENinstance *outsrc;
	    outsrc = CKTfndDev(ckt, job->LOOPoutSrc);
	    if (!outsrc || outsrc->GENmodPtr->GENmodType < 0) {
	        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Transfer function source %s not in circuit",
                             job->LOOPoutSrc);
		
                return E_NOTFOUND;
	    }
	    if (outsrc->GENmodPtr->GENmodType == CKTtypelook("Vsource"))
               job->LOOPoutIV = LOOP_IV_CURRENT;
            else if (outsrc->GENmodPtr->GENmodType == CKTtypelook("Isource")) {
	       /* weird user, but let's kindly reformulate for him */
               job->LOOPinIV = LOOP_IV_VOLTAGE;
	       job->LOOPoutPos = CKTnum2nod(ckt, GENnode(outsrc)[0]);
	       job->LOOPoutNeg = CKTnum2nod(ckt, GENnode(outsrc)[1]);
	       job->LOOPoutPosGiven = 1;
	       job->LOOPoutNegGiven = 1;
	       job->LOOPoutSrcGiven = 0;
            }
       } else if (job->LOOPoutPosGiven) {
	   job->LOOPoutIV = LOOP_IV_VOLTAGE;
       }
       
    }
    return OK;
}

enum {
   LOOP_CURVE_T = 0,
   LOOP_CURVE_D,
   LOOP_CURVE_H,
   LOOP_CURVE_HINF,
#if 1   
   LOOP_CURVE_TN,
   LOOP_CURVE_DN,
   LOOP_CURVE_H0,
   LOOP_CURVE_D0,
   LOOP_CURVE_DP,
   LOOP_CURVE_TP,
#endif
   /* insert new curves here */
   LOOP_NCURVES
} eLOOPcurves;

int
LOOPan(CKTcircuit *ckt, int restart)
{
    LOOPAN *job = (LOOPAN *) ckt->CKTcurJob;
    
    double freq;
    double freqTol; /* tolerence parameter for finding final frequency */
    double startdTime;
    double startsTime;
    double startlTime;
    double startkTime;
    double startTime;
    
    IFcomplex Vy[3],Iy[3];
    /*double iVy[3],iIy[3];*/
    IFcomplex Uout[3];
    double D, iD, T, iT;
    double phasemargin, gainmargin, maxgain, ugf, ipf;
    double gainsq, phase, curTr, pfreq;
    
    int error;
    int numNames;
    IFuid *nameList;  /* va: tmalloc'ed list of names */
    IFuid freqUid;
    static runDesc *acPlot = NULL;
    runDesc *plot = NULL;
    GENinstance *inst = NULL;
    int size, i;
    int branch_probe, branch_vin, branch_iout;
    int inPosNode, inNegNode;
    int dir;
    
    static char* plot_curves[] = {
       "T", /* loop gain */
       "D", /* discrepency factor D = T/(1+T) */
       "H", /* transfert function */
       "Hinf", /* transfert function if loop gain was infinite */
       "Tn", /* "output-nulled" loop gain*/
       "Dn", /* 1/(1+Tn) */
       "H0", /* forward path transfert function with zero loop gain */
       "D0",
       "Dp",
       "Tp",
    };


    if(job->LOOPinSrcGiven) {
        if(job->LOOPinIV == LOOP_IV_CURRENT) {
	   inst = CKTfndDev(ckt, job->LOOPinSrc);
	   inPosNode = GENnode(inst)[0];
	   inNegNode = GENnode(inst)[1];
	} else if (job->LOOPinIV == LOOP_IV_VOLTAGE) {
	   branch_vin = CKTfndBranch(ckt, job->LOOPinSrc);
	   #if 0
	   /* not needed but set anyway */
	   inst = CKTfndDev(ckt, job->LOOPinSrc);
	   inPosNode = GENnode(inst)[0];
	   inNegNode = GENnode(inst)[1];
	   #endif
	}
    }
    if(job->LOOPoutSrcGiven) {
        if(job->LOOPoutIV == LOOP_IV_CURRENT)
	   branch_iout = CKTfndBranch(ckt, job->LOOPoutSrc);
    }
    if(job->LOOPdirection<0 || job->LOOPdirection >= 2)
      dir=1; /* inverse direction */
    else
      dir=0; /* normal direction for 0 and 1 */
    inst = CKTfndDev(ckt, job->LOOPprobeSrc);
    
    if (!inst || inst->GENmodPtr->GENmodType < 0) {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Loop probe source %s not in circuit",
                             job->LOOPprobeSrc);
        return E_NOTFOUND;
    }
    if (inst->GENmodPtr->GENmodType != CKTtypelook("Vsource"))
    {
        SPfrontEnd->IFerrorf (ERR_WARNING,
                             "Loop probe source %s not of proper type",
                             job->LOOPprobeSrc);
        return E_NOTFOUND;
    }
    if(0) printf("LOOPan Vy at %s\n", CKTnodName(ckt,GENnode(inst)[dir]));
    branch_probe = CKTfndBranch(ckt, job->LOOPprobeSrc);
    phasemargin = 180;
    gainmargin = strtod("NaN", NULL);
    phase = gainmargin; /* NaN */
    pfreq = gainmargin; /* NaN */
    ugf = gainmargin; /* NaN */
    ipf = gainmargin; /* NaN */
    maxgain = 0;
    
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

    /* Tell the beginPlot routine what mode we're in */
    g_ipc.anal_type = IPC_ANAL_AC;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;
    g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif

    /* start at beginning */
    if (job->LOOPsaveFreq == 0 || restart) {
        if (job->LOOPnumSteps < 1)
            job->LOOPnumSteps = 1;

        switch (job->LOOPstepType) {

        case DECADE:
            if (job->LOOPstartFreq <= 0) {
                fprintf(stderr, "ERROR: LOOP startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->LOOPfreqDelta =
                exp(log(10.0)/job->LOOPnumSteps);
            break;
        case OCTAVE:
            if (job->LOOPstartFreq <= 0) {
                fprintf(stderr, "ERROR: LOOP startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->LOOPfreqDelta =
                exp(log(2.0)/job->LOOPnumSteps);
            break;
        case LINEAR:
            if (job->LOOPnumSteps-1 > 1)
                job->LOOPfreqDelta =
                    (job->LOOPstopFreq -
                     job->LOOPstartFreq) /
                    (job->LOOPnumSteps - 1);
            else
            /* Patch from: Richard McRoberts
            * This patch is for a rather pathological case:
            * a linear step with only one point */
                job->LOOPfreqDelta = 0;
            break;
        default:
            return(E_BADPARM);
    }
#ifdef XSPICE
/* gtri - begin - wbk - Call EVTop if event-driven instances exist */

    if(ckt->evt->counts.num_insts != 0) {
        error = EVTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter,
            MIF_TRUE);
        EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
        EVTop_save(ckt, MIF_TRUE, 0.0);
    }
    else 
#endif 
    /* If no event-driven instances, do what SPICE normally does */
    if (!ckt->CKTnoopac) { /* skip OP if option NOOPAC is set and circuit is linear */
        error = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);

        if(error){
            fprintf(stdout,"\nLOOP operating point failed -\n");
            CKTncDump(ckt);
            return(error);
        }
    }
    else
        fprintf(stdout,"\n Linear circuit, option noopac given: no OP analysis\n");
		
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

    /* Send the operating point results for Mspice compatibility */
    if(g_ipc.enabled) 
    {
        /* Call CKTnames to get names of nodes/branches used by 
            BeginPlot */
        /* Probably should free nameList after this block since 
            called again... */
        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);

        /* We have to do a beginPlot here since the data to return is
         * different for the DCOP than it is for the AC analysis.  
         * Moreover the begin plot has not even been done yet at this 
         * point... 
         */
        SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                   ckt->CKTcurJob->JOBname,
                                   NULL, IF_REAL,
                                   numNames, nameList, IF_REAL,
                                   &acPlot);
        txfree(nameList);

        ipc_send_dcop_prefix();
        CKTdump(ckt, 0.0, acPlot);
        ipc_send_dcop_suffix();

        SPfrontEnd->OUTendPlot (acPlot);
    }
/* gtri - end - wbk */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        error = CKTload(ckt);
        if(error) return(error);

        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);
	
	#if 0
	printf("check: branch probe is %s\n", nameList[branch_probe-1]);
	printf("check: out pos node is %s\n", nameList[job->LOOPoutPos->number-1]);
	printf("check: out neg node is %s\n", nameList[job->LOOPoutNeg->number-1]);
	printf("check: in branch node is %s\n", nameList[branch_vin-1]);
	#endif
	
	if (ckt->CKTkeepOpInfo) {
	    /* Dump operating point. */
            error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                               "LOOP Operating Point",
                                               NULL, IF_REAL,
                                               numNames, nameList, IF_REAL,
                                               &plot);
	    if(error) return(error);
	    CKTdump(ckt, 0.0, plot);
	    SPfrontEnd->OUTendPlot (plot);
	    plot = NULL;
	}

        /* setup for the real loop output plot */
        numNames = 2; /* T and D by default */
	if(1) /*  yet implemented */
	if(job->LOOPinSrcGiven)
        if(job->LOOPoutSrcGiven || (job->LOOPoutPosGiven && job->LOOPoutNegGiven))
            numNames = LOOP_NCURVES; /* all GFT functions */
            
        nameList = TMALLOC(IFuid, numNames);
        for(int i=0;i<numNames;i++)
        {
            SPfrontEnd->IFnewUid (ckt, &nameList[i], job->LOOPname, plot_curves[i], UID_OTHER, NULL);
        }
    
	SPfrontEnd->IFnewUid (ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           ckt->CKTcurJob->JOBname,
                                           freqUid, IF_REAL,
                                           numNames, nameList, IF_COMPLEX,
                                           &acPlot);
	if(error) return(error);
        if (job->LOOPstepType != LINEAR) {
	    SPfrontEnd->OUTattributes (acPlot, NULL, OUT_SCALE_LOG, NULL);
	}
	
	
	
        freq = job->LOOPstartFreq;

    } else {    /* continue previous analysis */
        freq = job->LOOPsaveFreq;
        job->LOOPsaveFreq = 0; /* clear the 'old' frequency */
	/* fix resume? saj, indeed !*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &acPlot);
	/* saj*/    
    }
        
    switch (job->LOOPstepType) {
    case DECADE:
    case OCTAVE:
        freqTol = job->LOOPfreqDelta *
            job->LOOPstopFreq * ckt->CKTreltol;
        break;
    case LINEAR:
        freqTol = job->LOOPfreqDelta * ckt->CKTreltol;
        break;
    default:
        return(E_BADPARM);
    }


#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Set anal_init and anal_type */

    g_mif_info.circuit.anal_init = MIF_TRUE;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_AC;

/* gtri - end - wbk */
#endif

    INIT_STATS();

    ckt->CKTcurrentAnalysis = DOING_AC;

    /* main loop through all scheduled frequencies */
    while (freq <= job->LOOPstopFreq + freqTol) {
        if(SPfrontEnd->IFpauseTest()) {
            /* user asked us to pause via an interrupt */
            job->LOOPsaveFreq = freq;
            return(E_PAUSE);
        }
        ckt->CKTomega = 2.0 * M_PI *freq;

        /* Update opertating point, if variable 'hertz' is given */
        if (ckt->CKTvarHertz) {
#ifdef XSPICE
            /* Call EVTop if event-driven instances exist */

            if(ckt->evt->counts.num_insts != 0) {
                error = EVTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter,
                    MIF_TRUE);
                EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
                EVTop_save(ckt, MIF_TRUE, 0.0);
            }
            else 
#endif 
// If no event-driven instances, do what SPICE normally does
                error = CKTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter);

            if(error){
                fprintf(stdout,"\nLOOP operating point failed -\n");
                CKTncDump(ckt);
                return(error);
            } 
            ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
            error = CKTload(ckt);
            if(error) return(error);
        }

        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODEAC;
	
	/************* Main part starts here ********************/
	size = SMPmatSize(ckt->CKTmatrix);
	for(int simno=0;simno<(numNames>2 ? 3 : 2);simno++)
	{
	  for (i=0; i<=size; i++)
	  {
	    ckt->CKTrhs[i] = 0.0;
	    ckt->CKTirhs[i] = 0.0;
	  }
	  error = CKTacLoad(ckt);
	  /* clear any AC stimulus set by acLoad */
	  for (i=1; i<=size; i++)
	  {
	    ckt->CKTrhs[i] = 0.0;
	    ckt->CKTirhs[i] = 0.0;
	  }
	  
	  switch(simno)
	  {
	  case 1:
	    /* AC current src magn=1 from ground to Vx */
	    ckt->CKTrhs[GENnode(inst)[1-dir]] -= 1;
	    if(job->LOOPrefNode)
	      ckt->CKTrhs[job->LOOPrefNode->number] += 1;
	    else
              ckt->CKTrhs[0] += 1;
	    break;
	  case 0: 
	    /* AC voltage src magn=1 accross probe */
	    ckt->CKTrhs[branch_probe] += 1;
	    break;
	  case 2: 
	    /* AC voltage at input source */
	    if(job->LOOPinIV==LOOP_IV_VOLTAGE) {
	        ckt->CKTrhs[branch_vin] += 1;
	    } else if (job->LOOPinIV==LOOP_IV_CURRENT) {
	        ckt->CKTrhs[inPosNode] += 1;
                ckt->CKTrhs[inNegNode] -= 1;
	    }
	    break;  
	  default:
	    printf("Don't know what to simulate for this index %d\n",simno);
	    return(E_BADPARM);
	  }
	
	
	  /*error = NIacIter(ckt);*/
	  error = NIdIter(ckt);
          if (error) {
            UPDATE_STATS(DOING_AC);
            return(error);
          }
	  if(job->LOOPrefNode) {
	    Vy[simno].real = ckt->CKTrhsOld[GENnode(inst)[dir]] - ckt->CKTrhsOld[job->LOOPrefNode->number];
	    Vy[simno].imag = ckt->CKTirhsOld[GENnode(inst)[dir]] - ckt->CKTirhsOld[job->LOOPrefNode->number];
	  } else {
	    Vy[simno].real = ckt->CKTrhsOld[GENnode(inst)[dir]];
	    Vy[simno].imag = ckt->CKTirhsOld[GENnode(inst)[dir]];
	  }
	  Iy[simno].real = ckt->CKTrhsOld[branch_probe];
	  Iy[simno].imag = ckt->CKTirhsOld[branch_probe];
	  if(job->LOOPoutIV == LOOP_IV_CURRENT) {
	     Uout[simno].real = ckt->CKTrhsOld[branch_iout];
	     Uout[simno].imag = ckt->CKTirhsOld[branch_iout];
	  } else if(job->LOOPoutIV == LOOP_IV_VOLTAGE) {
	     Uout[simno].real = ckt->CKTrhsOld[job->LOOPoutPos->number] -
	                          ckt->CKTrhsOld[job->LOOPoutNeg->number];
	     Uout[simno].imag = ckt->CKTirhsOld[job->LOOPoutPos->number] -
	                          ckt->CKTirhsOld[job->LOOPoutNeg->number];
	  }
	  
	  if(0) printf("|Vy| = %g, |Iy|=%g\n", sqrt(Vy[simno].real*Vy[simno].real+Vy[simno].imag*Vy[simno].imag), sqrt(Iy[simno].real*Iy[simno].real+Iy[simno].imag*Iy[simno].imag));
	}
	
	/* V(y)@1*I(Viy)@3-V(y)@3*I(Viy)@1 */
	{
	double ngainsq, nphase;
	
	D = Vy[0].real*Iy[1].real - Vy[0].imag*Iy[1].imag - Vy[1].real*Iy[0].real + Vy[1].imag*Iy[0].imag;
	iD = Vy[0].real*Iy[1].imag + Vy[0].imag*Iy[1].real - Vy[1].real*Iy[0].imag - Vy[1].imag*Iy[0].real;
	/*T = (D-iD*iD/(1-D))/(1-D+iD/(1-D));*/
	T = (D*(1-D)-iD*iD)/((1-D)*(1-D) + iD*iD);
	iT = iD/((1-D)*(1-D) + iD*iD);
	nphase = 180*atan2(-iD,-D)/M_PI;
	ngainsq = T*T + iT*iT;
	/* search for phase margin */
	if((ngainsq-1)*(gainsq-1)<0)
	{ /* gain cross 1 - interpolate lin/log for more accurate results */
		double crossphase, ugflog;
		crossphase = (log(gainsq)*nphase -  
		  log(ngainsq)*phase)/(log(gainsq)-log(ngainsq));
		if(fabs(crossphase)<phasemargin)
		    phasemargin = fabs(crossphase);
		ugflog = (log(gainsq)*log(freq) -  
		  log(ngainsq)*log(pfreq))/(log(gainsq)-log(ngainsq));
	        if(isnan(ugf))
		  ugf = exp(ugflog);
		/* idea: binary search/sim + for frequency where gain=1 ? */
        }
	if(ngainsq>=1)
	if(fabs(nphase)<phasemargin)
		    phasemargin = fabs(nphase);
		    
        /* search for gain margin */
        if(T<0 && curTr<0)
	if(nphase * phase <= 0)
	{
	     /* right quandrant and cross phase=0 */
	     double gmsqlog;
	     gmsqlog = (phase*log(ngainsq)-nphase*log(gainsq))/(phase-nphase);
	     if(0) printf("phase cross at %g, ngainsq=%g gainsq=%g qmsqlog=%g\n", freq, ngainsq, gainsq, exp(gmsqlog));
	     if(isnan(gainmargin) || (10*gmsqlog*M_LOG10E > gainmargin)) {
	       double ipflog;
	       gainmargin = 10*gmsqlog*M_LOG10E;
	       ipflog = (phase*log(freq)-nphase*log(pfreq))/(phase-nphase);
	       ipf = exp(ipflog);
	     }
	}
	
	gainsq=ngainsq;
	phase=nphase;
	curTr = T; /* remember real part only */
	if(gainsq > maxgain)
	  maxgain = gainsq;
	if(0) printf("f=%g |T| = %g, |D|=%g, phase=%g\n", freq, sqrt(T*T+iT*iT), sqrt(D*D+iD*iD), 180*atan2(-iD,-D)/M_PI);
	}
	
	
	/************* End of main part ********************/

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode&ACSEN) ){
            long save;
            int save1;

            save = ckt->CKTmode;
            ckt->CKTmode=(ckt->CKTmode&MODEUIC)|MODEDCOP|MODEINITSMSIG;
            save1 = ckt->CKTsenInfo->SENmode;
            ckt->CKTsenInfo->SENmode = ACSEN;
            if (freq == job->LOOPstartFreq) {
                ckt->CKTsenInfo->SENacpertflag = 1;
            }
            else{
                ckt->CKTsenInfo->SENacpertflag = 0;
            }
            error = CKTsenAC(ckt);
            if (error)
                return (error);
            ckt->CKTmode = save;
            ckt->CKTsenInfo->SENmode = save1;
        }
#endif

        /* error = CKTacDump(ckt,freq,acPlot); */
	{
	IFvalue  freqData;
	IFvalue  valueData;
	IFcomplex cval, cA;
	IFcomplex data[LOOP_NCURVES];
	freqData.rValue = freq;
	valueData.v.numValue = numNames;
	valueData.v.vec.cVec = data;
	data[LOOP_CURVE_T].real = T;
	data[LOOP_CURVE_T].imag = iT;
	data[LOOP_CURVE_D].real = D;
	data[LOOP_CURVE_D].imag = iD;
	if(numNames >= LOOP_CURVE_H) {
	data[LOOP_CURVE_H] = Uout[2];
	cA.real=0;
	cA.imag=0;
	for(int i=0;i<3;i++)
	  cA = caddf(cA, cmulf(Uout[i],csubf(cmulf(Vy[(i+1)%3],Iy[(i+2)%3]),cmulf(Vy[(i+2)%3],Iy[(i+1)%3]))));
	data[LOOP_CURVE_HINF] = cdivf(cA,data[LOOP_CURVE_D]);
	data[LOOP_CURVE_DN] = cdivf(Uout[2],cA);
	data[LOOP_CURVE_TN].real = (data[LOOP_CURVE_DN].real*(1-data[LOOP_CURVE_DN].real)-data[LOOP_CURVE_DN].imag*data[LOOP_CURVE_DN].imag)/((1-data[LOOP_CURVE_DN].real)*(1-data[LOOP_CURVE_DN].real) + data[LOOP_CURVE_DN].imag*data[LOOP_CURVE_DN].imag);
	data[LOOP_CURVE_TN].imag = data[LOOP_CURVE_DN].imag/((1-data[LOOP_CURVE_DN].real)*(1-data[LOOP_CURVE_DN].real) + data[LOOP_CURVE_DN].imag*data[LOOP_CURVE_DN].imag);
	data[LOOP_CURVE_D0].real = 1 - data[LOOP_CURVE_D].real;
	data[LOOP_CURVE_D0].imag = -data[LOOP_CURVE_D].imag;
	data[LOOP_CURVE_H0] = cdivf(csubf(data[LOOP_CURVE_H],cA),data[LOOP_CURVE_D0]);
	data[LOOP_CURVE_DP] = cmulf(data[LOOP_CURVE_D],data[LOOP_CURVE_DN]);
	data[LOOP_CURVE_TP] = cdivf(data[LOOP_CURVE_DP],csubf((IFcomplex) {1,0}, data[LOOP_CURVE_DP]));
	 /* A = Vo.2*(Vy.0*IViy.1-Vy.1*IViy.0)+Vo.0*(Vy.1*IViy.2-Vy.2*IViy.1)+Vo.1*(Vy.2*IViy.0-Vy.0*IViy.2) */
	 /* A = Vo[2]*(Vy[0]*Iy[1]-Vy[1]*Iy[0])+Vo[0]*(Vy[1]*Iy[2]-Vy[2]*Iy[1])+Vo[1]*(Vy[2]*Iy[0]-Vy[0]*Iy[2])*/
	}
#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
        if(g_ipc.enabled)
            ipc_send_data_prefix(freq);
#endif
	SPfrontEnd->OUTpData (acPlot, &freqData, &valueData);
	}
#ifdef XSPICE
        if(g_ipc.enabled)
            ipc_send_data_suffix();

/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#endif
        
	if (error) {
	    UPDATE_STATS(DOING_AC);
 	    return(error);
 	}

        /*  increment frequency */
        pfreq = freq;
        switch (job->LOOPstepType) {
        case DECADE:
        case OCTAVE:

/* inserted again 14.12.2001  */
#ifdef HAS_PROGREP
            {
                double endfreq   = job->LOOPstopFreq;
                double startfreq = job->LOOPstartFreq;
                endfreq   = log(endfreq);
                if (startfreq == 0.0)
                    startfreq = 1e-12;
                startfreq = log(startfreq);

                if (freq > 0.0)
                    SetAnalyse( "loop", (int)((log(freq)-startfreq) * 1000.0 / (endfreq-startfreq)));
            }
#endif

            freq *= job->LOOPfreqDelta;
            if (job->LOOPfreqDelta == 1) goto endsweep;
        break;
        case LINEAR:

#ifdef HAS_PROGREP
			 {
				 double endfreq   = job->LOOPstopFreq;
				 double startfreq = job->LOOPstartFreq;
				 SetAnalyse( "loop", (int)((freq - startfreq)* 1000.0 / (endfreq-startfreq)));
			 }
#endif

            freq += job->LOOPfreqDelta;
            if (job->LOOPfreqDelta == 0) goto endsweep;
            break;
        default:
            return(E_INTERN);
    
        }

    }
endsweep:
    printf("Loop analysis"); if(job->LOOPname) printf(" '%s'",job->LOOPname); printf(": ");
    if(maxgain>=1.0)
      printf("phase margin is %g [degrees]\n", phasemargin);
    else
      printf("no phase margin detected as loop gain is lower than 1\n");
    
    printf("Loop analysis"); if(job->LOOPname) printf(" '%s'",job->LOOPname); printf(": ");
    if(!isnan(gainmargin))
      printf("gain margin is %g [dB] at f=%g [Hz]\n", gainmargin, ipf);
    else
      printf("no gain margin detected as phase do not reach 0 degrees\n");
    
    printf("Loop analysis"); if(job->LOOPname) printf(" '%s'",job->LOOPname); printf(": ");
    if(!isnan(ugf))
      printf("unit gain frequency is %g [Hz]\n", ugf);
    else
      printf("loop gain do not cross 1\n");
    
    if(maxgain<1) {
      printf("Loop analysis"); if(job->LOOPname) printf(" '%s'",job->LOOPname); printf(": ");
      printf("hint: try to reverce the loop direction\n");
    }
        
    SPfrontEnd->OUTendPlot (acPlot);
    tfree(nameList);		
    acPlot = NULL;
    UPDATE_STATS(0);
    return(0);
}
