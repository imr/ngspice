/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified 2001: AlansFixes
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "acdefs.h"
#include "devdefs.h"
#include "sperror.h"

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add headers */ 
#include "mif.h"
#include "evtproto.h"
#include "ipctiein.h"
/* gtri - end - wbk */
#endif

#ifdef HAS_WINDOWS
void SetAnalyse( char * Analyse, int Percent);
#endif


int
ACan(CKTcircuit *ckt, int restart)
{

    double freq;
    double freqTol; /* tolerence parameter for finding final frequency */
    double startdTime;
    double startsTime;
    double startlTime;
    double startcTime;
    double startkTime;
    double startTime;
    int error;
    int numNames;
    IFuid *nameList;  /* va: tmalloc'ed list of names */
    IFuid freqUid;
    static void *acPlot = NULL;
    void *plot = NULL;


#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

    /* Tell the beginPlot routine what mode we're in */
    g_ipc.anal_type = IPC_ANAL_AC;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;
    g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif


    if(((ACAN*)ckt->CKTcurJob)->ACsaveFreq == 0 || restart) { 
        /* start at beginning */

	if (((ACAN*)ckt->CKTcurJob)->ACnumberSteps < 1)
	    ((ACAN*)ckt->CKTcurJob)->ACnumberSteps = 1;

        switch(((ACAN*)ckt->CKTcurJob)->ACstepType) {

        case DECADE:
            ((ACAN*)ckt->CKTcurJob)->ACfreqDelta =
                    exp(log(10.0)/((ACAN*)ckt->CKTcurJob)->ACnumberSteps);
            break;
        case OCTAVE:
            ((ACAN*)ckt->CKTcurJob)->ACfreqDelta =
                    exp(log(2.0)/((ACAN*)ckt->CKTcurJob)->ACnumberSteps);
            break;
        case LINEAR:
	    if (((ACAN*)ckt->CKTcurJob)->ACnumberSteps-1 > 1)
		((ACAN*)ckt->CKTcurJob)->ACfreqDelta =
                    (((ACAN*)ckt->CKTcurJob)->ACstopFreq -
                    ((ACAN*)ckt->CKTcurJob)->ACstartFreq)/
                    (((ACAN*)ckt->CKTcurJob)->ACnumberSteps-1);
	    else
		/* Patch from: Richard McRoberts
		 * This patch is for a rather pathological case:
		 * a linear step with only one point */
		((ACAN*)ckt->CKTcurJob)->ACfreqDelta = 0; 
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

            error = CKTop(ckt,
                (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                ckt->CKTdcMaxIter);

        if(error){
        	 fprintf(stdout,"\nAC operating point failed -\n");
           CKTncDump(ckt);
           return(error);
        	} 
		
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
            (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,(void*)ckt->CKTcurJob,
                ckt->CKTcurJob->JOBname,(IFuid)NULL,IF_REAL,numNames,nameList,
                IF_REAL,&acPlot);
	    tfree(nameList);

            ipc_send_dcop_prefix();
            CKTdump(ckt,(double)0,acPlot);
            ipc_send_dcop_suffix();

            (*(SPfrontEnd->OUTendPlot))(acPlot);
        }

/* gtri - end - wbk */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        error = CKTload(ckt);
        if(error) return(error);

        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);

	if (ckt->CKTkeepOpInfo) {
	    /* Dump operating point. */
	    error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
		(void*)ckt->CKTcurJob, "AC Operating Point",
		(IFuid)NULL,IF_REAL,numNames,nameList, IF_REAL,&plot);
	    if(error) return(error);
	    CKTdump(ckt,(double)0,plot);
	    (*(SPfrontEnd->OUTendPlot))(plot);
	    plot = NULL;
	}

        (*(SPfrontEnd->IFnewUid))((void *)ckt,&freqUid,(IFuid)NULL,
                "frequency", UID_OTHER,(void **)NULL);
        error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
		(void*)ckt->CKTcurJob,
                ckt->CKTcurJob->JOBname,freqUid,IF_REAL,numNames,nameList,
                IF_COMPLEX,&acPlot);
	tfree(nameList);		
	if(error) return(error);

        if (((ACAN*)ckt->CKTcurJob)->ACstepType != LINEAR) {
	    (*(SPfrontEnd->OUTattributes))((void *)acPlot,NULL,
		    OUT_SCALE_LOG, NULL);
	}
        freq = ((ACAN*)ckt->CKTcurJob)->ACstartFreq;

    } else {    /* continue previous analysis */
        freq = ((ACAN*)ckt->CKTcurJob)->ACsaveFreq;
        ((ACAN*)ckt->CKTcurJob)->ACsaveFreq = 0; /* clear the 'old' frequency */
	/* fix resume? saj*/
	error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
		(void*)ckt->CKTcurJob,
                ckt->CKTcurJob->JOBname,freqUid,IF_REAL,numNames,nameList,
                IF_COMPLEX,&acPlot);
	/* saj*/    
    }
        
    switch(((ACAN*)ckt->CKTcurJob)->ACstepType) {
    case DECADE:
    case OCTAVE:
        freqTol = ((ACAN*)ckt->CKTcurJob)->ACfreqDelta * 
                ((ACAN*)ckt->CKTcurJob)->ACstopFreq * ckt->CKTreltol;
        break;
    case LINEAR:
        freqTol = ((ACAN*)ckt->CKTcurJob)->ACfreqDelta * ckt->CKTreltol;
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


    startTime  = SPfrontEnd->IFseconds();
    startdTime = ckt->CKTstat->STATdecompTime;
    startsTime = ckt->CKTstat->STATsolveTime;
    startlTime = ckt->CKTstat->STATloadTime;
    startcTime = ckt->CKTstat->STATcombineTime;
    startkTime = ckt->CKTstat->STATsyncTime;
    while(freq <= ((ACAN*)ckt->CKTcurJob)->ACstopFreq+freqTol) {

        if( (*(SPfrontEnd->IFpauseTest))() ) { 
            /* user asked us to pause via an interrupt */
            ((ACAN*)ckt->CKTcurJob)->ACsaveFreq = freq;
            return(E_PAUSE);
        }
        ckt->CKTomega = 2.0 * M_PI *freq;
        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODEAC;

        error = NIacIter(ckt);
        if (error) {
	    ckt->CKTcurrentAnalysis = DOING_AC;
	    ckt->CKTstat->STATacTime += SPfrontEnd->IFseconds() - startTime;
	    ckt->CKTstat->STATacDecompTime += ckt->CKTstat->STATdecompTime -
		    startdTime;
	    ckt->CKTstat->STATacSolveTime += ckt->CKTstat->STATsolveTime -
 	    startsTime;
	    ckt->CKTstat->STATacLoadTime += ckt->CKTstat->STATloadTime -
		    startlTime;
 	    ckt->CKTstat->STATacCombTime += ckt->CKTstat->STATcombineTime -
 		    startcTime;
	    ckt->CKTstat->STATacSyncTime += ckt->CKTstat->STATsyncTime -
		    startkTime;
	    return(error);
	}
  


#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode&ACSEN) ){

            save = ckt->CKTmode;
            ckt->CKTmode=(ckt->CKTmode&MODEUIC)|MODEDCOP|MODEINITSMSIG;
            save1 = ckt->CKTsenInfo->SENmode;
            ckt->CKTsenInfo->SENmode = ACSEN;
            if(freq == ((ACAN*)ckt->CKTcurJob)->ACstartFreq){
                ckt->CKTsenInfo->SENacpertflag = 1;
            }
            else{
                ckt->CKTsenInfo->SENacpertflag = 0;
            }
            if(error = CKTsenAC(ckt)) return (error);
            ckt->CKTmode = save;
            ckt->CKTsenInfo->SENmode = save1;
        }
#endif

#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */

        if(g_ipc.enabled)
            ipc_send_data_prefix(freq);

        error = CKTacDump(ckt,freq,acPlot);

        if(g_ipc.enabled)
            ipc_send_data_suffix();

/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#else
        error = CKTacDump(ckt,freq,acPlot);
#endif	
        if (error) {
	    ckt->CKTcurrentAnalysis = DOING_AC;
	    ckt->CKTstat->STATacTime += SPfrontEnd->IFseconds() - startTime;
 	    ckt->CKTstat->STATacDecompTime += ckt->CKTstat->STATdecompTime -
		    startdTime;
 	    ckt->CKTstat->STATacSolveTime += ckt->CKTstat->STATsolveTime -
 		    startsTime;
 	    ckt->CKTstat->STATacLoadTime += ckt->CKTstat->STATloadTime -
 		    startlTime;
 	    ckt->CKTstat->STATacCombTime += ckt->CKTstat->STATcombineTime -
 		    startcTime;
	    ckt->CKTstat->STATacSyncTime += ckt->CKTstat->STATsyncTime -
		    startkTime;
 	    return(error);
 	}

        /*  increment frequency */

        switch(((ACAN*)ckt->CKTcurJob)->ACstepType) {
        case DECADE:
        case OCTAVE:

/* inserted again 14.12.2001  */
#ifdef HAS_WINDOWS
			 {
				 double endfreq   = ((ACAN*)ckt->CKTcurJob)->ACstopFreq;
				 double startfreq = ((ACAN*)ckt->CKTcurJob)->ACstartFreq;
				 double step      = ((ACAN*)ckt->CKTcurJob)->ACfreqDelta;
				 endfreq   = log(endfreq);
				 if (startfreq == 0.0)
                	startfreq = 1e-12;
				 startfreq = log(startfreq);

				 if (freq > 0.0)
					 SetAnalyse( "ac", (log(freq)-startfreq) * 100.0 / (endfreq-startfreq));
			 }
#endif

            freq *= ((ACAN*)ckt->CKTcurJob)->ACfreqDelta;
            if(((ACAN*)ckt->CKTcurJob)->ACfreqDelta==1) goto endsweep;
            break;
        case LINEAR:

#ifdef HAS_WINDOWS
			 {
				 double endfreq   = ((ACAN*)ckt->CKTcurJob)->ACstopFreq;
				 double startfreq = ((ACAN*)ckt->CKTcurJob)->ACstartFreq;
				 SetAnalyse( "ac", (freq - startfreq)* 100.0 / (endfreq-startfreq));
			 }
#endif

            freq += ((ACAN*)ckt->CKTcurJob)->ACfreqDelta;
            if(((ACAN*)ckt->CKTcurJob)->ACfreqDelta==0) goto endsweep;
            break;
        default:
            return(E_INTERN);
        }
    }
endsweep:
    (*(SPfrontEnd->OUTendPlot))(acPlot);
    acPlot = NULL;
    ckt->CKTcurrentAnalysis = 0;
    ckt->CKTstat->STATacTime += SPfrontEnd->IFseconds() - startTime;
    ckt->CKTstat->STATacDecompTime += ckt->CKTstat->STATdecompTime -
 	    startdTime;
    ckt->CKTstat->STATacSolveTime += ckt->CKTstat->STATsolveTime -
 	    startsTime;
    ckt->CKTstat->STATacLoadTime += ckt->CKTstat->STATloadTime -
 	    startlTime;
    ckt->CKTstat->STATacCombTime += ckt->CKTstat->STATcombineTime -
	    startcTime;
    ckt->CKTstat->STATacSyncTime += ckt->CKTstat->STATsyncTime -
	    startkTime;
    return(0);
}


    /* CKTacLoad(ckt)
     * this is a driver program to iterate through all the various
     * ac load functions provided for the circuit elements in the
     * given circuit 
     */


int
CKTacLoad(CKTcircuit *ckt)
{
    extern SPICEdev **DEVices;
    int i;
    int size;
    int error;
#ifdef PARALLEL_ARCH
    long type = MT_ACLOAD, length = 1;
#endif /* PARALLEL_ARCH */
    double startTime;

    startTime  = SPfrontEnd->IFseconds();
    size = SMPmatSize(ckt->CKTmatrix);
    for (i=0;i<=size;i++) {
        *(ckt->CKTrhs+i)=0;
        *(ckt->CKTirhs+i)=0;
    }
    SMPcClear(ckt->CKTmatrix);

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVacLoad != NULL) && (ckt->CKThead[i] != NULL) ){
            error = (*((*DEVices[i]).DEVacLoad))(ckt->CKThead[i],ckt);
#ifdef PARALLEL_ARCH
	    if (error) goto combine;
#else
            if(error) return(error);
#endif /* PARALLEL_ARCH */
        }
    }
    
#ifdef XSPICE
   /* gtri - begin - Put resistors to ground at all nodes. */
    /* Value of resistor is set by new "rshunt" option.     */

    if(ckt->enh->rshunt_data.enabled) {
       for(i = 0; i < ckt->enh->rshunt_data.num_nodes; i++) {
          *(ckt->enh->rshunt_data.diag[i]) +=
                               ckt->enh->rshunt_data.gshunt;
       }
    }

    /* gtri - end - Put resistors to ground at all nodes */



    /* gtri - add - wbk - 11/26/90 - reset the MIF init flags */

    /* init is set by CKTinit and should be true only for first load call */
    g_mif_info.circuit.init = MIF_FALSE;

    /* anal_init is set by CKTdoJob and is true for first call */
    /* of a particular analysis type */
    g_mif_info.circuit.anal_init = MIF_FALSE;

    /* gtri - end - wbk - 11/26/90 */
#endif

    
#ifdef PARALLEL_ARCH
combine:
    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds() - startTime;
    startTime  = SPfrontEnd->IFseconds();
    /* See if any of the DEVload functions bailed. If not, proceed. */
    IGOP_( &type, &error, &length, "max" );
    ckt->CKTstat->STATsyncTime += SPfrontEnd->IFseconds() - startTime;
    if (error == OK) {
      startTime  = SPfrontEnd->IFseconds();
      SMPcCombine( ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTrhsSpare,
	  ckt->CKTirhs, ckt->CKTirhsSpare );
      ckt->CKTstat->STATcombineTime += SPfrontEnd->IFseconds() - startTime;
      return(OK);
    } else {
      return(error);
    }
#else
    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds() - startTime;
    return(OK);
#endif /* PARALLEL_ARCH */
}
