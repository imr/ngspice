/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * This is a simple routine to dump the internal device states. It produces
 * states for .OP, .DC, & .TRAN simulations.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nbjtdefs.h"
#include "../../../ciderlib/oned/onedext.h"
#include "ngspice/cidersupt.h"
#include "ngspice/suffix.h"


/* Forward Declarations */
static void NBJTputHeader(FILE *, CKTcircuit *, NBJTinstance *);

/* State Counter */
static int state_numOP = 0;
static int state_numDC = 0;
static int state_numTR = 0;

void
NBJTdump(GENmodel *inModel, CKTcircuit *ckt)
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  OUTPcard *output;
  FILE *fpState;
  char fileName[BSIZE_SP];
  char description[BSIZE_SP];
  char *prefix;
  int *state_num;
  int anyOutput = 0;
  BOOLEAN writeAscii = TRUE;

  if (ckt->CKTmode & MODEDCOP) {
    prefix = "OP";
    state_num = &state_numOP;
    sprintf(description, "...");
  } else if (ckt->CKTmode & MODEDCTRANCURVE) {
    prefix = "DC";
    state_num = &state_numDC;
    sprintf(description, "sweep = % e", ckt->CKTtime);
  } else if (ckt->CKTmode & MODETRAN) {
    prefix = "TR";
    state_num = &state_numTR;
    sprintf(description, "time = % e", ckt->CKTtime);
  } else {
    /* Not a recognized CKT mode. */
    return;
  }

  for (; model != NULL; model = NBJTnextModel(model)) {
    output = model->NBJToutputs;
    for (inst = NBJTinstances(model); inst != NULL;
         inst = NBJTnextInstance(inst)) {

      if (inst->NBJTprintGiven) {
	if ((ckt->CKTmode & MODETRAN) &&
	    ((ckt->CKTstat->STATaccepted - 1) % inst->NBJTprint != 0)) {
	  continue;
	}
	anyOutput = 1;
	sprintf(fileName, "%s%s.%d.%s", output->OUTProotFile, prefix,
	    *state_num, inst->NBJTname);

	writeAscii = compareFiletypeVar("ascii");

	fpState = fopen(fileName, (writeAscii ? "w" : "wb"));
	if (!fpState) {
	  perror(fileName);
	} else {
	  NBJTputHeader(fpState, ckt, inst);
	  ONEprnSolution(fpState, inst->NBJTpDevice,
	      model->NBJToutputs, writeAscii, "nbjt");
	  fclose(fpState);
	  LOGmakeEntry(fileName, description);
	}
      }
    }
  }
  if (anyOutput) {
    (*state_num)++;
  }
}

#define NBJTnumOutputs 9

static
void 
NBJTputHeader(FILE *file, CKTcircuit *ckt, NBJTinstance *inst)
{
  char *reference;
  double refVal = 0.0;
  int numVars = NBJTnumOutputs;

  if (ckt->CKTmode & MODEDCOP) {
    reference = NULL;
  } else if (ckt->CKTmode & MODEDCTRANCURVE) {
    reference = "sweep";
    refVal = ckt->CKTtime;
    numVars++;
  } else if (ckt->CKTmode & MODETRAN) {
    reference = "time";
    refVal = ckt->CKTtime;
    numVars++;
  } else {
    reference = NULL;
  }
  fprintf(file, "Title: Device %s external state\n", inst->NBJTname);
  fprintf(file, "Plotname: Device Operating Point\n");
  fprintf(file, "Command: deftype v conductance S\n");
  fprintf(file, "Flags: real\n");
  fprintf(file, "No. Variables: %d\n", numVars);
  fprintf(file, "No. Points: 1\n");
  numVars = 0;
  fprintf(file, "Variables:\n");
  if (reference) {
    fprintf(file, "\t%d	%s	unknown\n", numVars++, reference);
  }
  fprintf(file, "\t%d	v13	voltage\n", numVars++);
  fprintf(file, "\t%d	v23	voltage\n", numVars++);
  fprintf(file, "\t%d	i1	current\n", numVars++);
  fprintf(file, "\t%d	i2	current\n", numVars++);
  fprintf(file, "\t%d	i3	current\n", numVars++);
  fprintf(file, "\t%d	g22	conductance\n", numVars++);
  fprintf(file, "\t%d	g21	conductance\n", numVars++);
  fprintf(file, "\t%d	g12	conductance\n", numVars++);
  fprintf(file, "\t%d	g11	conductance\n", numVars++);
  fprintf(file, "Values:\n0");
  if (reference) {
    fprintf(file, "\t% e\n", refVal);
  }
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTvce));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTvbe));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTic));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTie)
      - *(ckt->CKTstate0 + inst->NBJTic));
  fprintf(file, "\t% e\n", -*(ckt->CKTstate0 + inst->NBJTie));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTdIeDVbe)
      - *(ckt->CKTstate0 + inst->NBJTdIcDVbe));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTdIeDVce)
      - *(ckt->CKTstate0 + inst->NBJTdIcDVce));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTdIcDVbe));
  fprintf(file, "\t% e\n", *(ckt->CKTstate0 + inst->NBJTdIcDVce));
}

void
NBJTacct(GENmodel *inModel, CKTcircuit *ckt, FILE *file)
{
  register NBJTmodel *model = (NBJTmodel *) inModel;
  register NBJTinstance *inst;
  OUTPcard *output;

  NG_IGNORE(ckt);

  for (; model != NULL; model = NBJTnextModel(model)) {
    output = model->NBJToutputs;
    for (inst = NBJTinstances(model); inst != NULL;
         inst = NBJTnextInstance(inst)) {

      if (output->OUTPstats) {
	ONEmemStats(file, inst->NBJTpDevice);
	ONEcpuStats(file, inst->NBJTpDevice);
      }
    }
  }
}
