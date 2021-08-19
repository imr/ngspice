/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/carddefs.h"
#include "ngspice/spmatrix.h"
#include "onedext.h"
#include "oneddefs.h"

#include <inttypes.h>


void
ONEprnSolution(FILE *file, ONEdevice *pDevice, OUTPcard *output, BOOLEAN asciiSave, char *extra)
{
  int index, i, ii;
  int numVars = 0;
  ONEnode **nodeArray=NULL;
  ONEnode *pNode;
  ONEelem *pElem, *pPrevElem;
  ONEmaterial *info=NULL;
  double data[50];
  double eField, refPsi = 0.0, eGap, dGap;
  double mun, mup, jc, jd, jn, jp, jt;
  double coeff1, coeff2;


  if (output->OUTPnumVars == -1) {
    /* First pass. Need to count number of variables in output. */
    numVars++;			/* For the X scale */
    if (output->OUTPdoping) {
      numVars++;
    }
    if (output->OUTPpsi) {
      numVars++;
    }
    if (output->OUTPequPsi) {
      numVars++;
    }
    if (output->OUTPvacPsi) {
      numVars++;
    }
    if (output->OUTPnConc) {
      numVars++;
    }
    if (output->OUTPpConc) {
      numVars++;
    }
    if (output->OUTPphin) {
      numVars++;
    }
    if (output->OUTPphip) {
      numVars++;
    }
    if (output->OUTPphic) {
      numVars++;
    }
    if (output->OUTPphiv) {
      numVars++;
    }
    if (output->OUTPeField) {
      numVars++;
    }
    if (output->OUTPjc) {
      numVars++;
    }
    if (output->OUTPjd) {
      numVars++;
    }
    if (output->OUTPjn) {
      numVars++;
    }
    if (output->OUTPjp) {
      numVars++;
    }
    if (output->OUTPjt) {
      numVars++;
    }
    if (output->OUTPuNet) {
      numVars++;
    }
    if (output->OUTPmun) {
      numVars++;
    }
    if (output->OUTPmup) {
      numVars++;
    }
    output->OUTPnumVars = numVars;
  }
  /* generate the work array for printing node info */
  XCALLOC(nodeArray, ONEnode *, 1 + pDevice->numNodes);

  /* store the nodes in this work array and print out later */
  for (index = 1; index < pDevice->numNodes; index++) {
    pElem = pDevice->elemArray[index];
    if (refPsi == 0.0 && pElem->matlInfo->type == SEMICON) {
      refPsi = pElem->matlInfo->refPsi;
    }
    for (i = 0; i <= 1; i++) {
      if (pElem->evalNodes[i]) {
	pNode = pElem->pNodes[i];
	nodeArray[pNode->nodeI] = pNode;
      }
    }
  }

  /* Initialize rawfile */
  numVars = output->OUTPnumVars;
  if (extra != NULL) {
    fprintf(file, "Title: Device %s (%s) internal state\n", pDevice->name, extra);
  } else {
    fprintf(file, "Title: Device %s internal state\n", pDevice->name);
  }
  fprintf(file, "Plotname: Device Cross Section\n");
  fprintf(file, "Flags: real\n");
  fprintf(file, "Command: deftype p xs cross\n");
  fprintf(file, "Command: deftype v distance m\n");
  fprintf(file, "Command: deftype v concentration cm^-3\n");
  fprintf(file, "Command: deftype v electric_field V/cm\n");
  fprintf(file, "Command: deftype v current_density A/cm^2\n");
  fprintf(file, "Command: deftype v concentration/time cm^-3/s\n");
  fprintf(file, "Command: deftype v mobility cm^2/Vs\n");
  fprintf(file, "No. Variables: %d\n", numVars);
  fprintf(file, "No. Points: %d\n", pDevice->numNodes);
  numVars = 0;
  fprintf(file, "Variables:\n");
  fprintf(file, "\t%d	x	distance\n", numVars++);
  if (output->OUTPpsi) {
    fprintf(file, "\t%d	psi	voltage\n", numVars++);
  }
  if (output->OUTPequPsi) {
    fprintf(file, "\t%d	equ.psi	voltage\n", numVars++);
  }
  if (output->OUTPvacPsi) {
    fprintf(file, "\t%d	vac.psi	voltage\n", numVars++);
  }
  if (output->OUTPphin) {
    fprintf(file, "\t%d	phin	voltage\n", numVars++);
  }
  if (output->OUTPphip) {
    fprintf(file, "\t%d	phip	voltage\n", numVars++);
  }
  if (output->OUTPphic) {
    fprintf(file, "\t%d	phic	voltage\n", numVars++);
  }
  if (output->OUTPphiv) {
    fprintf(file, "\t%d	phiv	voltage\n", numVars++);
  }
  if (output->OUTPdoping) {
    fprintf(file, "\t%d	dop	concentration\n", numVars++);
  }
  if (output->OUTPnConc) {
    fprintf(file, "\t%d	n	concentration\n", numVars++);
  }
  if (output->OUTPpConc) {
    fprintf(file, "\t%d	p	concentration\n", numVars++);
  }
  if (output->OUTPeField) {
    fprintf(file, "\t%d	e	electric_field\n", numVars++);
  }
  if (output->OUTPjc) {
    fprintf(file, "\t%d	jc	current_density\n", numVars++);
  }
  if (output->OUTPjd) {
    fprintf(file, "\t%d	jd	current_density\n", numVars++);
  }
  if (output->OUTPjn) {
    fprintf(file, "\t%d	jn	current_density\n", numVars++);
  }
  if (output->OUTPjp) {
    fprintf(file, "\t%d	jp	current_density\n", numVars++);
  }
  if (output->OUTPjt) {
    fprintf(file, "\t%d	jt	current_density\n", numVars++);
  }
  if (output->OUTPuNet) {
    fprintf(file, "\t%d	unet	concentration/time\n", numVars++);
  }
  if (output->OUTPmun) {
    fprintf(file, "\t%d	mun	mobility\n", numVars++);
  }
  if (output->OUTPmup) {
    fprintf(file, "\t%d	mup	mobility\n", numVars++);
  }
  if (asciiSave) {
    fprintf(file, "Values:\n");
  } else {
    fprintf(file, "Binary:\n");
  }

  for (index = 1; index <= pDevice->numNodes; index++) {
    pNode = nodeArray[index];
    if ((index > 1) && (index < pDevice->numNodes)) {
      pElem = pNode->pRightElem;
      pPrevElem = pNode->pLeftElem;
      if (pElem->evalNodes[0]) {
	info = pElem->matlInfo;
      } else if (pPrevElem->evalNodes[1]) {
	info = pPrevElem->matlInfo;
      }
      coeff1 = pPrevElem->dx / (pPrevElem->dx + pElem->dx);
      coeff2 = pElem->dx / (pPrevElem->dx + pElem->dx);
      eField = -coeff1 * pElem->pEdge->dPsi * pElem->rDx
	  - coeff2 * pPrevElem->pEdge->dPsi * pPrevElem->rDx;
      mun = coeff1 * pElem->pEdge->mun + coeff2 * pPrevElem->pEdge->mun;
      mup = coeff1 * pElem->pEdge->mup + coeff2 * pPrevElem->pEdge->mup;
      jn = coeff1 * pElem->pEdge->jn + coeff2 * pPrevElem->pEdge->jn;
      jp = coeff1 * pElem->pEdge->jp + coeff2 * pPrevElem->pEdge->jp;
      jd = coeff1 * pElem->pEdge->jd + coeff2 * pPrevElem->pEdge->jd;
    } else if (index == 1) {
      info = pNode->pRightElem->matlInfo;
      eField = 0.0;
      mun = pNode->pRightElem->pEdge->mun;
      mup = pNode->pRightElem->pEdge->mup;
      jn = pNode->pRightElem->pEdge->jn;
      jp = pNode->pRightElem->pEdge->jp;
      jd = pNode->pRightElem->pEdge->jd;
    } else {
      info = pNode->pLeftElem->matlInfo;
      eField = 0.0;
      mun = pNode->pLeftElem->pEdge->mun;
      mup = pNode->pLeftElem->pEdge->mup;
      jn = pNode->pLeftElem->pEdge->jn;
      jp = pNode->pLeftElem->pEdge->jp;
      jd = pNode->pLeftElem->pEdge->jd;
    }
    jc = jn + jp;
    jt = jc + jd;
    /* Crude hack to get around the fact that the base node wipes out 'eg' */
    if (index == pDevice->baseIndex) {
      eGap = info->eg0;
      dGap = 0.0;
    } else {
      eGap = pNode->eg * VNorm;
      dGap = 0.5 * (info->eg0 - eGap);
    }

    /* Now fill in the data array */
    numVars = 0;
    data[numVars++] = pNode->x * 1e-2;
    if (output->OUTPpsi) {
      data[numVars++] = (pNode->psi - refPsi) * VNorm;
    }
    if (output->OUTPequPsi) {
      data[numVars++] = (pNode->psi0 - refPsi) * VNorm;
    }
    if (output->OUTPvacPsi) {
      data[numVars++] = pNode->psi * VNorm;
    }
    if (output->OUTPphin) {
      if (info->type != INSULATOR) {
	data[numVars++] = (pNode->psi - refPsi - log(pNode->nConc / pNode->nie))
	    * VNorm;
      } else {
	data[numVars++] = 0.0;
      }
    }
    if (output->OUTPphip) {
      if (info->type != INSULATOR) {
	data[numVars++] = (pNode->psi - refPsi + log(pNode->pConc / pNode->nie))
	    * VNorm;
      } else {
	data[numVars++] = 0.0;
      }
    }
    if (output->OUTPphic) {
      data[numVars++] = (pNode->psi + pNode->eaff) * VNorm + dGap;
    }
    if (output->OUTPphiv) {
      data[numVars++] = (pNode->psi + pNode->eaff) * VNorm + dGap + eGap;
    }
    if (output->OUTPdoping) {
      data[numVars++] = pNode->netConc * NNorm;
    }
    if (output->OUTPnConc) {
      data[numVars++] = pNode->nConc * NNorm;
    }
    if (output->OUTPpConc) {
      data[numVars++] = pNode->pConc * NNorm;
    }
    if (output->OUTPeField) {
      data[numVars++] = eField * ENorm;
    }
    if (output->OUTPjc) {
      data[numVars++] = jc * JNorm;
    }
    if (output->OUTPjd) {
      data[numVars++] = jd * JNorm;
    }
    if (output->OUTPjn) {
      data[numVars++] = jn * JNorm;
    }
    if (output->OUTPjp) {
      data[numVars++] = jp * JNorm;
    }
    if (output->OUTPjt) {
      data[numVars++] = jt * JNorm;
    }
    if (output->OUTPuNet) {
      data[numVars++] = pNode->uNet * NNorm / TNorm;
    }
    if (output->OUTPmun) {
      data[numVars++] = mun;
    }
    if (output->OUTPmup) {
      data[numVars++] = mup;
    }
    if (asciiSave) {
      for (ii = 0; ii < numVars; ii++) {
        if (ii == 0) {
          fprintf(file, "%d", index - 1);
        }
        fprintf(file, "\t%e\n", data[ii]);
      }
    } else {
      fwrite(data, sizeof(double), (size_t) numVars, file);
    }
  }
  FREE(nodeArray);
}

/*
 * XXX This is what the SPARSE element structure looks like. We can't take it
 * from its definition because the include file redefines all sorts of
 * things.  Note that we are violating data encapsulation to find out the
 * size of this thing.
 */
struct MatrixElement {
  spREAL Real;
  spREAL Imag;
  int Row;
  int Col;
  struct MatrixElement *NextInRow;
  struct MatrixElement *NextInCol;
};

void
ONEmemStats(FILE *file, ONEdevice *pDevice)
{
  const char memFormat[] = "%-20s" "%10d" "%10" PRIuPTR "\n";
/*  static const char sumFormat[] = "%20s          %-10d\n";*/
  int size;
  size_t memory;
  ONEmaterial *pMaterial;
  ONEcontact *pContact;
  int numContactNodes;


  if (!pDevice) { return; }
  fprintf(file, "----------------------------------------\n");
  fprintf(file, "Device %s Memory Usage:\n", pDevice->name);
  fprintf(file, "Item                     Count     Bytes\n");
  fprintf(file, "----------------------------------------\n");

  size = 1;
  memory = (size_t) size * sizeof(ONEdevice);
  fprintf(file, memFormat, "Device", size, memory);
  size = pDevice->numNodes - 1;
  memory = (size_t) size * sizeof(ONEelem);
  fprintf(file, memFormat, "Elements", size, memory);
  size = pDevice->numNodes;
  memory = (size_t) size * sizeof(ONEnode);
  fprintf(file, memFormat, "Nodes", size, memory);
  size = pDevice->numNodes - 1;
  memory = (size_t) size * sizeof(ONEedge);
  fprintf(file, memFormat, "Edges", size, memory);

  size = pDevice->numNodes;
  memory = (size_t) size * sizeof(ONEelem *);
  size = 0;
  for (pMaterial = pDevice->pMaterials; pMaterial; pMaterial = pMaterial->next)
    size++;
  memory += (size_t) size * sizeof(ONEmaterial);
  size = numContactNodes = 0;
  for (pContact = pDevice->pFirstContact; pContact; pContact = pContact->next) {
    numContactNodes += pContact->numNodes;
    size++;
  }
  memory += (size_t) size * sizeof(ONEcontact);
  size = numContactNodes;
  memory += (size_t) size * sizeof(ONEnode *);
  size = 0;
  fprintf(file, "%-20s%10s%10" PRIuPTR "\n", "Misc Mesh", "n/a", memory);

  size = pDevice->numOrigEquil;
  memory = (size_t) size * sizeof(struct MatrixElement);
  fprintf(file, memFormat, "Equil Orig NZ", size, memory);
  size = pDevice->numFillEquil;
  memory = (size_t) size * sizeof(struct MatrixElement);
  fprintf(file, memFormat, "Equil Fill NZ", size, memory);
  size = pDevice->numOrigEquil + pDevice->numFillEquil;
  memory = (size_t) size * sizeof(struct MatrixElement);
  fprintf(file, memFormat, "Equil Tot  NZ", size, memory);
  size = pDevice->dimEquil;
  memory = (size_t) size * 4 * sizeof(double);
  fprintf(file, memFormat, "Equil Vectors", size, memory);

  size = pDevice->numOrigBias;
  memory = (size_t) size * sizeof(struct MatrixElement);
  fprintf(file, memFormat, "Bias Orig NZ", size, memory);
  size = pDevice->numFillBias;
  memory = (size_t) size * sizeof(struct MatrixElement);
  fprintf(file, memFormat, "Bias Fill NZ", size, memory);
  size = pDevice->numOrigBias + pDevice->numFillBias;
  memory = (size_t) size * sizeof(struct MatrixElement);
  fprintf(file, memFormat, "Bias Tot  NZ", size, memory);
  size = pDevice->dimBias;
  memory = (size_t) size * 5 * sizeof(double);
  fprintf(file, memFormat, "Bias Vectors", size, memory);

  size = (pDevice->numNodes - 1) * ONEnumEdgeStates +
      pDevice->numNodes * ONEnumNodeStates;
  memory = (size_t) size * sizeof(double);
  fprintf(file, memFormat, "State Vector", size, memory);
}

void
ONEcpuStats(FILE *file, ONEdevice *pDevice)
{
  static const char cpuFormat[] = "%-20s%10g%10g%10g%10g%10g\n";
  ONEstats *pStats = NULL;
  double total;
  int iTotal;

  if (!pDevice) { return; }
  pStats = pDevice->pStats;
  fprintf(file,
      "----------------------------------------------------------------------\n");
  fprintf(file,
      "Device %s Time Usage:\n", pDevice->name);
  fprintf(file,
      "Item                     SETUP        DC      TRAN        AC     TOTAL\n");
  fprintf(file,
      "----------------------------------------------------------------------\n");

  total = pStats->setupTime[STAT_SETUP] +
      pStats->setupTime[STAT_DC] +
      pStats->setupTime[STAT_TRAN] +
      pStats->setupTime[STAT_AC];
  fprintf(file, cpuFormat, "Setup Time",
      pStats->setupTime[STAT_SETUP],
      pStats->setupTime[STAT_DC],
      pStats->setupTime[STAT_TRAN],
      pStats->setupTime[STAT_AC],
      total);

  total = pStats->loadTime[STAT_SETUP] +
      pStats->loadTime[STAT_DC] +
      pStats->loadTime[STAT_TRAN] +
      pStats->loadTime[STAT_AC];
  fprintf(file, cpuFormat, "Load Time",
      pStats->loadTime[STAT_SETUP],
      pStats->loadTime[STAT_DC],
      pStats->loadTime[STAT_TRAN],
      pStats->loadTime[STAT_AC],
      total);

  total = pStats->orderTime[STAT_SETUP] +
      pStats->orderTime[STAT_DC] +
      pStats->orderTime[STAT_TRAN] +
      pStats->orderTime[STAT_AC];
  fprintf(file, cpuFormat, "Order Time",
      pStats->orderTime[STAT_SETUP],
      pStats->orderTime[STAT_DC],
      pStats->orderTime[STAT_TRAN],
      pStats->orderTime[STAT_AC],
      total);

  total = pStats->factorTime[STAT_SETUP] +
      pStats->factorTime[STAT_DC] +
      pStats->factorTime[STAT_TRAN] +
      pStats->factorTime[STAT_AC];
  fprintf(file, cpuFormat, "Factor Time",
      pStats->factorTime[STAT_SETUP],
      pStats->factorTime[STAT_DC],
      pStats->factorTime[STAT_TRAN],
      pStats->factorTime[STAT_AC],
      total);

  total = pStats->solveTime[STAT_SETUP] +
      pStats->solveTime[STAT_DC] +
      pStats->solveTime[STAT_TRAN] +
      pStats->solveTime[STAT_AC];
  fprintf(file, cpuFormat, "Solve Time",
      pStats->solveTime[STAT_SETUP],
      pStats->solveTime[STAT_DC],
      pStats->solveTime[STAT_TRAN],
      pStats->solveTime[STAT_AC],
      total);

  total = pStats->updateTime[STAT_SETUP] +
      pStats->updateTime[STAT_DC] +
      pStats->updateTime[STAT_TRAN] +
      pStats->updateTime[STAT_AC];
  fprintf(file, cpuFormat, "Update Time",
      pStats->updateTime[STAT_SETUP],
      pStats->updateTime[STAT_DC],
      pStats->updateTime[STAT_TRAN],
      pStats->updateTime[STAT_AC],
      total);

  total = pStats->checkTime[STAT_SETUP] +
      pStats->checkTime[STAT_DC] +
      pStats->checkTime[STAT_TRAN] +
      pStats->checkTime[STAT_AC];
  fprintf(file, cpuFormat, "Check Time",
      pStats->checkTime[STAT_SETUP],
      pStats->checkTime[STAT_DC],
      pStats->checkTime[STAT_TRAN],
      pStats->checkTime[STAT_AC],
      total);

  total = pStats->setupTime[STAT_SETUP] +
      pStats->setupTime[STAT_DC] +
      pStats->setupTime[STAT_TRAN] +
      pStats->setupTime[STAT_AC];
  fprintf(file, cpuFormat, "Misc Time",
      pStats->miscTime[STAT_SETUP],
      pStats->miscTime[STAT_DC],
      pStats->miscTime[STAT_TRAN],
      pStats->miscTime[STAT_AC],
      total);

  fprintf(file, "%-40s%10g%10s%10g\n", "LTE Time",
      pStats->lteTime,
      "", pStats->lteTime);

  total = pStats->totalTime[STAT_SETUP] +
      pStats->totalTime[STAT_DC] +
      pStats->totalTime[STAT_TRAN] +
      pStats->totalTime[STAT_AC];
  fprintf(file, cpuFormat, "Total Time",
      pStats->totalTime[STAT_SETUP],
      pStats->totalTime[STAT_DC],
      pStats->totalTime[STAT_TRAN],
      pStats->totalTime[STAT_AC],
      total);

  iTotal = pStats->numIters[STAT_SETUP] +
      pStats->numIters[STAT_DC] +
      pStats->numIters[STAT_TRAN] +
      pStats->numIters[STAT_AC];
  fprintf(file, "%-20s%10d%10d%10d%10d%10d\n", "Iterations",
      pStats->numIters[STAT_SETUP],
      pStats->numIters[STAT_DC],
      pStats->numIters[STAT_TRAN],
      pStats->numIters[STAT_AC],
      iTotal);
}
