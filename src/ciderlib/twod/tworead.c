/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Functions needed to read solutions for 2D devices.
 */

#include "ngspice/ngspice.h"
#include "ngspice/plot.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/twodev.h"
#include "ngspice/twomesh.h"
#include "twoddefs.h"
#include "twodext.h"
#include "ngspice/cidersupt.h"


int
TWOreadState(TWOdevice *pDevice, char *fileName, int numVolts, double *pV1, 
             double *pV2, double *pV3)
	/* char *fileName: File containing raw data           */
	/* int numVolts: Number of voltage differences        */
	/* double *pV1, *pV2, *pV3: Pointer to return them in */
{
  int dataLength;
  int i, index, xIndex, yIndex;
  TWOnode ***nodeArray = NULL;
  TWOnode *pNode;
  TWOelem *pElem;
  double refPsi = 0.0;
  double *psiData, *nData, *pData;
  double *vData[3];
  struct plot *stateDB;
  struct plot *voltsDB;
  char voltName[80];

  stateDB = DBread( fileName );
  if (stateDB == NULL) return (-1);
  voltsDB = stateDB->pl_next;
  if (voltsDB == NULL) return (-1);

  for (i=0; i < numVolts; i++ ) {
    sprintf( voltName, "v%d%d", i+1, numVolts+1 );
    vData[i] = DBgetData( voltsDB, voltName, 1 );
    if (vData[i] == NULL) return (-1);
  }
  dataLength = pDevice->numXNodes * pDevice->numYNodes;
  psiData = DBgetData( stateDB, "psi", dataLength );
  nData = DBgetData( stateDB, "n", dataLength );
  pData = DBgetData( stateDB, "p", dataLength );
  if (psiData == NULL || nData == NULL || pData == NULL) return (-1);

  if (pV1 != NULL) {
    *pV1 = vData[0][0];
    FREE( vData[0] );
  }
  if (pV2 != NULL) {
    *pV2 = vData[1][0];
    FREE( vData[1] );
  }
  if (pV3 != NULL) {
    *pV3 = vData[2][0];
    FREE( vData[2] );
  }

  /* generate the work array for copying node info */
  XCALLOC(nodeArray, TWOnode **, 1 + pDevice->numXNodes);
  for (xIndex = 1; xIndex <= pDevice->numXNodes; xIndex++) {
    XCALLOC(nodeArray[xIndex], TWOnode *, 1 + pDevice->numYNodes);
  }

  /* store the nodes in this work array and use later */
  for (xIndex = 1; xIndex < pDevice->numXNodes; xIndex++) {
    for (yIndex = 1; yIndex < pDevice->numYNodes; yIndex++) {
      pElem = pDevice->elemArray[xIndex][yIndex];
      if (pElem != NULL) {
	if (refPsi == 0.0 && pElem->matlInfo->type == SEMICON) {
	  refPsi = pElem->matlInfo->refPsi;
	}
	for (index = 0; index <= 3; index++) {
	  if (pElem->evalNodes[index]) {
	    pNode = pElem->pNodes[index];
	    nodeArray[pNode->nodeI][pNode->nodeJ] = pNode;
	  }
	}
      }
    }
  }
  index = 0;
  for (xIndex = 1; xIndex <= pDevice->numXNodes; xIndex++) {
    for (yIndex = 1; yIndex <= pDevice->numYNodes; yIndex++) {
      pNode = nodeArray[xIndex][yIndex];
      index++;
      if (pNode != NULL) {
	pNode->psi = psiData[index-1]/VNorm + refPsi;
	pNode->nConc = nData[index-1]/NNorm;
	pNode->pConc = pData[index-1]/NNorm;
      }
    }
  }
  /* Delete work array. */
  for (xIndex = 1; xIndex <= pDevice->numXNodes; xIndex++) {
    FREE(nodeArray[xIndex]);
  }
  FREE(nodeArray);

  FREE(psiData);
  FREE(nData);
  FREE(pData);

  return (0);
}
