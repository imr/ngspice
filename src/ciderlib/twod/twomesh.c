/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "ngspice/bool.h"
#include "twoddefs.h"
#include "twodext.h"



static void doMobCoeffs(TWOelem *, int); 
static void resetEvalFlag(TWOdevice *pDevice);

extern void CiderLoaded(int);

void
TWObuildMesh(TWOdevice *pDevice, TWOdomain *pDomain, 
             TWOelectrode *pElectrode, TWOmaterial *pMaterial)
{
  int xIndex, yIndex, eIndex, index;
  int elemType;
  TWOelem *pElem, *pElem1;
  TWOnode *pNode, *pNode1, *pNextHNode, *pNextVNode, *pNextDNode;
  TWOnode ***nodeArray = NULL;
  TWOedge *pEdge;
  TWOedge ***edgeArrayH = NULL, ***edgeArrayV = NULL;
  TWOdomain *pD;
  TWOelectrode *pE;
  TWOmaterial *pM;
  BOOLEAN interiorNode;
  int poiEqn, numEqn, numElem, numNodes, numEdges;
  int numXNodes = pDevice->numXNodes;
  int numYNodes = pDevice->numYNodes;
  double *xScale = pDevice->xScale;
  double *yScale = pDevice->yScale;
#ifdef NOTDEF
  FILE *meshFile;
#endif

  /* Generate work arrays. */
  XCALLOC(nodeArray, TWOnode **, 1 + numXNodes);
  for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
    XCALLOC(nodeArray[xIndex], TWOnode *, 1 + numYNodes);
  }

  /* Generate the nodes. */
  for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
    for (yIndex = 1; yIndex <= numYNodes; yIndex++) {
      XCALLOC(pNode, TWOnode, 1);
      pNode->nodeI = xIndex;
      pNode->nodeJ = yIndex;
      pNode->poiEqn = 0;
      nodeArray[xIndex][yIndex] = pNode;
    }
  }

  /* Mark the semiconductor/insulator domains. */
  if (pDomain == NULL) {
    fprintf(stderr, "Error: domains not defined for device\n");
    exit(-1);
  }
  for (pD = pDomain; pD != NULL; pD = pD->next) {
    for (pM = pMaterial; pM != NULL; pM = pM->next) {
      if (pD->material == pM->id) {
	break;
      }
    }
    elemType = pM->type;
    for (xIndex = pD->ixLo; xIndex <= pD->ixHi; xIndex++) {
      for (yIndex = pD->iyLo; yIndex <= pD->iyHi; yIndex++) {
	pNode = nodeArray[xIndex][yIndex];
	pNode->nodeType = elemType;
      }
    }
  }
  /* Now mark all the metallic domains */
  for (pE = pElectrode; pE != NULL; pE = pE->next) {
    for (xIndex = pE->ixLo; xIndex <= pE->ixHi; xIndex++) {
      for (yIndex = pE->iyLo; yIndex <= pE->iyHi; yIndex++) {
	pNode = nodeArray[xIndex][yIndex];
	pNode->nodeType = CONTACT;
      }
    }
  }
  /*
   * Now mark as NULL any node in the interior of an electrode, i.e. none of
   * its neighbors is a different material.
   */
  for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
    for (yIndex = 1; yIndex <= numYNodes; yIndex++) {
      pNode = nodeArray[xIndex][yIndex];
      if (pNode->nodeType == CONTACT) {
	interiorNode = TRUE;
	if (xIndex > 1) {
	  pNode1 = nodeArray[xIndex - 1][yIndex];
	  if (pNode1->nodeType != 0
	      && pNode1->nodeType != CONTACT) {
	    interiorNode = FALSE;
	  }
	}
	if (interiorNode && xIndex < numXNodes) {
	  pNode1 = nodeArray[xIndex + 1][yIndex];
	  if (pNode1->nodeType != 0
	      && pNode1->nodeType != CONTACT) {
	    interiorNode = FALSE;
	  }
	}
	if (interiorNode && yIndex > 1) {
	  pNode1 = nodeArray[xIndex][yIndex - 1];
	  if (pNode1->nodeType != 0
	      && pNode1->nodeType != CONTACT) {
	    interiorNode = FALSE;
	  }
	}
	if (interiorNode && yIndex < numYNodes) {
	  pNode1 = nodeArray[xIndex][yIndex + 1];
	  if (pNode1->nodeType != 0
	      && pNode1->nodeType != CONTACT) {
	    interiorNode = FALSE;
	  }
	}
	if (interiorNode) {
	  pNode->nodeType = 0;
	}
      }
    }
  }

  /* Delete nodes that do not belong to any domain. */
  numNodes = 0;
  for (yIndex = 1; yIndex <= numYNodes; yIndex++) {
    for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
      pNode = nodeArray[xIndex][yIndex];
      if (pNode->nodeType == 0) {
	/* This node doesn't belong to a domain so delete it. */
	nodeArray[xIndex][yIndex] = NULL;
	FREE(pNode);
      } else {
	numNodes++;
      }
    }
  }
  pDevice->numNodes = numNodes;

  /* Now relabel any remaining nodes that are part of electrodes. */
  setupContacts(pDevice, pElectrode, nodeArray);

  /* Done with the nodes. Now construct the elements and the edges. */
  numEdges = 0;

  /* Generate the horizontal edges and store in a work array. */
  XCALLOC(edgeArrayH, TWOedge **, numXNodes);
  for (xIndex = 1; xIndex < numXNodes; xIndex++) {
    XCALLOC(edgeArrayH[xIndex], TWOedge *, 1 + numYNodes);
  }
  for (yIndex = 1; yIndex <= numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pNode = nodeArray[xIndex][yIndex];
      pNextHNode = nodeArray[xIndex + 1][yIndex];
      if (pNode != NULL &&
	  pNextHNode != NULL) {
	XCALLOC(pEdge, TWOedge, 1);
	numEdges++;
	edgeArrayH[xIndex][yIndex] = pEdge;
      }
    }
  }

  /* Generate the vertical edges and store in a work array. */
  XCALLOC(edgeArrayV, TWOedge **, 1 + numXNodes);
  for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
    XCALLOC(edgeArrayV[xIndex], TWOedge *, numYNodes);
  }
  for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
    for (yIndex = 1; yIndex < numYNodes; yIndex++) {
      pNode = nodeArray[xIndex][yIndex];
      pNextVNode = nodeArray[xIndex][yIndex + 1];
      if (pNode != NULL &&
	  pNextVNode != NULL) {
	XCALLOC(pEdge, TWOedge, 1);
	numEdges++;
	edgeArrayV[xIndex][yIndex] = pEdge;
      }
    }
  }
  pDevice->numEdges = numEdges;

  /* Now construct and count the elements and store the node/edge pointers. */
  numElem = 1;
  for (yIndex = 1; yIndex < numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pNode = nodeArray[xIndex][yIndex];
      pNextHNode = nodeArray[xIndex + 1][yIndex];
      pNextVNode = nodeArray[xIndex][yIndex + 1];
      pNextDNode = nodeArray[xIndex + 1][yIndex + 1];
      if (pNode != NULL &&
	  pNextHNode != NULL &&
	  pNextVNode != NULL &&
	  pNextDNode != NULL) {
	numElem++;
	XCALLOC(pElem, TWOelem, 1);
	pElem->pTLNode = pNode;
	pElem->pTRNode = pNextHNode;
	pElem->pBLNode = pNextVNode;
	pElem->pBRNode = pNextDNode;
	pElem->pTopEdge = edgeArrayH[xIndex][yIndex];
	pElem->pBotEdge = edgeArrayH[xIndex][yIndex + 1];
	pElem->pLeftEdge = edgeArrayV[xIndex][yIndex];
	pElem->pRightEdge = edgeArrayV[xIndex + 1][yIndex];
	pDevice->elemArray[xIndex][yIndex] = pElem;
      } else {
	pDevice->elemArray[xIndex][yIndex] = NULL;
      }
    }
  }

  /* Create and pack the 1D element array. */
  pDevice->numElems = numElem - 1;
  XCALLOC(pDevice->elements, TWOelem *, 1 + numElem);
  numElem = 1;
  for (yIndex = 1; yIndex < numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pElem = pDevice->elemArray[xIndex][yIndex];
      if (pElem != NULL) {
	pDevice->elements[numElem++] = pElem;
      }
    }
  }

  /* Now create back links from elements to nodes. */
  for (yIndex = 1; yIndex < numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pElem = pDevice->elemArray[xIndex][yIndex];
      if (pElem != NULL) {
	pElem->pTLNode->pBRElem = pElem;
	pElem->pTRNode->pBLElem = pElem;
	pElem->pBLNode->pTRElem = pElem;
	pElem->pBRNode->pTLElem = pElem;
	if (xIndex > 1) {
	  pElem->pLeftElem = pDevice->elemArray[xIndex-1][yIndex];
	}
	if (xIndex < numXNodes - 1) {
	  pElem->pRightElem = pDevice->elemArray[xIndex+1][yIndex];
	}
	if (yIndex > 1) {
	  pElem->pTopElem = pDevice->elemArray[xIndex][yIndex-1];
	}
	if (yIndex < numYNodes - 1) {
	  pElem->pBotElem = pDevice->elemArray[xIndex][yIndex+1];
	}
      }
    }
  }

  /* Establish the element types using domain info. */
  for (pD = pDomain; pD != NULL; pD = pD->next) {
    for (pM = pMaterial; pM != NULL; pM = pM->next) {
      if (pD->material == pM->id) {
	break;
      }
    }
    elemType = pM->type;
    for (yIndex = pD->iyLo; yIndex < pD->iyHi; yIndex++) {
      for (xIndex = pD->ixLo; xIndex < pD->ixHi; xIndex++) {
	pElem = pDevice->elemArray[xIndex][yIndex];
	if (pElem != NULL) {
	  pElem->domain = pD->id;
	  pElem->elemType = elemType;
	  pElem->matlInfo = pM;
	}
      }
    }
  }

  /* Establish the edge types. */
  for (yIndex = 1; yIndex < numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pElem = pDevice->elemArray[xIndex][yIndex];
      if (pElem != NULL) {
	for (index = 0; index <= 3; index++) {
	  pEdge = pElem->pEdges[index];
	  pNode = pElem->pNodes[index];
	  pNode1 = pElem->pNodes[(index + 1) % 4];
	  pElem1 = pNode1->pElems[index];

	  if (pNode->nodeType == CONTACT &&
	      pNode1->nodeType == CONTACT) {
	    /* Contact edge */
	    pEdge->edgeType = CONTACT;
	  } else if (pNode->nodeType == SCHOTTKY &&
	      pNode1->nodeType == SCHOTTKY) {
	    /* Schottky edge */
	    pEdge->edgeType = SCHOTTKY;
	  } else if (pElem1 == NULL) {
	    /* Neumann edge */
	    pEdge->edgeType = pElem->elemType;
	  } else if (pElem->elemType != pElem1->elemType) {
	    /* Interface edge */
	    pEdge->edgeType = INTERFACE;
	  } else {		/* Ignore heterojnxns for now */
	    /* Heterojunction or Homojunction edge */
	    pEdge->edgeType = pElem->elemType;
	  }
	}
      }
    }
  }

  resetEvalFlag(pDevice);
  /* Set evaluation flags on nodes and edges. */
  /* Assign the dx and dy terms in the elements while we're at it. */
  for (yIndex = 1; yIndex < numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pElem = pDevice->elemArray[xIndex][yIndex];
      if (pElem != NULL) {
	pElem->dx = xScale[xIndex + 1] - xScale[xIndex];
	pElem->dy = yScale[yIndex + 1] - yScale[yIndex];
	pElem->dxOverDy = pElem->dx / pElem->dy;
	pElem->dyOverDx = pElem->dy / pElem->dx;

	/*
	 * Semiconductor elements take precedence over Insulator elements, so
	 * set them up first.
	 */
	for (index = 0; index <= 3; index++) {
	  if (pElem->elemType == SEMICON) {
	    pNode = pElem->pNodes[index];
	    if (!pNode->evaluated) {
	      pNode->evaluated = TRUE;
	      pElem->evalNodes[index] = TRUE;
	    } else {
	      pElem->evalNodes[index] = FALSE;
	    }
	    pEdge = pElem->pEdges[index];
	    if (!pEdge->evaluated) {
	      pEdge->evaluated = TRUE;
	      pElem->evalEdges[index] = TRUE;
	    } else {
	      pElem->evalEdges[index] = FALSE;
	    }
	  }
	}
      }
    }
  }

  /* Do a second setup pass for Insulator elements */
  /* Do mobility coefficients now, because we set up dx and dy
   * in the previous pass
   */
  for (yIndex = 1; yIndex < numYNodes; yIndex++) {
    for (xIndex = 1; xIndex < numXNodes; xIndex++) {
      pElem = pDevice->elemArray[xIndex][yIndex];
      if (pElem != NULL) {
	pElem->direction = 0;
	pElem->channel = 0;
	pElem->surface = FALSE;
	for (index = 0; index <= 3; index++) {
	  if (pElem->elemType == SEMICON) {
	    doMobCoeffs( pElem, index );
	  } else if (pElem->elemType == INSULATOR) {
	    pNode = pElem->pNodes[index];
	    if (!pNode->evaluated) {
	      pNode->evaluated = TRUE;
	      pElem->evalNodes[index] = TRUE;
	    } else {
	      pElem->evalNodes[index] = FALSE;
	    }
	    pEdge = pElem->pEdges[index];
	    if (!pEdge->evaluated) {
	      pEdge->evaluated = TRUE;
	      pElem->evalEdges[index] = TRUE;
	    } else {
	      pElem->evalEdges[index] = FALSE;
	    }
	  }
	}
      }
    }
  }

  /* Set up the equation numbers for nodes. */
  poiEqn = numEqn = 1;
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (index = 0; index <= 3; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	if (pNode->nodeType != CONTACT) {
	  /* First assign potential equation numbers */
	  if (pNode->nodeType != SCHOTTKY) {
	    pNode->poiEqn = poiEqn++;
	    pNode->psiEqn = numEqn++;
	  }
	  /* Now assign carrier-concentration equation numbers */
	  if (pElem->elemType == INSULATOR) {
	    pNode->nEqn = 0;
	    pNode->pEqn = 0;
	  } else {
	    if (OneCarrier) {
	      /* n and p get same number */
	      pNode->nEqn = numEqn;
	      pNode->pEqn = numEqn++;
	    } else {
	      pNode->nEqn = numEqn++;
	      pNode->pEqn = numEqn++;
	    }
	  }
	} else {		/* This is a contact node. */
	  pNode->poiEqn = 0;
	  pNode->psiEqn = 0;
	  pNode->nEqn = 0;
	  pNode->pEqn = 0;
	}
      }
    }
  }
  pDevice->dimEquil = poiEqn;
  pDevice->dimBias = numEqn;

  /* Open and Print Mesh Output File for Debugging */
  /* Nuked from release version */
#ifdef NOTDEF
  if (!(meshFile = fopen("mesh.out", "w"))) {
    perror("mesh.out");
    exit(-1);
  }
  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (index = 0; index <= 3; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	fprintf(meshFile, "node: %5d %5d %5d %5d\n", pNode->nodeI,
	    pNode->nodeJ, pNode->poiEqn, pNode->psiEqn);
      }
    }
  }
  fflush(meshFile);
  fclose(meshFile);
#endif

  /* Delete work arrays. */
  for (xIndex = 1; xIndex <= numXNodes; xIndex++) {
    FREE(nodeArray[xIndex]);
    FREE(edgeArrayV[xIndex]);
  }
  for (xIndex = 1; xIndex < numXNodes; xIndex++) {
    FREE(edgeArrayH[xIndex]);
  }
  FREE(nodeArray);
  FREE(edgeArrayV);
  FREE(edgeArrayH);

  {
    CiderLoaded(1);
  }
  /*
   * TWOprnMesh( pDevice );
   */
}

void
TWOprnMesh(TWOdevice *pDevice)
{
  int eIndex, index;
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pEdge;
  char *name;

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    fprintf(stderr, "elem %5d:\n", eIndex);
    for (index = 0; index <= 3; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	switch (pNode->nodeType) {
	case SEMICON:
	  name = "semiconductor";
	  break;
	case INSULATOR:
	  name = "insulator";
	  break;
	case CONTACT:
	  name = "contact";
	  break;
	case SCHOTTKY:
	  name = "schottky";
	  break;
	case INTERFACE:
	  name = "interface";
	  break;
	default:
	  name = "unknown";
	  break;
	}
	fprintf(stderr, "node %5d: %s %5d %5d\n", index, name,
	    pNode->nodeI, pNode->nodeJ);
      }
      if (pElem->evalEdges[index]) {
	pEdge = pElem->pEdges[index];
	switch (pEdge->edgeType) {
	case SEMICON:
	  name = "semiconductor";
	  break;
	case INSULATOR:
	  name = "insulator";
	  break;
	case CONTACT:
	  name = "contact";
	  break;
	case SCHOTTKY:
	  name = "schottky";
	  break;
	case INTERFACE:
	  name = "interface";
	  break;
	default:
	  name = "unknown";
	  break;
	}
	fprintf(stderr, "edge %5d: %s\n", index, name);
      }
    }
  }
}

/*
 * We have a separate function for this, so that the setup routines can
 * reset the state pointers without rebuilding the entire mesh.
 */
void
TWOgetStatePointers(TWOdevice *pDevice, int *numStates)
{
  int eIndex, index;
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pEdge;

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (index = 0; index <= 3; index++) {
      if (pElem->evalNodes[index]) {
	pNode = pElem->pNodes[index];
	pNode->nodeState = *numStates;
	*numStates += TWOnumNodeStates;
      }
      if (pElem->evalEdges[index]) {
	pEdge = pElem->pEdges[index];
	pEdge->edgeState = *numStates;
	*numStates += TWOnumEdgeStates;
      }
    }
  }
}

/*
 * This function computes the percentages of the total semiconductor
 * width of an edge on the negative and positive sides of the edge
 */
static void
doMobCoeffs(TWOelem *pElem, int index)
{
  TWOelem *pNElem;
  TWOedge *pEdge;
  double dl1 = 0.0, dl2 = 0.0;

  pNElem = pElem->pElems[ index ];
  pEdge = pElem->pEdges[ index ];

  /* If neighbor is not SEMICON, assign and return */
  if ( (pNElem == NULL) || (pNElem->elemType == INSULATOR) ) {
    if ( index == 0 || index == 3 ) {
      pEdge->kNeg = 0.0;
      pEdge->kPos = 1.0;
    } else {
      pEdge->kNeg = 1.0;
      pEdge->kPos = 0.0;
    }
    return;
  }

  /* Find appropriate dimensions of the elements */
  switch ( index ) {
  case 0:
    dl1 = pNElem->dy;
    dl2 = pElem->dy;
    break;
  case 1:
    dl1 = pElem->dx;
    dl2 = pNElem->dx;
    break;
  case 2:
    dl1 = pElem->dy;
    dl2 = pNElem->dy;
    break;
  case 3:
    dl1 = pNElem->dx;
    dl2 = pElem->dx;
    break;
  }

  /* Assign coefficients */
  pEdge->kNeg = dl1 / (dl1 + dl2);
  pEdge->kPos = dl2 / (dl1 + dl2);

}

static void 
resetEvalFlag(TWOdevice *pDevice)
{
  int index, eIndex;
  TWOelem *pElem;

  for (eIndex = 1; eIndex <= pDevice->numElems; eIndex++) {
    pElem = pDevice->elements[eIndex];
    for (index = 0; index <= 3; index++) {
      pElem->pNodes[index]->evaluated = FALSE;
      pElem->pEdges[index]->evaluated = FALSE;
    }
  }
}
