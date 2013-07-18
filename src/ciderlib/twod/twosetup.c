/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

/**********
Two-Dimensional Numerical Device Setup Routines
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "ngspice/carddefs.h"		/* XXX Not really modular if we need this. */
#include "twoddefs.h"
#include "twodext.h"
#include "ngspice/cidersupt.h"


/* compute node parameters */
void TWOsetup(TWOdevice *pDevice)
{
  double temp1, deltaEg, avgConc, totalConc, absNetConc;
  double ncv0, dBand, dNie, psiBand[4];
  double *xScale = pDevice->xScale;
  double *yScale = pDevice->yScale;
  int index, eIndex;
  int numContactNodes;
  TWOnode *pNode;
  TWOelem *pElem;
  TWOedge *pEdge;
  TWOcontact *pC;
  TWOmaterial *info;

  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    info = pElem->matlInfo;

    pElem->dx = xScale[ pElem->pTRNode->nodeI ]-xScale[ pElem->pTLNode->nodeI ];
    pElem->dy = yScale[ pElem->pBLNode->nodeJ ]-yScale[ pElem->pTLNode->nodeJ ];
    pElem->epsRel = info->eps;

    if ( pElem->elemType == INSULATOR ) {
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  if (pNode->nodeType == CONTACT) {
	    pNode->eaff = PHI_METAL;
	    pNode->eg = 0.0;
	  } else {
	    pNode->eaff = info->affin;
	    pNode->eg = info->eg0;
	  }
	}
      }
    } else if ( pElem->elemType == SEMICON ) {
      ncv0 = sqrt( info->nc0 ) * sqrt( info->nv0 );
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalNodes[ index ] ) {
	  pNode = pElem->pNodes[ index ];
	  
          /* Narrowing of Energy-Band Gap */
	  if (BandGapNarrowing) {
	    absNetConc = ABS( pNode->netConc );
	    if ( pNode->netConc > 0.0 ) {
	      temp1 = log( absNetConc / info->nrefBGN[ELEC] );
	      deltaEg = - info->dEgDn[ELEC] * (temp1 + sqrt(temp1*temp1 + 0.5));
	      pNode->eg = info->eg0 + deltaEg;
	    } else if ( pNode->netConc < 0.0 ) {
	      temp1 = log( absNetConc / info->nrefBGN[HOLE] );
	      deltaEg = - info->dEgDn[HOLE] * (temp1 + sqrt(temp1*temp1 + 0.5));
	      pNode->eg = info->eg0 + deltaEg;
	    } else {
	      pNode->eg = info->eg0;
	    }
	  } else {
	    pNode->eg = info->eg0;
	  }
	  pNode->nie = ncv0 * exp ( -0.5 * pNode->eg / Vt );
	  pNode->eaff = info->affin;
	  /* Save band structure parameter. */
	  psiBand[ index ] = - info->refPsi;
	  
	  /* Ionized-Impurity-Scattering Reduction of Carrier Lifetime */
	  if (ConcDepLifetime) {
	    totalConc = pNode->totalConc;
	    temp1 = 1.0 / ( 1.0 + totalConc / info->nrefSRH[ELEC] );
	    pNode->tn = info->tau0[ELEC] * temp1;
	    temp1 = 1.0 / ( 1.0 + totalConc / info->nrefSRH[HOLE] );
	    pNode->tp = info->tau0[HOLE] * temp1;
	  } else {
	    pNode->tn = info->tau0[ELEC];
	    pNode->tp = info->tau0[HOLE];
	  }
	}
      }
      
      for ( index = 0; index <= 3; index++ ) {
	if ( pElem->evalEdges[ index ] ) {
	  pEdge = pElem->pEdges[ index ];

	  /* Fixed Interface Charge */
	  pEdge->qf = 0.0;

          /* Variable Band Built-In Potential */
	  if ( index <= 1 ) {
	    dBand = psiBand[index+1] - psiBand[index];
	    dNie = log( pElem->pNodes[index+1]->nie /
	                pElem->pNodes[index]->nie );
	    pEdge->dCBand = dBand + dNie;
	    pEdge->dVBand = - dBand + dNie;
	  } else {
	    dBand = psiBand[index] - psiBand[(index+1)%4];
	    dNie = log( pElem->pNodes[index]->nie /
	                pElem->pNodes[(index+1)%4]->nie );
	    pEdge->dCBand = dBand + dNie;
	    pEdge->dVBand = - dBand + dNie;
	  }
	}
      }
      
      /* Evaluate conc.-dep. mobility. */
      /* Average conc of all four nodes. */
      avgConc = 0.25*(pElem->pTLNode->totalConc + pElem->pTRNode->totalConc +
		      pElem->pBRNode->totalConc + pElem->pBLNode->totalConc);
      MOBconcDep( info, avgConc, &pElem->mun0, &pElem->mup0 );
    }
  }

  for ( pC = pDevice->pFirstContact; pC != NULL; pC = pC->next ) {
    numContactNodes = pC->numNodes;
    for ( index = 0; index < numContactNodes; index++ ) {
      pNode = pC->pNodes[ index ];
      pNode->eaff = pC->workf;		/* Affinity aka work function */
    }
  }
}

/* Transfer BC info from card to nodes and edges. */
static void
TWOcopyBCinfo(TWOdevice *pDevice, TWOelem *pElem, BDRYcard *card, int index )
{
  TWOnode *pNode;
  TWOelem *pNElem;
  TWOedge *pEdge;
  TWOmaterial *info;
  TWOchannel *newChannel;
  int eIndex, nIndex;
  int direction = index%2;
  double length, area, width, layerWidth;
  double dop, na = 0.0, nd = 0.0;

  /* First add fixed charge. */
  pEdge = pElem->pEdges[index];
  pEdge->qf += card->BDRYqf;

  /* Now add surface recombination. */
  if ( direction == 0 ) {	/* Horizontal Edge */
    length = 0.5 * pElem->dx;
  } else {
    length = 0.5 * pElem->dy;
  }
  for (nIndex = 0; nIndex <= 1; nIndex++) {
    pNode = pElem->pNodes[ (index+nIndex)%4 ];
    /* Compute semiconductor area around this node. */
    area = 0.0;
    for (eIndex = 0; eIndex <= 3; eIndex++) {
      pNElem = pNode->pElems[eIndex];
      if (pNElem != NULL && pElem->elemType == SEMICON) {
	area += 0.25 * pElem->dx * pElem->dy;
      }
    }
    if (card->BDRYsnGiven) {
      pNode->tn = pNode->tn /
	(1.0 + ((card->BDRYsn * TNorm)*length*pNode->tn) / area);
    }
    if (card->BDRYspGiven) {
      pNode->tp = pNode->tp /
	(1.0 + ((card->BDRYsp * TNorm)*length*pNode->tp) / area);
    }
    /* Compute doping just in case we need it later. */
    na += 0.5 * pNode->na;
    nd += 0.5 * pNode->nd;
  }

  /* Finally do surface layer. */
  pNElem = pElem->pElems[index];
  if (card->BDRYlayerGiven && SurfaceMobility && pElem->elemType == SEMICON
      && pElem->channel == 0 && pNElem && pNElem->elemType == INSULATOR
      && pElem->pNodes[index]->nodeType != CONTACT &&
      pElem->pNodes[(index+1)%4]->nodeType != CONTACT ) {
    /* Find the layer width. */
    layerWidth = card->BDRYlayer;
    if (card->BDRYlayer <= 0.0) {  /* Need to compute extrinsic Debye length */
      info = pElem->matlInfo;
      dop = MAX(MAX(na,nd),info->ni0);
      layerWidth = sqrt((Vt * info->eps) / (CHARGE * dop));
    }

    /* Add a channel to the list of channels. */
    XCALLOC( newChannel, TWOchannel, 1);
    newChannel->pSeed = pElem;
    newChannel->pNElem = pNElem;
    newChannel->type = index;
    if (pDevice->pChannel != NULL) {
      newChannel->id = pDevice->pChannel->id + 1;
      newChannel->next = pDevice->pChannel;
    } else {
      newChannel->id = 1;
      newChannel->next = NULL;
    }
    pDevice->pChannel = newChannel;

    /* Now add elements to channel until we're more than layerWidth away
     * from the interface.  If we encounter a missing element or an
     * element that's already part of a different channel, quit.
     * The seed element is at the surface.
     */
    width = 0.0;
    eIndex = (index+2)%4;
    pElem->surface = TRUE;
    while (width < layerWidth && pElem && pElem->channel == 0) {
      pElem->channel = newChannel->id;
      pElem->direction = direction;
      /*
       * Surface mobility is normally concentration-independent in
       * the default model. Overwrite concentration-dependent value
       * calculated earlier unless matching of low-field surface
       * and bulk mobilities is requested.
       */
      if (!MatchingMobility) {
	pElem->mun0 = pElem->matlInfo->mus[ELEC];
	pElem->mup0 = pElem->matlInfo->mus[HOLE];
      }
      if ( direction == 0 ) {
	width += pElem->dy;
      } else {
	width += pElem->dx;
      }
      pElem = pElem->pElems[ eIndex ];
    }
  }
}

/* Compute boundary condition parameters. */
void TWOsetBCparams(TWOdevice *pDevice, BDRYcard *cardList)
{
  int index, xIndex, yIndex;		/* Need to access in X/Y order. */
  TWOelem *pElem, *pNElem;
  BDRYcard *card;

  for ( card = cardList; card != NULL; card = card->BDRYnextCard ) {
    for (xIndex = card->BDRYixLow; xIndex < card->BDRYixHigh; xIndex++) {
      for (yIndex = card->BDRYiyLow; yIndex < card->BDRYiyHigh; yIndex++) {
	pElem = pDevice->elemArray[ xIndex ][ yIndex ];
	if (pElem != NULL) {
	  if (pElem->domain == card->BDRYdomain) {
	    for (index = 0; index <= 3; index++) {
	      if (pElem->evalEdges[index]) {
		pNElem = pElem->pElems[index];
		if (card->BDRYneighborGiven) {
		  if (pNElem && pNElem->domain == card->BDRYneighbor) {
		    /* Found an interface edge. */
		    TWOcopyBCinfo( pDevice, pElem, card, index );
		  }
		} else {
		  if (!pNElem || pNElem->domain != pElem->domain) {
		    /* Found a boundary edge. */
		    TWOcopyBCinfo( pDevice, pElem, card, index );
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
}

void TWOnormalize(TWOdevice *pDevice)
{
  int index, eIndex;
  TWOelem *pElem;
  TWOnode *pNode;
  TWOedge *pEdge;
  
  for ( eIndex = 1; eIndex <= pDevice->numElems; eIndex++ ) {
    pElem = pDevice->elements[ eIndex ];
    
    pElem->dx /= LNorm;
    pElem->dy /= LNorm;
    pElem->epsRel /= EpsNorm;
    for ( index = 0; index <= 3; index++ ) {
      if ( pElem->evalNodes[ index ] ) {
	pNode = pElem->pNodes[ index ];
	pNode->netConc /= NNorm;
	pNode->nd /= NNorm;
	pNode->na /= NNorm;
	pNode->nie /= NNorm;
	pNode->eg /= VNorm;
	pNode->eaff /= VNorm;
      }
      if ( pElem->evalEdges[ index ] ) {
	pEdge = pElem->pEdges[ index ];
	pEdge->qf /= NNorm*LNorm;
      }
    }
  }
}
