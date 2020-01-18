/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/
/**********
Mesh Setup & Query Routines.
**********/

/* Imports */
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/meshdefs.h"
#include "ngspice/meshext.h"
#include "ngspice/gendev.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* Local Constants */
#define CMP_TOL	1.0e-9		/* Tolerance on (double) comparisons */
#define RAT_TOL 1.0e-6		/* Error allowed in ratio calc's */
#define RAT_LIM	50		/* Maximum number of ratio iterations */
#define UM_TO_CM 1.0e-4		/* Micron to centimeter conversion */

/* Forward Declarations */
static int oneSideSpacing( double, double, double,
    double *, int * );
static int twoSideSpacing( double, double, double, double,
    double *, double *, int *, int * );
static int maxLimSpacing( double, double, double, double,
    double *, int *, int * );
static int oneSideRatio( double, double, double *, int );
static int twoSideRatio( double, double, double, double *, int, int );
static int MESHspacing( MESHcard *, double *, double *, int *, int *, int * );

/* END OF HEADER */


/*
 * Name:	MESHmkArray
 * Purpose:	Turn a coordinate list into a coordinate array.
 * Formals:	< I > coordList: a sorted list of all the coordinates
 *		< I > numCoords: the length of the listi, if 0 find it
 * Returns:	a (double) array of those coordinates, with length in a[0]
 * Users:	routines used to create the final mesh
 * Calls:	(none)
 */
double *
MESHmkArray(MESHcoord *coordList, int numCoords)
{
  double *array = NULL;
  MESHcoord *coord;

  if ( numCoords <= 0 ) {
    numCoords = 0;
    for ( coord = coordList; coord != NULL; coord = coord->next ) {
      numCoords++;
    }
  }
  if ( numCoords != 0 ) {
    XALLOC( array, double, 1 + numCoords );
    numCoords = 0;
    array[ 0 ] = (double) numCoords;
    numCoords = 1;
    for ( coord = coordList; coord != NULL; coord = coord->next ) {
      array[ numCoords++ ] = coord->location;
    }
    return array;
  }
  else {
    return NULL;
  }
  /* NOTREACHED */
}


/*
 * Name:	MESHiBounds
 * Purpose:	Find the minimum and maximum indices in a mesh list.
 * Formals:	< I > coordList: a sorted list of all the coordinates
 * 		< O > ixMin: the minimum index
 *		< O > ixMax: the maximum index
 * Returns:	(none)
 * Users:	routines wanting to fine the ends of a mesh
 * Calls:	(none)
 */
void
MESHiBounds(MESHcoord *coordList, int *ixMin, int *ixMax)
{
  MESHcoord *last;

  if (coordList) {
   *ixMin = coordList->number;
   for ( last = coordList; last->next != NULL; last = last->next )
      ;
   *ixMax = last->number;
  }
  else {
   *ixMin = *ixMax = -1;
  }
}


/*
 * Name:	MESHlBounds
 * Purpose:	Find the minimum and maximum locations in a mesh list.
 * Formals:	< I > coordList: a sorted list of all the coordinates
 * 		< O > lcMin: the minimum location
 *		< O > lcMax: the maximum location
 * Returns:	(none)
 * Users:	routines wanting to find the ends of a mesh
 * Calls:	(none)
 */
void
MESHlBounds(MESHcoord *coordList, double *lcMin, double *lcMax)
{
  MESHcoord *last;

  if (coordList) {
   *lcMin = coordList->location;
   for ( last = coordList; last->next != NULL; last = last->next )
      ;
   *lcMax = last->location;
  }
  else {
   *lcMin = *lcMax = 0.0;
  }
}


/*
 * Name:	MESHlocate
 * Purpose:	Finds the index of the MESHcoord nearest to a location.
 * Formals:	< I > coordList: a sorted list of all available coordinates
 *		< I > location: the location to find
 * Returns:	index / -1 (list empty)
 * Users:	routines that convert distances to indices
 * Calls:	(none)
 */
int
MESHlocate(MESHcoord *coordList, double location)
{
  MESHcoord *coord, *prevCoord = NULL;
  int index;

/* Find the coordinates which flank the location. */
  for ( coord = coordList; coord != NULL; coord = coord->next ) {
    if ( coord->location > location ) break;
    prevCoord = coord;
  }

/* Get the index. */
  if (prevCoord && coord) {
    if ( location <= (prevCoord->location + coord->location) / 2.0 ) {
      index = prevCoord->number;
    }
    else {
      index = coord->number;
    }
  }
  else if ( coord ) {
    index = coord->number;
  }
  else if ( prevCoord ) {
    index = prevCoord->number;
  }
  else {
    index = -1;
  }
  return( index );
}



/*
 * Name:	MESHcheck
 * Purpose:	Checks a list of mesh cards for input errors.
 * Formals:	<I/O> cardList: pointer to head of linked list of MESHcard's
 *		< I > dim: 'x', 'y' or 'z' dimension
 * Returns:     OK / E_PRIVATE
 * Users:	setup routines
 * Calls:	error-message handler
 */
int
MESHcheck(char dim, MESHcard *cardList)
{
  MESHcard *card;
  int cardNum = 0;
  double locStart = 0.0, locEnd;
  double ratio;
  int error = OK;

  if ( cardList == NULL ) {
    SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card list is empty", dim );
    locEnd = locStart;
    return( E_PRIVATE );
  }

  for ( card = cardList; card != NULL; card = card->MESHnextCard ) {
    cardNum++;

/* Am I trying to find number of nodes directly & indirectly? */
    if (card->MESHnumberGiven && card->MESHratioGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "%c.mesh card %d uses both number and ratio - number ignored", dim, cardNum );

      card->MESHnumberGiven = FALSE;
    }

/* Will I be able to locate endpoints? */
    if (!card->MESHlocationGiven && !card->MESHwidthGiven) {
      SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d has no distances", dim, cardNum );
      locEnd = locStart;
      error = E_PRIVATE;
    }
    else if (card->MESHlocationGiven && card->MESHwidthGiven) {
      SPfrontEnd->IFerrorf( ERR_INFO, "%c.mesh card %d uses both location and width - location ignored", dim, cardNum );

      card->MESHlocationGiven = FALSE;
      locEnd = locStart + card->MESHwidth;
    }
    else if (card->MESHlocationGiven) {
      locEnd = card->MESHlocation;
      if (cardNum == 1) locStart = locEnd;
    }
    else {
      locEnd = locStart + card->MESHwidth;
    }

/* Are the endpoints in the wrong order? */
    if ( locEnd - locStart < - CMP_TOL ) {
      SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d uses negative width", dim, cardNum );
      error = E_PRIVATE;
    }

/* Are the endpoints too close together? */
    else if ( (locEnd - locStart <= CMP_TOL) ) {
      if ( !(cardNum == 1 && locStart == locEnd) ) {
	SPfrontEnd->IFerrorf( ERR_INFO, "%c.mesh card %d has negligible width - ignored", dim, cardNum );
	locStart = locEnd;
      }
    }

/* Is the ratio out of the acceptable range? */
    if (card->MESHratioGiven) {
      ratio = card->MESHratio;
    }
    else {
      ratio = 1.0;
    }
    if ((ratio < 1.0) || (ratio > 10.0)) {
      SPfrontEnd->IFerrorf( ERR_INFO, "%c.mesh card %d has ratio out of range - reset to 1.0", dim, cardNum );
      ratio = 1.0;
    }

/* Check sizes of h.start, h.end and h.max. */
    if ((card->MESHhStartGiven && (card->MESHhStart <= 0.0)) ||
	(card->MESHhEndGiven && (card->MESHhEnd <= 0.0)) ||
	(card->MESHhMaxGiven && (card->MESHhMax <= 0.0))) {
      SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d wants to use a non-positive spacing", dim, cardNum );
      error = E_PRIVATE;
    }

/* Is the max spacing being used improperly? */
    if (card->MESHhMaxGiven && (
	( card->MESHhStartGiven &&  card->MESHhEndGiven) ||
	(!card->MESHhStartGiven && !card->MESHhEndGiven))) {
      SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d needs to use one of h.start or h.end with h.max", dim, cardNum );
      error = E_PRIVATE;
    }
    else if (card->MESHhMaxGiven && card->MESHhStartGiven) {
      if (card->MESHhStart > card->MESHhMax) {
	SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d wants h.start > h.max", dim, cardNum );
	error = E_PRIVATE;
      }
      else {
	card->MESHhEnd = card->MESHhMax;
      }
    }
    else if (card->MESHhMaxGiven && card->MESHhEndGiven) {
      if (card->MESHhEnd > card->MESHhMax) {
	SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d wants h.end > h.max", dim, cardNum );
	error = E_PRIVATE;
      }
      else {
	card->MESHhStart = card->MESHhMax;
      }
    }

/* Return now if anything has failed. */
    if (error) return(error);

/* Note: at this point we still aren't sure whether node numbers are OK. */

/* Fill-in newly computed information. */
    card->MESHlocStart = locStart;
    card->MESHlocEnd = locEnd;
    card->MESHratio = ratio;

/* Advance current location. */
    locStart = locEnd;
  }
  return(OK);
}


/*
 * Name:	geomSum
 * Purpose:	Computes the sum of n terms of a geometric series.
 * Formals:	< I > r: ratio of one term to the next
 *		< I > n: number of terms to sum
 * Returns:	sum / 0.0
 * Users:	spacing routines
 * Calls:	pow
 */
static double
geomSum(double r, double n)
{
  double sum;

  if ((r < 0.0) || (n <= 0.0)) {
    sum = 0.0;
  }
  else if (r == 0.0) {
    sum = 1.0;
  }
  else {
    if (ABS(r - 1.0) < 1.0e-4) {
      sum = n * (1.0 + (n - 1.0)*(r - 1.0)/2.0);
    }
    else {
      sum = (1.0 - pow(r,n))/(1.0 - r);
    }
  }
  return( sum );
}


/*
 * Name:	addCoord
 * Purpose:	add a new coordinate to the tail of a linked list
 * Formals:	<I/O> head: head of linked list
 *		<I/O> tail: tail of linked list
 *		< I > number: node number of coordinate
 *		< I > location: location of coordinate
 * Returns:	OK / E_NOMEM
 * Users:	MESHsetup
 * Calls:	memory allocator
 */
static int
addCoord(MESHcoord **head, MESHcoord **tail, int number, double location)
{
  MESHcoord *newCoord;

  if (*head == NULL) {
    RALLOC( *tail, MESHcoord, 1 );
    newCoord = *head = *tail;
  }
  else {
    RALLOC( (*tail)->next, MESHcoord, 1 );
    newCoord = *tail = (*tail)->next;
  }
  newCoord->next = NULL;
  newCoord->number = number;
  newCoord->location = location * UM_TO_CM;
  return(OK);
}


/*
 * Name:	MESHsetup
 * Purpose:	Converts a list of input MESHcard's to a list of MESHcoord's.
 *		Expansion is performed so that node numbers in the final list
 *		increase by one from coordinate to coordinate.  The list
 *		will grow until the input ends or a bad card is found.
 * Formals:	< I > dim: 'x', 'y', or 'z' dimension
 *		<I/O> cardList: the list of input cards
 *		< O > coordList: the final list of coordinates
 *		< O > numCoords: the number of coords in coordList
 * Returns:	OK / E_PRIVATE
 * Users:	numerical device setup routines
 * Calls:	MESHcheck, MESHspacing, error-message handler
 */
int
MESHsetup(char dim, MESHcard *cardList, MESHcoord **coordList, int *numCoords)
{
  MESHcard *card;
  MESHcoord *endCoord;
  int cardNum = 0;
  int i, totCoords, numStart=1, numEnd = 0, nspStart, nspEnd, nspMax, nspLeft;
  double locStart, locEnd =  0.0, location, space;
  double hStart, hEnd, hMax, hBig;
  double ratStart, ratEnd;
  int error = OK;

/* Initialize list of coordinates. */
  *coordList = endCoord = NULL;
  *numCoords = totCoords = 0;

/* Check the card list. */
  if ((error = MESHcheck( dim, cardList )) != 0) return( error );

/* Print info header. */ 
#ifdef NOTDEF
  fprintf( stdout, " %c.Mesh Card Information\n", toupper_c(dim) );
  fprintf( stdout, "-------------------------\n" );
  fprintf( stdout, " %3s %3s %3s %9s %9s %9s %9s %9s %9s\n",
      "n.s", "n.m", "n.e", "l.e", "h.s", "h.e", "h.m", "r.s", "r.e" );
#endif

  for ( card = cardList; card != NULL; card = card->MESHnextCard ) {
    cardNum++;
    locStart = card->MESHlocStart;
    locEnd = card->MESHlocEnd;

    if (locEnd == locStart) {		/* This card has no width. */
/* First update the node number. */
      if (card->MESHnumberGiven) {
	if (card->MESHlocationGiven) {	/* Absolute node number given */
	  numEnd = card->MESHnumber;
	  if (cardNum == 1) numStart = numEnd;
	}
	else {				/* Number of spaces instead */
	  numEnd = numStart + card->MESHnumber;
	}
      }
/* Are node numbers in the wrong order? */
      if ( numEnd < numStart ) {
	SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d has out-of-order node numbers ( %d > %d )", dim, cardNum, numStart, numEnd );
	error = E_PRIVATE;
      }
    }
    else {				/* This card has some width. */
/* First update the node number. */
      if (card->MESHnumberGiven) {	/* Uniform mesh */
	if (card->MESHlocationGiven) {	/* Absolute node number given */
	  numEnd = card->MESHnumber;
	  if (cardNum == 1) numStart = numEnd;
	  nspStart = numEnd - numStart;
	}
	else {				/* Number of spaces instead */
	  nspStart = card->MESHnumber;
	  numEnd = numStart + nspStart;
	}
	ratStart = 1.0;
	ratEnd = 0.0;
	nspEnd = 0;
	nspMax = 0;
        if ( nspStart > 0 ) {
	  card->MESHhStart = (locEnd - locStart) / (double)nspStart;
	  card->MESHhEnd = 0.0;
	}
      }
      else {				/* Nonuniform mesh */
	error = MESHspacing( card, &ratStart, &ratEnd,
	    &nspStart, &nspMax, &nspEnd );
	if (error) {
	  SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d can't be spaced automatically", dim, cardNum );
	  return( error );
	}
	else {
	  numEnd = numStart + nspStart + nspMax + nspEnd;
	}
      }
/* Are the node numbers properly ordered? */
      if ( numEnd <= numStart ) {
	SPfrontEnd->IFerrorf( ERR_FATAL, "%c.mesh card %d results in out-of-order node numbers ( %d > %d )", dim, cardNum, numStart, numEnd );
	error = E_PRIVATE;
      }
      else {
/* Create the new MESHcoord's. */
        hStart = card->MESHhStart;
	hEnd = card->MESHhEnd;
	hMax = card->MESHhMax;
	hBig = 0.0;

/* Generate the first coord in this section */
	location = locStart;
	error = addCoord( coordList, &endCoord, ++totCoords, location );
	if (error) return(error);

/* Generate new coords for the starting section. */
        nspLeft = nspStart + nspMax + nspEnd;
	if ( nspStart != 0 ) {
	  hBig = MAX( hBig, hStart*pow( ratStart, (double) (nspStart - 1) ) );
	  space = hStart;
	  for ( i = 0; (i < nspStart)&&(nspLeft > 1); i++, nspLeft-- ) {
	    location += space;
	    space *= ratStart;
	    error = addCoord( coordList, &endCoord, ++totCoords, location );
	    if (error) return(error);
	  }
	}

/* Generate new coords for the maximum section. */
	if ( nspMax != 0 ) {
	  hBig = MAX( hBig, hMax );
	  space = hMax;
	  for ( i = 0; (i < nspMax)&&(nspLeft > 1); i++, nspLeft-- ) {
	    location += space;
	    error = addCoord( coordList, &endCoord, ++totCoords, location );
	    if (error) return(error);
	  }
	}

/* Generate new coords for the ending section. */
	if ( nspEnd != 0 ) {
	  hBig = MAX( hBig, hEnd*pow( ratEnd, (double) (nspEnd - 1) ) );
	  space = hEnd * pow( ratEnd, (double) (nspEnd - 1) );
	  for ( i = 0; (i < nspEnd)&&(nspLeft > 1); i++, nspLeft-- ) {
	    location += space;
	    space /= ratEnd;
	    error = addCoord( coordList, &endCoord, ++totCoords, location );
	    if (error) return(error);
	  }
	}
#ifdef NOTDEF
	fprintf( stdout, "             %9.5f\n", locStart );
	fprintf( stdout,
	    " %3d %3d %3d           %9.5f %9.5f %9.5f %9.5f %9.5f\n",
	    nspStart, nspMax, nspEnd,
	    hStart, hEnd, hBig, ratStart, ratEnd );
#endif
      }
    }

/* Return now if anything has failed. */
    if (error) return(error);

/* Advance the node number. */
    numStart = numEnd;
  }

/*
 * If the mesh is not empty, then the loop above has exited before
 * adding the final coord to the list. So we need to do that now.
 */
  if (*coordList != NULL) {
    error = addCoord( coordList, &endCoord, ++totCoords, locEnd );
    if (error) return(error);
#ifdef NOTDEF
    fprintf( stdout, "             %9.5f\n", locEnd );
#endif
  }
#ifdef NOTDEF
  fprintf( stdout, "\n" );
#endif
  *numCoords = totCoords;
  return(OK);
}



/*
 * Name:	MESHspacing
 * Purpose:	Find ratios, spacings, and node numbers for a mesh span.
 * Formals:	< I > card: the input card for this span
 *		< O > rS: ratio found for spacings at start of span
 *		< O > rE: ratio found for spacings at end of span
 *		< O > nS: number of start spaces
 *		< O > nM: number of max spaces
 *		< O > nE: number of end spaces
 * Returns:	OK / E_PRIVATE
 * Users:	MESHsetup
 * Calls:	oneSideSpacing, twoSideSpacing, maxLimSpacing
 */
static int
MESHspacing(MESHcard *card, double *rS, double *rE, int *nS, int *nM, int *nE)
{
  int error = OK;
  int hStartGiven = card->MESHhStartGiven;
  int hEndGiven = card->MESHhEndGiven;
  int hMaxGiven = card->MESHhMaxGiven;
  double hS = card->MESHhStart;
  double hE = card->MESHhEnd;
  double hM = card->MESHhMax;
  double rW = card->MESHratio;		/* The ratio wanted */
  double width;

  width = card->MESHlocEnd - card->MESHlocStart;

/* Call subsidiary routine depending on how flags are set. */
  if      (!hStartGiven &&  hEndGiven && !hMaxGiven ) {
/* End section only */
    error = oneSideSpacing( width, hE, rW, rE, nE );
    *nM = *nS = 0;
    *rS = 0.0;
  }
  else if ( hStartGiven && !hEndGiven && !hMaxGiven ) {
/* Start section only */
    error = oneSideSpacing( width, hS, rW, rS, nS );
    *nM = *nE = 0;
    *rE = 0.0;
  }
  else if ( hStartGiven &&  hEndGiven && !hMaxGiven ) {
/* Both a start and an end section */
    error = twoSideSpacing( width, hS, hE, rW, rS, rE, nS, nE );
    *nM = 0;
  }
  else if ( hStartGiven && !hEndGiven &&  hMaxGiven ) {
/* Limited size in end section */ 
    error = maxLimSpacing(  width, hS, hM, rW, rS, nS, nM );
    *nE = 0;
    *rE = 1.0;
  }
  else if (!hStartGiven &&  hEndGiven &&  hMaxGiven ) {
/* Limited size in start section */ 
    error = maxLimSpacing(  width, hE, hM, rW, rE, nE, nM );
    *nS = 0;
    *rS = 1.0;
  }
  else if ( hStartGiven &&  hEndGiven &&  hMaxGiven ) {
/* Limited size somewhere in the middle */
    /* NOT IMPLEMENTED */
    /*
    error = midLimSpacing(  width, hS, hE, hM, rW, rS, rE, nS, nE, nM );
    */
    error = E_PRIVATE;
  }
  else {
/* Illegal situations */
    error = E_PRIVATE;
  }

  return( error );
}



/*
 * Name:	stepsInSpan
 * Purpose:	Finds the number of steps needed to go a given distance
 *		while increasing each step by a given ratio.
 * Formals:	< I > width: size of total distance
 *		< I > spacing: size of initial step
 *		< O > ratio: increase with each step
 * Returns:	number of steps
 * Users:	spacing routines
 * Calls:	log
 */
static double
stepsInSpan(double width, double spacing, double ratio)
{
  double nSpaces;

/* Handle ratios near 1.0 specially. */
  if ( ABS(ratio - 1.0) < 1.0e-4 ) {
    nSpaces = (width/spacing);
  }
  else {
    nSpaces = (log(1.0-width*(1.0-ratio)/spacing)/log(ratio));
  }
  return(nSpaces);
}


/*
 * Name:	oneSideSpacing
 * Purpose:	Find compatible number of spaces and ratio when the spacing
 *		is constrained at one end of a span.
 * Formals:	< I > width: width of the span
 *		< I > spacing: spacing constraint
 *		< I > rWanted: ideal ratio of one spacing to the next
 *		< O > rFound: actual ratio discovered
 *		< O > nFound: number of spaces found
 * Returns:	OK / E_PRIVATE
 * Users:	MESHspacing
 * Calls:	oneSideRatio, stepsInSpan
 */
static int
oneSideSpacing(double width, double spacing, double rWanted, double *rFound,
               int *nFound)
{
  int nSpaces;			/* Number of spaces */
  double rTemp1, rTemp2;	/* For temporarily calc'ed ratios */

/* Make sure we can take at least one step. */
  if ( width < spacing ) {
    SPfrontEnd->IFerrorf( ERR_WARNING, "one-sided spacing can't find an acceptable solution\n");
    *rFound = 0.0;
    *nFound = 0;
    return(E_PRIVATE);
  }

  nSpaces = (int)stepsInSpan( width, spacing, rWanted );

/* Check to see whether a flat span is acceptable. */
  if ( ABS(nSpaces*spacing - width) < 1.0e-3*spacing ) {
    *rFound = 1.0;
    *nFound = nSpaces;
    return( OK );
  }
  else if ( ABS((nSpaces+1)*spacing - width) < 1.0e-3*spacing ) {
    *rFound = 1.0;
    *nFound = nSpaces + 1;
    return( OK );
  }

/* Too much error involved in flat span means we have to ramp up. */
  rTemp1 = rTemp2 = rWanted;
  oneSideRatio( width, spacing, &rTemp1, nSpaces );
  oneSideRatio( width, spacing, &rTemp2, nSpaces+1 );
  if ( (rTemp1 == 0.0) && (rTemp2 == 0.0) ) {
    SPfrontEnd->IFerrorf( ERR_WARNING, "one-sided spacing can't find an acceptable solution\n");
    *rFound = 0.0;
    *nFound = 0;
    return(E_PRIVATE);
  }
  else if (rTemp1 == 0.0) {
    *rFound = rTemp2;
    *nFound = nSpaces + 1;
  }
  else if (rTemp2 == 0.0) {
    *rFound = rTemp1;
    *nFound = nSpaces;
  }
  else if (ABS(rWanted-rTemp2) < 4.0*ABS(rWanted-rTemp1)) {
    *rFound = rTemp2;
    *nFound = nSpaces + 1;
  }
  else {
    *rFound = rTemp1;
    *nFound = nSpaces;
  }
  return(OK);
}



/*
 * Name:	oneSideRatio
 * Purpose:	Compute the unique ratio 'r' which satisfies the following
 *		constraint:  w = hs*(1-r^ns)/(1-r)
 * Formals:	< I > w : width of a span
 *		< I > hs: step at one end of the span
 *		<I/O> argRatio: ratio found, contains initial guess at entry
 *		< I > ns: number of spaces to use in the span
 * Returns:	OK / E_PRIVATE
 * Users:	oneSideSpacing, maxLimSpacing
 * Calls:	error-message handler
 */
static int
oneSideRatio(double w, double hs, double *argRatio, int ns)
{
  double funcLow, funcUpp, func;
  double ratLow, ratUpp, ratio = *argRatio;
  double dns = (double)ns;
  int i;

/* Get lower bound on solution. */
  ratLow = 0.0;
  funcLow = hs - w;
  if ((funcLow > 0.0) || ((funcLow < 0.0)&&(ns <= 1))) {
    *argRatio = 0.0;
    return(E_PRIVATE);
  }

/* Find upper bound on solution. */
  ratUpp = ratio;
  do {
    ratUpp += 0.2;
    funcUpp = hs*geomSum(ratUpp, dns) - w;
  } while (funcUpp < 0.0);

/* Do bisections to find new ratio. */
  for ( i=0; i < RAT_LIM; i++ ) {
    ratio = ratLow + 0.5 * (ratUpp - ratLow);
    func = hs*geomSum(ratio, dns) - w;
    if ((func == 0.0) || (ratUpp - ratLow < RAT_TOL)) break;

    funcLow = hs*geomSum(ratLow, dns) - w;
    if (funcLow*func > 0.0) {
      ratLow = ratio;
    }
    else {
      ratUpp = ratio;
    }
  }

  if (i == RAT_LIM) { /* No solution found */
    *argRatio = 0.0;
    return(E_PRIVATE);
  }
  else {
    *argRatio = ratio;
    return(OK);
  }
}



/* Name:	quadRoots
 * Purpose:	Find real roots of a quadratic equation if they exist.
 * Formals:	< I > a, b, c: coefficients in ax^2+bx+c
 *		< O > rp: the root using the positive sqrt value
 *		< O > rn: the root using the negative sqrt value
 * Returns:     TRUE / FALSE
 * Users:       general
 * Calls:	sqrt
 */
static int
quadRoots(double a, double b, double c, double *rp, double *rn)
{
  double d;			/* Discriminant */
  double f;			/* Root factor */

  if (a == 0.0) return(FALSE);

  if (b == 0.0) {
    d = -c/a;
    if (d >= 0.0) {
      *rn = - (*rp = sqrt(d));
    }
    else {
      return(FALSE);
    }
  }
  else {
    d = 1.0 - 4*a*c/(b*b);
    if (d >= 0.0) {
      f = (1.0 + sqrt(d))/2.0;
      *rp = - (b*f)/a;
      *rn = - c/(b*f);
    }
    else {
      return(FALSE);
    }
  }
  return(TRUE);
}


/*
 * Name:	twoSideSpacing
 * Purpose:	Find a compatible set of ratios and node numbers when the
 *		spacing is constrained at both ends of a span.
 * Formals:	< I > width: size the span
 *		< I > hStart: spacing at start of span
 *		< I > hEnd: spacing at end of span
 *		< I > rWanted: desired ratio of spacings
 *		< O > rSfound: ratio found for start of span
 *		< O > rEfound: ratio found for end of span
 *		< O > nSfound: number of start spaces
 *		< O > nEfound: number of end spaces
 * Returns:	OK / E_PRIVATE
 * Users:	MESHspacing
 * Calls:	twoSideRatio, error-message handler
 */
static int
twoSideSpacing(double width, double hStart, double hEnd, double rWanted, 
               double *rSfound, double *rEfound, int *nSfound, int *nEfound)
{
  int nSpaceS;			/* Number of spaces at the start */
  int nSpaceE;			/* Number of spaces at the end */
  int nSpaceT;			/* Number of spaces total */
  double dSpaceS;		/* Exact value of nSpaceS */
  double dSpaceE;		/* Exact value of nSpaceE */
  double dSpaceT;		/* Exact value of nSpaceT */
  double dDiff;			/* Difference between dSpaceS & dSpaceE */
  double remaining;		/* Length of span between hs and he */
  double rTempS = 0.0, rTempE = 0.0;	/* For temporarily calc'ed ratios */
  double hsLast, heLast;	/* Used to ensure ratio is valid */
  double rConnect;		/* " */
  double hMax, hMin;		/* Max and min between hStart and hEnd */
  double tmp;
  int i;			/* Indices for searching for best ratio */
  int solnFound;		/* For partial search termination */
  int solnError;		/* For partial search termination */
  int nSaveS = 0;		/* Saves best solution so far */
  int nSaveE = 0;		/* " */
  double rSaveS = 0.0;		/* " */
  double rSaveE = 0.0;		/* " */

/*
 * It's an error if there isn't enough width to fit in both spaces.
 */
  remaining = width - (hStart + hEnd);
  if (remaining < 0.0) {
    SPfrontEnd->IFerrorf( ERR_WARNING, "two-sided spacing can't find an acceptable solution\n");
    *rSfound = *rEfound = 0.0;
    *nSfound = *nEfound = 0;
    return(E_PRIVATE);
  }

/* Adjust ratio wanted to acceptable limits, and find number of extra spaces
 * needed to bring the smaller ratio up to the size of the bigger one.
 */
  hMax = MAX( hStart, hEnd );
  hMin = MIN( hStart, hEnd );

  if ( hMax == hMin ) {
    dDiff = 0.0;
  }
  else {
/* Does a solution exist if we allow the number of spaces to take on 
 * a non-integral value?
 * If not, then adjust the ratio to lie within acceptable bounds.
 * Since the choice of whether or not to require a peak in the plot
 * of "spacing vs number" is arbitrary, both cases are checked, and
 * the one that gives the closest answer to the original ratio
 * is chosen.  The function quadRoots is used to find limits for the
 * ratio in the peaked case.  The unpeaked case can find a lower
 * bound more easily.
 */
    if (quadRoots( hMax, hMax - width, remaining, &rTempS, &rTempE )) {
      rWanted = MIN(rWanted, rTempS);
      rTempS = 1.0 + (hMax - hMin)/(width - hMax);
      rWanted = MAX(rWanted, rTempS);
      if ((rWanted != rTempS) && (rTempE > rWanted)) {
	if (ABS(rWanted - rTempE) < 4.0*ABS(rWanted - rTempS)) {
	  rWanted = rTempE;
	}
	else {
	  rWanted = rTempS;
	}
      }
    }
    else { /* Complex roots */
      rTempS = 1.0 + (hMax - hMin)/(width - hMax);
      rWanted = MAX(rWanted, rTempS);
    }
    dDiff = log(hMax/hMin)/log(rWanted);
    dDiff *= ( hStart < hEnd ) ? -1.0 : 1.0;
  }

/* Find the number of spaces at the start and at the end. */
/* Handle ratio near 1.0 carefully. */
  if ( ABS(rWanted - 1.0) < 1.0e-4 ) {
    dSpaceS = (width - dDiff*hEnd)/(hStart+hEnd);
  }
  else {
    tmp = (hStart+hEnd-width+width*rWanted)/
      (hStart+hEnd*pow(rWanted,dDiff));
    dSpaceS = log(tmp)/log(rWanted);
  }
  dSpaceE = dSpaceS + dDiff;
  dSpaceT = dSpaceS + dSpaceE;

/* Search until an acceptable solution is found.  Some
 * cases may be repeated, but no harm is done.
 */
  for (i = 0; i <= 1; i++) {
    nSpaceT = (int)dSpaceT + i;
/* Guess a starting point which is guaranteed to have a solution. */
    nSpaceS = MIN( nSpaceT - 1, MAX( 4, (int) dSpaceS) );
    nSpaceE = nSpaceT - nSpaceS;

    solnFound = solnError = FALSE;
    while ( !solnFound ) { 

/* Take care of special cases first. */
      if ((nSpaceE <= 0) || (nSpaceS <= 0)) {
	solnError = TRUE;
      }
      else if (nSpaceT == 2) { 	/* Check for exact fit */
	if (ABS(remaining) < 1.0e-3*hMax ) {
	  rTempS = hEnd / hStart;
	  rTempE = 1.0 / rTempS;
	  nSpaceS = nSpaceE = 1;
	}
	else {
	  solnError = TRUE;
	}
      }
      else if (nSpaceT == 3) { 	/* Trivial to solve */
	if (remaining > 0.0) {
	  rTempS = remaining / hStart;
	  rTempE = remaining / hEnd;
	  nSpaceS = 2;		/* Always put middle space at start */
	  nSpaceE = 1;
	}
	else {
	  solnError = TRUE;
	}
      }
      else { /* Finally, the general case */
	if (remaining > 0.0) {
	  rTempS = rWanted;
	  twoSideRatio( width, hStart, hEnd, &rTempS, nSpaceS, nSpaceE );
	  rTempE = rTempS;
	}
	else {
	  solnError = TRUE;
	}
      }

      if ( solnError )
	break;		/* while loop */

/* Check whether the ratio discovered is good or not. */
      hsLast = hStart*pow(rTempS, (double)nSpaceS-1.0);
      heLast = hEnd*pow(rTempE, (double)nSpaceE-1.0);
      rConnect = heLast/hsLast;
      if ( rConnect < 1.0/rTempE - RAT_TOL ) {
	nSpaceS--;
	nSpaceE++;
      }
      else if ( rConnect > rTempS + RAT_TOL ) {
	nSpaceS++;
	nSpaceE--;
      }
      else {
	solnFound = TRUE;
/* Save if this solution is better than the previous one. */
	if (ABS(rWanted - rTempS) <= ABS(rWanted - rSaveS)) {
	  rSaveS = rTempS;
	  rSaveE = rTempE;
	  nSaveS = nSpaceS;
	  nSaveE = nSpaceE;
	}
      }
    }
  }

/* Prepare return values. */
  if (rSaveS == 0.0) {
    SPfrontEnd->IFerrorf( ERR_WARNING, "two-sided spacing can't find an acceptable solution\n");
    *rSfound = *rEfound = 0.0;
    *nSfound = *nEfound = 0;
    return(E_PRIVATE);
  }
  else {
    *rSfound = rSaveS;
    *rEfound = rSaveE;
    *nSfound = nSaveS;
    *nEfound = nSaveE;
    return(OK);
  }
}



/*
 * Name:	twoSideRatio
 * Purpose:	Finds the unique ratio 'r' which satisfies the
 *		constraint:
 *			w = hs*(1-r^ns)/(1-r) + he*(1-r^ne)/(1-r)
 * Formals:	< I > w:  size of a span
 *		< I > hs: spacing at start of span
 *		< I > he: spacing at end of span
 *		<I/O> argRatio: returns r, contains initial guess for r
 *		< I > ns: number of steps to take at start
 *		< I > ne: number of steps to take at end
 * Returns:	OK / E_PRIVATE
 * Users:	twoSideSpacing
 * Calls:	error-message handler
 */
static int
twoSideRatio(double w, double hs, double he, double *argRatio, int ns, 
             int ne)
{
  double funcLow, funcUpp, func;
  double ratLow, ratUpp, ratio = *argRatio;
  double dns = (double)ns;
  double dne = (double)ne;
  int i;

/* Get lower bound on solution. */
  ratLow = 0.0;
  funcLow = hs + he - w;
  if ((funcLow > 0.0) || ((funcLow < 0.0) && (MAX(ns,ne) <= 1))) {
    *argRatio = 0.0;
    return(E_PRIVATE);
  }

/* Find upper bound on solution. */
  ratUpp = ratio;
  do {
    ratUpp += 0.2;
    funcUpp = hs*geomSum(ratUpp, dns) + he*geomSum(ratUpp, dne) - w;
  } while (funcUpp < 0.0);

/* Do bisections to find new ratio. */
  for ( i=0; i < RAT_LIM; i++ ) {
    ratio = ratLow + 0.5 * (ratUpp - ratLow);
    func = hs*geomSum(ratio, dns) + he*geomSum(ratio, dne) - w;
    if ((func == 0.0) || (ratUpp - ratLow < RAT_TOL)) break;

    funcLow = hs*geomSum(ratLow, dns) + he*geomSum(ratLow, dne) - w;
    if (funcLow*func > 0.0) {
      ratLow = ratio;
    }
    else {
      ratUpp = ratio;
    }
  }

  if (i == RAT_LIM) { /* No solution found */
    *argRatio = 0.0;
    return(E_PRIVATE);
  }
  else {
    *argRatio = ratio;
    return(OK);
  }
}



/*
 * Name:	maxLimSpacing
 * Purpose:	Find compatible number of spaces and ratio when the spacing
 *		is constrained at start of a span, and has to be smaller
 *		than a user-specified maximum at the end.
 * Formals:	< I > width: width of the span
 *		< I > hStart: spacing constraint at one end
 *		< I > hMax: maximum spacing allowable
 *		< I > rWanted: ideal ratio of one spacing to the next
 *		< O > rFound: actual ratio discovered
 *		< O > nSfound: number of start spaces
 *		< O > nMfound: number of maximum spaces
 * Returns:	OK / E_PRIVATE
 * Users:	MESHspacing
 * Calls:	oneSideRatio, stepsInSpan
 */
static int
maxLimSpacing(double width, double hStart, double hMax, double rWanted, 
              double *rFound, int *nSfound, int *nMfound)
{
  int nSpaceS;				/* Number of spaces at the start */
  int nSpaceM;				/* Number of spaces with maximum size */
  int nSpaceT;				/* Total number of spaces */
  double dSpaceS;			/* Exact number of start spaces needed */
  double dSpaceM;			/* Exact number of max spaces needed */
  double dSpaceT;			/* Exact total number of spaces */
  double rTempS;			/* For temporarily calc'ed ratio */
  double remaining;			/* Width taken up by start spaces */
  double rConnect, hBiggest = 0.0;	/* Used to ensure ratio is valid */
  double rSaveS = 0.0;			/* Saves best solution so far */
  int nSaveS = 0;			/* " */
  int nSaveM = 0;			/* " */
  int i;				/* Searching indices */
  int solnFound;			/* For partial search termination */
  int solnError;			/* For partial search termination */

/* Compute the ratio needed to exactly go from hStart to hMax
 * in the given width.  If hMax is really big, then we know
 * the spaces can't exceed it.
 */
  if ( width > hMax ) {
    rTempS = 1.0 + (hMax - hStart)/(width - hMax);
  }
  else {
    rTempS = 1.0e6;	 	/* an impossibly large value */
  }

  if (rWanted <= rTempS) { /* Spacings stay below maximum allowed */
    dSpaceS = stepsInSpan( width, hStart, rWanted );
    dSpaceM = 0.0;
  }
  else {
/* Find number of spaces needed to increase hStart to hMax. */
    dSpaceS = log(hMax/hStart)/log(rWanted);
    remaining = hStart*geomSum(rWanted, dSpaceS);
    dSpaceM = (width - remaining)/hMax;
  }
  dSpaceT = dSpaceS + dSpaceM;

/* Search until an acceptable solution is found.  Some
 * cases may be repeated, but no harm is done.
 */
  for (i = 0; i <= 1; i++) {
    nSpaceT = (int)dSpaceT + i;
/* Guess a starting point which is guaranteed to have a solution. */
    nSpaceS = MIN( nSpaceT, MAX( 3, (int) dSpaceS) );
    nSpaceM = nSpaceT - nSpaceS;

    solnFound = solnError = FALSE;
    while ( !solnFound ) { 
      remaining = width - hMax*nSpaceM;

/* Test for the various special cases first. */
      if ( nSpaceM < 0 || nSpaceS <= 0 ) {
        solnError = TRUE;
      }
      else if (nSpaceS == 1) { /* check for exact fit */
	if ( ABS(remaining - hStart) < 1.0e-3*hStart ) {
	  hBiggest = hStart;
          if (nSpaceM == 0) {
            rTempS = 1.0;
          }
          else {
            rTempS = hMax / hStart;
          }
	}
	else {
          solnError = TRUE;
	}
      }
      else if (nSpaceS == 2) {	/* Easy to solve */
	if (remaining > hStart) {
	  hBiggest = remaining - hStart;
	  rTempS = hBiggest / hStart;
	}
	else {
          solnError = TRUE;
	}
      }
      else {
	if (remaining > hStart) {
	  rTempS = rWanted;
	  oneSideRatio( remaining, hStart, &rTempS, nSpaceS );
	  hBiggest = hStart*pow(rTempS, (double)nSpaceS - 1.0);
	}
	else {
          solnError = TRUE;
	}
      }

      if ( solnError )
	break;		/* while loop */

      rConnect = hMax / hBiggest;
      if ( rConnect < 1.0 - RAT_TOL ) {
	nSpaceS--;
	nSpaceM++;
      }
      else if ( (rConnect > rTempS + RAT_TOL) && nSpaceM != 0 ) {
	nSpaceS++;
	nSpaceM--;
      }
      else {
	solnFound = TRUE;
/* Save solution if it's better. */
        if ( (rTempS >= 1.0 - RAT_TOL)
	    && ABS(rWanted - rTempS) <= ABS(rWanted - rSaveS)) {
	  rSaveS = rTempS;
	  nSaveS = nSpaceS;
	  nSaveM = nSpaceM;
	}
      }
    }
  }

/* Prepare return values. */
  if (rSaveS == 0.0) {
    SPfrontEnd->IFerrorf( ERR_WARNING, "max-limited spacing can't find an acceptable solution\n");
    *rFound = 0.0;
    *nSfound = *nMfound = 0;
    return(E_PRIVATE);
  }
  else {
    *rFound = rSaveS;
    *nSfound = nSaveS;
    *nMfound = nSaveM;
    return(OK);
  }
}
