/* $Header$ */
/*
 * draw.c: xgraph drawing code
 *
 * Routines:
 *	void DrawWindow();
 *
 * $Log$
 * Revision 1.1  2004-01-25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.1.1.1  1999/12/03 23:15:53  heideman
 * xgraph-12.0
 *
 */
#ifndef lint
static char rcsid[] = "$Id$";
#endif

#include "copyright.h"
#include <stdio.h>
#include <math.h>
#include <pwd.h>
#include <ctype.h>
#include "xgraph.h"
#include "xtb.h"
#include "hard_devices.h"
#include "params.h"


static void DrawTitle();
static void DrawGridAndAxis();
static void WriteValue();
static void DrawData();
static void DrawLegend();
static int TransformCompute();
static double initGrid();
static double stepGrid();
static double RoundUp();
static void set_mark_flags();

void
DrawWindow(win_info)
LocalWin *win_info;		/* Window information */

/*
 * Draws the data in the window.  Does not clear the window.
 * The data is scaled so that all of the data will fit.
 * Grid lines are drawn at the nearest power of 10 in engineering
 * notation.  Draws axis numbers along bottom and left hand edges.
 * Centers title at top of window.
 */
{
    /* Figure out the transformation constants */
    if (TransformCompute(win_info)) {

	/* Draw the title */
	DrawTitle(win_info);

	/* Draw the legend */
	if (!PM_BOOL("NoLegend"))
	    DrawLegend(win_info);

	/* Draw the axis unit labels,  grid lines,  and grid labels */
	DrawGridAndAxis(win_info);

	/* Draw the data sets themselves */
	DrawData(win_info);
    }
}




static void
DrawTitle(wi)
LocalWin *wi;			/* Window information    */

/*
 * This routine draws the title of the graph centered in
 * the window.  It is spaced down from the top by an amount
 * specified by the constant PADDING.  The font must be
 * fixed width.  The routine returns the height of the
 * title in pixels.
 */
{
    if (wi->DOrder == 0) 
      wi->dev_info.xg_text(wi->dev_info.user_state,
			 (int)(wi->dev_info.area_w*0.95) ,
			 wi->dev_info.axis_pad,
			 PM_STR("TitleText"), T_UPPERRIGHT, T_TITLE);
    else if (wi->DOrder == 1)
      wi->dev_info.xg_text(wi->dev_info.user_state,
			 (int)(wi->dev_info.area_w*0.95) ,
			 wi->dev_info.axis_pad,
			 "First Derivative", T_UPPERRIGHT, T_TITLE);
    else if (wi->DOrder == 2)
      wi->dev_info.xg_text(wi->dev_info.user_state,
			 (int)(wi->dev_info.area_w*0.95) ,
			 wi->dev_info.axis_pad,
			 "Second Derivative", T_UPPERRIGHT, T_TITLE);
}




static int
TransformCompute(wi)
LocalWin *wi;			/* Window information          */

/*
 * This routine figures out how to draw the axis labels and grid lines.
 * Both linear and logarithmic axes are supported.  Axis labels are
 * drawn in engineering notation.  The power of the axes are labeled
 * in the normal axis labeling spots.  The routine also figures
 * out the necessary transformation information for the display
 * of the points (it touches XOrgX, XOrgY, UsrOrgX, UsrOrgY, and
 * UnitsPerPixel).
 */
{
    double  bbCenX,
            bbCenY,
            bbHalfWidth,
            bbHalfHeight;
    int     idx,
            maxName,
            leftWidth;
    char    err[MAXBUFSIZE];
    char   *XUnitText = PM_STR("XUnitText");

    /*
     * First,  we figure out the origin in the X window.  Above the space we
     * have the title and the Y axis unit label. To the left of the space we
     * have the Y axis grid labels.
     */

    wi->XOrgX = wi->dev_info.bdr_pad + (7 * wi->dev_info.axis_width)
	+ wi->dev_info.bdr_pad;
    wi->XOrgY = wi->dev_info.bdr_pad + wi->dev_info.title_height
	+ wi->dev_info.bdr_pad + wi->dev_info.axis_height
	+ wi->dev_info.axis_height / 2 + wi->dev_info.bdr_pad;

    /*
     * Now we find the lower right corner.  Below the space we have the X axis
     * grid labels.  To the right of the space we have the X axis unit label
     * and the legend.  We assume the worst case size for the unit label.
     */

    maxName = 0;
    for (idx = 0; idx < MAXSETS; idx++) {
	if (PlotData[idx].list) {
	    int     tempSize;

	    tempSize = strlen(PlotData[idx].setName);
	    if (tempSize > maxName)
		maxName = tempSize;
	}
    }
    if (PM_BOOL("NoLegend"))
	maxName = 0;
    /* Worst case size of the X axis label: */
    leftWidth = (strlen(XUnitText)) * wi->dev_info.axis_width;
    if ((maxName * wi->dev_info.axis_width) + wi->dev_info.bdr_pad > leftWidth)
	leftWidth = maxName * wi->dev_info.axis_width + wi->dev_info.bdr_pad;

    wi->XOppX = wi->dev_info.area_w - wi->dev_info.bdr_pad - leftWidth;
    wi->XOppY = wi->dev_info.area_h - wi->dev_info.bdr_pad
	- wi->dev_info.axis_height - wi->dev_info.bdr_pad;

    if ((wi->XOrgX >= wi->XOppX) || (wi->XOrgY >= wi->XOppY)) {
	do_error(strcpy(err, "Drawing area is too small\n"));
	return 0;
    }

    /*
     * We now have a bounding box for the drawing region. Figure out the units
     * per pixel using the data set bounding box.
     */
    wi->XUnitsPerPixel = (wi->hiX - wi->loX) / 
			 ((double) (wi->XOppX - wi->XOrgX));
    wi->YUnitsPerPixel = (wi->hiY - wi->loY) / 
			 ((double) (wi->XOppY - wi->XOrgY));

    /*
     * Find origin in user coordinate space.  We keep the center of the
     * original bounding box in the same place.
     */
    bbCenX = (wi->loX + wi->hiX) / 2.0;
    bbCenY = (wi->loY + wi->hiY) / 2.0;
    bbHalfWidth = ((double) (wi->XOppX - wi->XOrgX)) / 2.0 * wi->XUnitsPerPixel;
    bbHalfHeight = ((double) (wi->XOppY - wi->XOrgY)) / 2.0 * wi->YUnitsPerPixel;
    wi->UsrOrgX = bbCenX - bbHalfWidth;
    wi->UsrOrgY = bbCenY - bbHalfHeight;
    wi->UsrOppX = bbCenX + bbHalfWidth;
    wi->UsrOppY = bbCenY + bbHalfHeight;

    /*
     * Everything is defined so we can now use the SCREENX and SCREENY
     * transformations.
     */
    return 1;
}

static void
DrawGridAndAxis(wi)
LocalWin *wi;			/* Window information         */

/*
 * This routine draws grid line labels in engineering notation,
 * the grid lines themselves,  and unit labels on the axes.
 */
{
    int     expX,
            expY;		/* Engineering powers */
    int     startX;
    int     Yspot,
            Xspot;
    char    power[10],
            value[10],
            final[MAXBUFSIZE + 10];
    double  Xincr,
            Yincr,
            Xstart,
            Ystart,
            Yindex,
            Xindex,
            larger;
    XSegment segs[2];
    double  initGrid(),
            stepGrid();
    int     tickFlag = PM_BOOL("Ticks");
    int	    axisFlag = PM_BOOL("TickAxis");
    int     logXFlag = PM_BOOL("LogX");
    int     logYFlag = PM_BOOL("LogY");
    char   *XUnitText = PM_STR("XUnitText");
    char   *YUnitText = PM_STR("YUnitText");

    /*
     * Grid display powers are computed by taking the log of the largest
     * numbers and rounding down to the nearest multiple of 3.
     */
    if (logXFlag) {
	expX = 0;
    }
    else {
	if (fabs(wi->UsrOrgX) > fabs(wi->UsrOppX)) {
	    larger = fabs(wi->UsrOrgX);
	}
	else {
	    larger = fabs(wi->UsrOppX);
	}
	expX = ((int) floor(nlog10(larger) / 3.0)) * 3;
    }
    if (logYFlag) {
	expY = 0;
    }
    else {
	if (fabs(wi->UsrOrgY) > fabs(wi->UsrOppY)) {
	    larger = fabs(wi->UsrOrgY);
	}
	else {
	    larger = fabs(wi->UsrOppY);
	}
	expY = ((int) floor(nlog10(larger) / 3.0)) * 3;
    }

    /*
     * With the powers computed,  we can draw the axis labels.
     */
    if (expY != 0) {
	(void) strcpy(final, YUnitText);
	(void) strcat(final, " x 10");
	Xspot = wi->dev_info.bdr_pad +
	    ((strlen(YUnitText) + 5) * wi->dev_info.axis_width);
	Yspot = wi->dev_info.bdr_pad * 2 + wi->dev_info.title_height +
	    wi->dev_info.axis_height / 2;
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     Xspot, Yspot, final, T_RIGHT, T_AXIS);
	(void) sprintf(power, "%d", expY);
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     Xspot, Yspot, power, T_LOWERLEFT, T_AXIS);
    }
    else {
	Yspot = wi->dev_info.bdr_pad * 2 + wi->dev_info.title_height;
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     wi->dev_info.bdr_pad, Yspot, YUnitText,
			     T_UPPERLEFT, T_AXIS);
    }

    startX = wi->dev_info.area_w - wi->dev_info.bdr_pad;
    if (expX != 0) {
	(void) sprintf(power, "%d", expX);
	startX -= (strlen(power) * wi->dev_info.axis_width);
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     startX, wi->XOppY, power, T_LOWERLEFT, T_AXIS);
	(void) strcpy(final, XUnitText);
	(void) strcat(final, " x 10");
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     startX, wi->XOppY, final, T_RIGHT, T_AXIS);
    }
    else {
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     startX, wi->XOppY, XUnitText, T_RIGHT, T_AXIS);
    }

    /*
     * First,  the grid line labels
     */
    Yincr = (wi->dev_info.axis_pad + wi->dev_info.axis_height) *
	    wi->YUnitsPerPixel;
    Ystart = initGrid(wi->UsrOrgY, Yincr, logYFlag);
    for (Yindex = Ystart; Yindex < wi->UsrOppY; Yindex = stepGrid()) {
	Yspot = SCREENY(wi, Yindex);
	/* Write the axis label */
	WriteValue(value, PM_STR("Format X"), Yindex, expY, logYFlag);
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     wi->dev_info.bdr_pad +
			     (7 * wi->dev_info.axis_width),
			     Yspot, value, T_RIGHT, T_AXIS);
    }

    Xincr = (wi->dev_info.axis_pad + (wi->dev_info.axis_width * 7)) * 
	    wi->XUnitsPerPixel;
    Xstart = initGrid(wi->UsrOrgX, Xincr, logXFlag);
	
    for (Xindex = Xstart; Xindex < wi->UsrOppX; Xindex = stepGrid()) {
	Xspot = SCREENX(wi, Xindex);
	/* Write the axis label */
	WriteValue(value, PM_STR("Format Y"), Xindex, expX, logXFlag);
	wi->dev_info.xg_text(wi->dev_info.user_state,
			     Xspot,
			     wi->dev_info.area_h - wi->dev_info.bdr_pad,
			     value, T_BOTTOM, T_AXIS);
    }

    /*
     * Now,  the grid lines or tick marks
     */
    Yincr = (wi->dev_info.axis_pad + wi->dev_info.axis_height) * 
	    wi->YUnitsPerPixel;
    Ystart = initGrid(wi->UsrOrgY, Yincr, logYFlag);
    for (Yindex = Ystart; Yindex < wi->UsrOppY; Yindex = stepGrid()) {
	Yspot = SCREENY(wi, Yindex);
	/* Draw the grid line or tick mark */
	if (tickFlag && !(axisFlag && Yindex == Ystart)) {
	    segs[0].x1 = wi->XOrgX;
	    segs[0].x2 = wi->XOrgX + wi->dev_info.tick_len;
	    segs[1].x1 = wi->XOppX - wi->dev_info.tick_len;
	    segs[1].x2 = wi->XOppX;
	    segs[0].y1 = segs[0].y2 = segs[1].y1 = segs[1].y2 = Yspot;
	}
	else {
	    segs[0].x1 = wi->XOrgX;
	    segs[0].x2 = wi->XOppX;
	    segs[0].y1 = segs[0].y2 = Yspot;
	}
	if ((ABS(Yindex) < ZERO_THRES) && !logYFlag) {
	    wi->dev_info.xg_seg(wi->dev_info.user_state,
				1, segs, PM_INT("ZeroWidth"),
				L_ZERO, 0, 0);
	    if (tickFlag) {
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &(segs[1]), PM_INT("ZeroWidth"),
				    L_ZERO, 0, 0);
	    }
	}
	else {
	    wi->dev_info.xg_seg(wi->dev_info.user_state,
				1, segs, PM_INT("GridSize"),
				L_AXIS, 0, 0);
	    if (tickFlag) {
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &(segs[1]), PM_INT("GridSize"),
				    L_AXIS, 0, 0);
	    }
	}
    }

    Xincr = (wi->dev_info.axis_pad + (wi->dev_info.axis_width * 7)) *
	wi->XUnitsPerPixel;
    Xstart = initGrid(wi->UsrOrgX, Xincr, logXFlag);
    for (Xindex = Xstart; Xindex < wi->UsrOppX; Xindex = stepGrid()) {
	Xspot = SCREENX(wi, Xindex);
	/* Draw the grid line or tick marks */
	if (tickFlag && !(axisFlag && Xindex == Xstart)) {
	    segs[0].x1 = segs[0].x2 = segs[1].x1 = segs[1].x2 = Xspot;
	    segs[0].y1 = wi->XOrgY;
	    segs[0].y2 = wi->XOrgY + wi->dev_info.tick_len;
	    segs[1].y1 = wi->XOppY - wi->dev_info.tick_len;
	    segs[1].y2 = wi->XOppY;
	}
	else {
	    segs[0].x1 = segs[0].x2 = Xspot;
	    segs[0].y1 = wi->XOrgY;
	    segs[0].y2 = wi->XOppY;
	}
	if ((ABS(Xindex) < ZERO_THRES) && !logXFlag) {
	    wi->dev_info.xg_seg(wi->dev_info.user_state,
				1, segs, PM_INT("ZeroWidth"), L_ZERO, 0, 0);
	    if (tickFlag) {
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &(segs[1]), PM_INT("ZeroWidth"),
				    L_ZERO, 0, 0);

	    }
	}
	else {
	    wi->dev_info.xg_seg(wi->dev_info.user_state,
				1, segs, PM_INT("GridSize"), L_AXIS, 0, 0);
	    if (tickFlag) {
		wi->dev_info.xg_seg(wi->dev_info.user_state,
			     1, &(segs[1]), PM_INT("GridSize"), L_AXIS, 0, 0);
	    }
	}
    }
    /* Check to see if he wants a bounding box */
    if (PM_BOOL("BoundBox")) {
	XSegment bb[4];

	/* Draw bounding box */
	bb[0].x1 = bb[0].x2 = bb[1].x1 = bb[3].x2 = wi->XOrgX;
	bb[0].y1 = bb[2].y2 = bb[3].y1 = bb[3].y2 = wi->XOrgY;
	bb[1].x2 = bb[2].x1 = bb[2].x2 = bb[3].x1 = wi->XOppX;
	bb[0].y2 = bb[1].y1 = bb[1].y2 = bb[2].y1 = wi->XOppY;
	wi->dev_info.xg_seg(wi->dev_info.user_state,
			    4, bb, PM_INT("GridSize"), L_AXIS, 0, 0);
    }
}

static double gridBase,
        gridStep,
        gridJuke[101];
static int gridNJuke,
        gridCurJuke;

#define ADD_GRID(val)	(gridJuke[gridNJuke++] = log10(val))

static double
initGrid(low, step, logFlag)
double  low;			/* desired low value          */
double  step;			/* desired step (user coords) */
int     logFlag;		/* is axis logarithmic?       */
{
    double  ratio,
            x;
    double  RoundUp(),
            stepGrid();

    gridNJuke = gridCurJuke = 0;
    gridJuke[gridNJuke++] = 0.0;

    if (logFlag) {
	ratio = pow(10.0, step);
	gridBase = floor(low);
	gridStep = ceil(step);
	if (ratio <= 3.0) {
	    if (ratio > 2.0) {
		ADD_GRID(3.0);
	    }
	    else if (ratio > 1.333) {
		ADD_GRID(2.0);
		ADD_GRID(5.0);
	    }
	    else if (ratio > 1.25) {
		ADD_GRID(1.5);
		ADD_GRID(2.0);
		ADD_GRID(3.0);
		ADD_GRID(5.0);
		ADD_GRID(7.0);
	    }
	    else {
		for (x = 1.0; x < 10.0 && (x + .5) / (x + .4) >= ratio; x += .5) {
		    ADD_GRID(x + .1);
		    ADD_GRID(x + .2);
		    ADD_GRID(x + .3);
		    ADD_GRID(x + .4);
		    ADD_GRID(x + .5);
		}
		if (floor(x) != x)
		    ADD_GRID(x += .5);
		for (; x < 10.0 && (x + 1.0) / (x + .5) >= ratio; x += 1.0) {
		    ADD_GRID(x + .5);
		    ADD_GRID(x + 1.0);
		}
		for (; x < 10.0 && (x + 1.0) / x >= ratio; x += 1.0) {
		    ADD_GRID(x + 1.0);
		}
		if (x == 7.0) {
		    gridNJuke--;
		    x = 6.0;
		}
		if (x < 7.0) {
		    ADD_GRID(x + 2.0);
		}
		if (x == 10.0)
		    gridNJuke--;
	    }
	    x = low - gridBase;
	    for (gridCurJuke = -1; x >= gridJuke[gridCurJuke + 1]; gridCurJuke++) {
	    }
	}
    }
    else {
	gridStep = RoundUp(step);
	gridBase = floor(low / gridStep) * gridStep;
    }
    return (stepGrid());
}

static double
stepGrid()
{
    if (++gridCurJuke >= gridNJuke) {
	gridCurJuke = 0;
	gridBase += gridStep;
    }
    return (gridBase + gridJuke[gridCurJuke]);
}

static double
RoundUp(val)
double  val;			/* Value */

/*
 * This routine rounds up the given positive number such that
 * it is some power of ten times either 1, 2, or 5.  It is
 * used to find increments for grid lines.
 */
{
    int     exponent,
            idx;

    exponent = (int) floor(nlog10(val));
    if (exponent < 0) {
	for (idx = exponent; idx < 0; idx++) {
	    val *= 10.0;
	}
    }
    else {
	for (idx = 0; idx < exponent; idx++) {
	    val /= 10.0;
	}
    }
    if (val > 5.0)
	val = 10.0;
    else if (val > 2.0)
	val = 5.0;
    else if (val > 1.0)
	val = 2.0;
    else
	val = 1.0;
    if (exponent < 0) {
	for (idx = exponent; idx < 0; idx++) {
	    val /= 10.0;
	}
    }
    else {
	for (idx = 0; idx < exponent; idx++) {
	    val *= 10.0;
	}
    }
    return val;
}

static void
WriteValue(str, fmt, val, expv, logFlag)
char   *str;			/* String to write into */
char   *fmt;			/* Format to print str	 */
double  val;			/* Value to print       */
int     expv;			/* Exponent             */
int     logFlag;		/* Is this a log axis?  */

/*
 * Writes the value provided into the string in a fixed format
 * consisting of seven characters.  The format is:
 *   -ddd.dd
 */
{
    int     idx;

    if (logFlag) {
	if (val == floor(val)) {
	    if (strcmp(fmt, "%.2f") == 0)
		fmt = "%.0e";
	    val = pow(10.0, val);
	}
	else {
	    if (strcmp(fmt, "%.2f") == 0)
		fmt = "%.2g";
	    val = pow(10.0, val - floor(val));
	}
    }
    else {
	if (expv < 0) {
	    for (idx = expv; idx < 0; idx++) {
		val *= 10.0;
	    }
	}
	else {
	    for (idx = 0; idx < expv; idx++) {
		val /= 10.0;
	    }
	}
    }
    if (strchr(fmt, 'd') || strchr(fmt, 'x'))
	(void) sprintf(str, fmt, (int) val);
    else
	(void) sprintf(str, fmt, val);
}


#define LEFT_CODE	0x01
#define RIGHT_CODE	0x02
#define BOTTOM_CODE	0x04
#define TOP_CODE	0x08

/* Clipping algorithm from Neumann and Sproull by Cohen and Sutherland */
#define C_CODE(xval, yval, rtn) \
rtn = 0; \
if ((xval) < wi->UsrOrgX) rtn = LEFT_CODE; \
else if ((xval) > wi->UsrOppX) rtn = RIGHT_CODE; \
if ((yval) < wi->UsrOrgY) rtn |= BOTTOM_CODE; \
else if ((yval) > wi->UsrOppY) rtn |= TOP_CODE

void
EraseData(wi)
LocalWin *wi;

/*
 * This routine draws the data sets themselves using the macros
 * for translating coordinates.
 */
{
    double  sx1,
            sy1,
            sx2,
            sy2,
            tx = 0,
            ty = 0;
    int     idx,
            subindex;
    int     code1,
            code2,
            cd,
            mark_inside;
    int     X_idx, StoreIDX; /* PW */
    XSegment *ptr;
    PointList *thisList,
              *lastList;
    int     markFlag,
            pixelMarks,
            bigPixel,
            colorMark;
    int     noLines = PM_BOOL("NoLines");
    int     lineWidth = PM_INT("LineWidth");

    /* PW Suggests we Flush and set first */
    set_mark_flags(&markFlag, &pixelMarks, &bigPixel, &colorMark);
    for (idx = 0; idx < MAXSETS; idx++) {
        if (wi->DOrder == 0)
	  thisList = PlotData[idx].list;
        else if (wi->DOrder == 1)
	    thisList = DataD1[idx].list;
        else if (wi->DOrder == 2)
	    thisList = DataD2[idx].list;
        else {
          printf ("Internal Error differentiating - order > 2!\n");
          exit (1);
        }
	while (thisList) {
	    X_idx = 0;
	    for (subindex = 0; subindex < thisList->numPoints - 1; subindex++) {
		/* Put segment in (sx1,sy1) (sx2,sy2) */
		sx1 = thisList->xvec[subindex];
		sy1 = thisList->yvec[subindex];
		sx2 = thisList->xvec[subindex + 1];
		sy2 = thisList->yvec[subindex + 1];
		/* Now clip to current window boundary */
		C_CODE(sx1, sy1, code1);
		C_CODE(sx2, sy2, code2);
		mark_inside = (code1 == 0);
		while (code1 || code2) {
		    if (code1 & code2)
			break;
		    cd = (code1 ? code1 : code2);
		    if (cd & LEFT_CODE) {	/* Crosses left edge */
			ty = sy1 + (sy2 - sy1) * (wi->UsrOrgX - sx1) / 
						 (sx2 - sx1);
			tx = wi->UsrOrgX;
		    }
		    else if (cd & RIGHT_CODE) {	/* Crosses right edge */
			ty = sy1 + (sy2 - sy1) * (wi->UsrOppX - sx1) / 
						 (sx2 - sx1);
			tx = wi->UsrOppX;
		    }
		    else if (cd & BOTTOM_CODE) {/* Crosses bottom edge */
			tx = sx1 + (sx2 - sx1) * (wi->UsrOrgY - sy1) / 
						 (sy2 - sy1);
			ty = wi->UsrOrgY;
		    }
		    else if (cd & TOP_CODE) {	/* Crosses top edge */
			tx = sx1 + (sx2 - sx1) * (wi->UsrOppY - sy1) / 
						 (sy2 - sy1);
			ty = wi->UsrOppY;
		    }
		    if (cd == code1) {
			sx1 = tx;
			sy1 = ty;
			C_CODE(sx1, sy1, code1);
		    }
		    else {
			sx2 = tx;
			sy2 = ty;
			C_CODE(sx2, sy2, code2);
		    }
		}
		if (!code1 && !code2) {
		    /* Add segment to list */
		    Xsegs[0][X_idx].x1 = Xsegs[1][X_idx].x1;
		    Xsegs[0][X_idx].y1 = Xsegs[1][X_idx].y1;
		    Xsegs[0][X_idx].x2 = Xsegs[1][X_idx].x2;
		    Xsegs[0][X_idx].y2 = Xsegs[1][X_idx].y2;
		    Xsegs[1][X_idx].x1 = SCREENX(wi, sx1);
		    Xsegs[1][X_idx].y1 = SCREENY(wi, sy1);
		    Xsegs[1][X_idx].x2 = SCREENX(wi, sx2);
		    Xsegs[1][X_idx].y2 = SCREENY(wi, sy2);
		    X_idx++;
		}

		/* Draw markers if requested and they are in drawing region */
		if (markFlag && mark_inside) {
		    if (pixelMarks) {
			if (bigPixel) {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x1,
						Xsegs[1][X_idx - 1].y1,
						P_DOT, 0, idx % MAXATTR);
			}
			else {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x1,
						Xsegs[1][X_idx - 1].y1,
						P_PIXEL, 0, PIXVALUE(idx));
			}
		    }
		    else {
			/* Distinctive markers */
			wi->dev_info.xg_dot(wi->dev_info.user_state,
					    Xsegs[1][X_idx - 1].x1,
					    Xsegs[1][X_idx - 1].y1,
					    P_MARK, MARKSTYLE(idx),
					    PIXVALUE(idx));
		    }
		}

		/* Draw bar elements if requested */
		if (PM_BOOL("BarGraph")) {
		    int     barPixels,
		            baseSpot;
		    XSegment line;

		    barPixels = (int) ((PM_DBL("BarWidth") /
					wi->XUnitsPerPixel) + 0.5);
		    if (barPixels <= 0)
			barPixels = 1;
		    baseSpot = SCREENY(wi, PM_DBL("BarBase"));
		    line.x1 = line.x2 = Xsegs[1][X_idx - 1].x1 + 
					(int) ((PM_DBL("BarOffset") * idx /
					    wi->XUnitsPerPixel) + 0.5);
		    if (PM_BOOL("StackGraph") && idx != 0)
			line.y1 = Xsegs[0][X_idx - 1].y1;
		    else
			line.y1 = baseSpot;
		    line.y2 = Xsegs[1][X_idx - 1].y1;
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					1, &line, barPixels, L_VAR,
					LINESTYLE(idx), PIXVALUE(idx));
		}
	    }
	    /* Handle last marker */
	    if (markFlag && (thisList->numPoints > 0)) {
		C_CODE(thisList->xvec[thisList->numPoints - 1],
		       thisList->yvec[thisList->numPoints - 1],
		       mark_inside);
		if (mark_inside == 0) {
		    if (pixelMarks) {
			if (bigPixel) {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x2,
						Xsegs[1][X_idx - 1].y2,
						P_DOT, 0, idx % MAXATTR);
			}
			else {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x2,
						Xsegs[1][X_idx - 1].y2,
						P_PIXEL, 0, PIXVALUE(idx));
			}
		    }
		    else {
			/* Distinctive markers */
			wi->dev_info.xg_dot(wi->dev_info.user_state,
					    Xsegs[1][X_idx - 1].x2,
					    Xsegs[1][X_idx - 1].y2,
					    P_MARK, MARKSTYLE(idx),
					    PIXVALUE(idx));
		    }
		}
	    }
	    /* Handle last bar */
	    if ((thisList->numPoints > 0) && PM_BOOL("BarGraph")) {
		int     barPixels,
		        baseSpot;
		XSegment line;

		barPixels = (int) ((PM_DBL("BarWidth") / 
				   wi->XUnitsPerPixel) + 0.5);
		if (barPixels <= 0)
		    barPixels = 1;
		baseSpot = SCREENY(wi, PM_DBL("BarBase"));
		line.x1 = line.x2 = Xsegs[1][X_idx - 1].x2 +
					(int) ((PM_DBL("BarOffset") * idx /
					    wi->XUnitsPerPixel) + 0.5);
		if (PM_BOOL("StackGraph") && idx != 0)
		    line.y1 = Xsegs[0][X_idx - 1].y2;
		else
		    line.y1 = baseSpot;
		line.y2 = Xsegs[1][X_idx - 1].y2;
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &line, barPixels, L_VAR,
				    LINESTYLE(idx), PIXVALUE(idx));
	    }

	    /* Erase segments */
	    if ((thisList->numPoints > 0) && (!noLines) && (X_idx > 0)) {
		ptr = Xsegs[1];
		while (X_idx > wi->dev_info.max_segs) {
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					wi->dev_info.max_segs, ptr,
					lineWidth, L_VAR,
					16, (int)(1));
					/*LINESTYLE(8), (int)(1));*/
		    ptr += wi->dev_info.max_segs;
		    X_idx -= wi->dev_info.max_segs;
		}
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    X_idx, ptr,
				    lineWidth, L_VAR,
				    16,(int)(1));
	    }
	    /* Next subset */
            thisList = thisList->next;
	}
    }
    XFlush (disp);
}


static void
DrawData(wi)
LocalWin *wi;

/*
 * This routine draws the data sets themselves using the macros
 * for translating coordinates.
 */
{
    double  sx1,
            sy1,
            sx2,
            sy2,
            tx = 0,
            ty = 0;
    int     idx,
            subindex;
    int     code1,
            code2,
            cd,
            mark_inside;
    int     X_idx, StoreIDX; /* PW */
    XSegment *ptr;
    PointList *thisList,
              *lastList;
    int     markFlag,
            pixelMarks,
            bigPixel,
            colorMark;
    int     noLines = PM_BOOL("NoLines");
    int     lineWidth = PM_INT("LineWidth");
    /* PW */
    int     theDelay;

    /* PW Suggests we Flush and set first */
    theDelay = PM_INT("DelayValue")*100000;
    XFlush(disp);
    if (PM_BOOL("Animate")) sleep(1);
    set_mark_flags(&markFlag, &pixelMarks, &bigPixel, &colorMark);
    for (idx = 0; idx < MAXSETS; idx++) {
        if (wi->DOrder == 0)
	  thisList = PlotData[idx].list;
        else if (wi->DOrder == 1)
	    thisList = DataD1[idx].list;
        else if (wi->DOrder == 2)
	    thisList = DataD2[idx].list;
        else {
          printf ("Internal Error differentiating - order > 2!\n");
          exit (1);
        }
	while (thisList) {
	    X_idx = 0;
	    for (subindex = 0; subindex < thisList->numPoints - 1; subindex++) {
		/* Put segment in (sx1,sy1) (sx2,sy2) */
		sx1 = thisList->xvec[subindex];
		sy1 = thisList->yvec[subindex];
		sx2 = thisList->xvec[subindex + 1];
		sy2 = thisList->yvec[subindex + 1];
		/* Now clip to current window boundary */
		C_CODE(sx1, sy1, code1);
		C_CODE(sx2, sy2, code2);
		mark_inside = (code1 == 0);
		while (code1 || code2) {
		    if (code1 & code2)
			break;
		    cd = (code1 ? code1 : code2);
		    if (cd & LEFT_CODE) {	/* Crosses left edge */
			ty = sy1 + (sy2 - sy1) * (wi->UsrOrgX - sx1) / 
						 (sx2 - sx1);
			tx = wi->UsrOrgX;
		    }
		    else if (cd & RIGHT_CODE) {	/* Crosses right edge */
			ty = sy1 + (sy2 - sy1) * (wi->UsrOppX - sx1) / 
						 (sx2 - sx1);
			tx = wi->UsrOppX;
		    }
		    else if (cd & BOTTOM_CODE) {/* Crosses bottom edge */
			tx = sx1 + (sx2 - sx1) * (wi->UsrOrgY - sy1) / 
						 (sy2 - sy1);
			ty = wi->UsrOrgY;
		    }
		    else if (cd & TOP_CODE) {	/* Crosses top edge */
			tx = sx1 + (sx2 - sx1) * (wi->UsrOppY - sy1) / 
						 (sy2 - sy1);
			ty = wi->UsrOppY;
		    }
		    if (cd == code1) {
			sx1 = tx;
			sy1 = ty;
			C_CODE(sx1, sy1, code1);
		    }
		    else {
			sx2 = tx;
			sy2 = ty;
			C_CODE(sx2, sy2, code2);
		    }
		}
		if (!code1 && !code2) {
		    /* Add segment to list */
		    Xsegs[0][X_idx].x1 = Xsegs[1][X_idx].x1;
		    Xsegs[0][X_idx].y1 = Xsegs[1][X_idx].y1;
		    Xsegs[0][X_idx].x2 = Xsegs[1][X_idx].x2;
		    Xsegs[0][X_idx].y2 = Xsegs[1][X_idx].y2;
		    Xsegs[1][X_idx].x1 = SCREENX(wi, sx1);
		    Xsegs[1][X_idx].y1 = SCREENY(wi, sy1);
		    Xsegs[1][X_idx].x2 = SCREENX(wi, sx2);
		    Xsegs[1][X_idx].y2 = SCREENY(wi, sy2);
		    X_idx++;
		}

		/* Draw markers if requested and they are in drawing region */
		if (markFlag && mark_inside) {
		    if (pixelMarks) {
			if (bigPixel) {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x1,
						Xsegs[1][X_idx - 1].y1,
						P_DOT, 0, idx % MAXATTR);
			}
			else {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x1,
						Xsegs[1][X_idx - 1].y1,
						P_PIXEL, 0, PIXVALUE(idx));
			}
		    }
		    else {
			/* Distinctive markers */
			wi->dev_info.xg_dot(wi->dev_info.user_state,
					    Xsegs[1][X_idx - 1].x1,
					    Xsegs[1][X_idx - 1].y1,
					    P_MARK, MARKSTYLE(idx),
					    PIXVALUE(idx));
		    }
		}

		/* Draw bar elements if requested */
		if (PM_BOOL("BarGraph")) {
		    int     barPixels,
		            baseSpot;
		    XSegment line;

		    barPixels = (int) ((PM_DBL("BarWidth") /
					wi->XUnitsPerPixel) + 0.5);
		    if (barPixels <= 0)
			barPixels = 1;
		    baseSpot = SCREENY(wi, PM_DBL("BarBase"));
		    line.x1 = line.x2 = Xsegs[1][X_idx - 1].x1 + 
					(int) ((PM_DBL("BarOffset") * idx /
					    wi->XUnitsPerPixel) + 0.5);
		    if (PM_BOOL("StackGraph") && idx != 0)
			line.y1 = Xsegs[0][X_idx - 1].y1;
		    else
			line.y1 = baseSpot;
		    line.y2 = Xsegs[1][X_idx - 1].y1;
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					1, &line, barPixels, L_VAR,
					LINESTYLE(idx), PIXVALUE(idx));
		}
	    }
	    /* Handle last marker */
	    if (markFlag && (thisList->numPoints > 0)) {
		C_CODE(thisList->xvec[thisList->numPoints - 1],
		       thisList->yvec[thisList->numPoints - 1],
		       mark_inside);
		if (mark_inside == 0) {
		    if (pixelMarks) {
			if (bigPixel) {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x2,
						Xsegs[1][X_idx - 1].y2,
						P_DOT, 0, idx % MAXATTR);
			}
			else {
			    wi->dev_info.xg_dot(wi->dev_info.user_state,
						Xsegs[1][X_idx - 1].x2,
						Xsegs[1][X_idx - 1].y2,
						P_PIXEL, 0, PIXVALUE(idx));
			}
		    }
		    else {
			/* Distinctive markers */
			wi->dev_info.xg_dot(wi->dev_info.user_state,
					    Xsegs[1][X_idx - 1].x2,
					    Xsegs[1][X_idx - 1].y2,
					    P_MARK, MARKSTYLE(idx),
					    PIXVALUE(idx));
		    }
		}
	    }
	    /* Handle last bar */
	    if ((thisList->numPoints > 0) && PM_BOOL("BarGraph")) {
		int     barPixels,
		        baseSpot;
		XSegment line;

		barPixels = (int) ((PM_DBL("BarWidth") / 
				   wi->XUnitsPerPixel) + 0.5);
		if (barPixels <= 0)
		    barPixels = 1;
		baseSpot = SCREENY(wi, PM_DBL("BarBase"));
		line.x1 = line.x2 = Xsegs[1][X_idx - 1].x2 +
					(int) ((PM_DBL("BarOffset") * idx /
					    wi->XUnitsPerPixel) + 0.5);
		if (PM_BOOL("StackGraph") && idx != 0)
		    line.y1 = Xsegs[0][X_idx - 1].y2;
		else
		    line.y1 = baseSpot;
		line.y2 = Xsegs[1][X_idx - 1].y2;
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &line, barPixels, L_VAR,
				    LINESTYLE(idx), PIXVALUE(idx));
	    }

	    /* Draw segments */
            if (!PM_BOOL("Animate")) {
	      if (thisList->numPoints > 0 && (!noLines) && (X_idx > 0)) {
		ptr = Xsegs[1];
		while (X_idx > wi->dev_info.max_segs) {
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					wi->dev_info.max_segs, ptr,
					lineWidth, L_VAR,
					LINESTYLE(idx), PIXVALUE(idx));
		    ptr += wi->dev_info.max_segs;
		    X_idx -= wi->dev_info.max_segs;
		}
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    X_idx, ptr,
				    lineWidth, L_VAR,
				    LINESTYLE(idx), PIXVALUE(idx));
	      }
            } else {
              StoreIDX = X_idx;
	      if (thisList->numPoints > 0 && (!noLines) && (X_idx > 0)) {
		ptr = Xsegs[1];
		while (X_idx > wi->dev_info.max_segs) {
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					wi->dev_info.max_segs, ptr,
					lineWidth, L_VAR,
					LINESTYLE(1), PIXVALUE(2));
		    ptr += wi->dev_info.max_segs;
		    X_idx -= wi->dev_info.max_segs;
		}
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    X_idx, ptr,
				    lineWidth, L_VAR,
				    LINESTYLE(1), PIXVALUE(2));
	      }
              XFlush (disp);
	      for (X_idx=1;X_idx<theDelay;X_idx++);
              X_idx = StoreIDX;
	      if ((thisList->numPoints > 0) && (!noLines) && (X_idx > 0)) {
		ptr = Xsegs[1];
		while (X_idx > wi->dev_info.max_segs) {
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					wi->dev_info.max_segs, ptr,
					lineWidth, L_VAR,
					16, (int)(1));
					/*LINESTYLE(8), (int)(1));*/
		    ptr += wi->dev_info.max_segs;
		    X_idx -= wi->dev_info.max_segs;
		}
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    X_idx, ptr,
				    lineWidth, L_VAR,
				    16,(int)(1));
	      }
            }
	    /* Next subset */
            lastList = thisList;
            thisList = thisList->next;
	} /* End While */
      }
        if (PM_BOOL("Animate")) {
          X_idx = StoreIDX;
          thisList = lastList;
	  if (thisList->numPoints > 0 && (!noLines) && (X_idx > 0)) {
	    ptr = Xsegs[1];
	       while (X_idx > wi->dev_info.max_segs) {
		    wi->dev_info.xg_seg(wi->dev_info.user_state,
					wi->dev_info.max_segs, ptr,
					lineWidth, L_VAR,
					LINESTYLE(1), PIXVALUE(2));
		    ptr += wi->dev_info.max_segs;
		    X_idx -= wi->dev_info.max_segs;
		}
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    X_idx, ptr,
				    lineWidth, L_VAR,
				    LINESTYLE(1), PIXVALUE(2));
	    }
          }
    XFlush (disp);
}



static void
DrawLegend(wi)
LocalWin *wi;

/*
 * This draws a legend of the data sets displayed.  Only those that
 * will fit are drawn.
 */
{
    int     idx,
            spot,
            lineLen,
            oneLen, 
	    incr;
    XSegment leg_line;
    int     markFlag,
            pixelMarks,
            bigPixel,
            colorMark;

    set_mark_flags(&markFlag, &pixelMarks, &bigPixel, &colorMark);
    spot = wi->XOrgY;
    lineLen = 0;
    incr = 2 + wi->dev_info.axis_height + wi->dev_info.bdr_pad;
    /* First pass draws the text */
    for (idx = 0; idx < MAXSETS; idx++) {
	if ((PlotData[idx].list) &&
	    (spot + wi->dev_info.axis_height + 2 < wi->XOppY)) {
	    /* Meets the criteria */
	    oneLen = strlen(PlotData[idx].setName);
	    if (oneLen > lineLen)
		lineLen = oneLen;
	    wi->dev_info.xg_text(wi->dev_info.user_state,
				 wi->XOppX + wi->dev_info.bdr_pad,
				 spot + 2,
				 PlotData[idx].setName,
				 T_UPPERLEFT, T_AXIS);
	    spot += incr;
	}
    }
    lineLen = lineLen * wi->dev_info.axis_width;
    leg_line.x1 = wi->XOppX + wi->dev_info.bdr_pad;
    leg_line.x2 = leg_line.x1 + lineLen;
    spot = wi->XOrgY;
    /* second pass draws the lines */
    for (idx = 0; idx < MAXSETS; idx++) {
	if ((PlotData[idx].list) &&
	    (spot + wi->dev_info.axis_height + 2 < wi->XOppY)) {
	    leg_line.y1 = leg_line.y2 = spot - wi->dev_info.legend_pad;
	    if (PM_BOOL("BarGraph")) 
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &leg_line,
				    incr / 4,
				    L_VAR, LINESTYLE(idx), PIXVALUE(idx));
	    if (!PM_BOOL("NoLines"))
		wi->dev_info.xg_seg(wi->dev_info.user_state,
				    1, &leg_line, 1, L_VAR,
				    LINESTYLE(idx), PIXVALUE(idx));
	    if (markFlag && !pixelMarks) {
		wi->dev_info.xg_dot(wi->dev_info.user_state,
				    leg_line.x1, leg_line.y1,
				    P_MARK, MARKSTYLE(idx), PIXVALUE(idx));

	    }
	    spot += incr;
	}
    }
}
static void
set_mark_flags(markFlag, pixelMarks, bigPixel, colorMark)
int    *markFlag;
int    *pixelMarks;
int    *bigPixel;
int    *colorMark;

/*
 * Determines the values of the old boolean flags based on the
 * new values in the parameters database.
 */
{
    *markFlag = 0;
    *pixelMarks = 0;
    *colorMark = 0;
    *bigPixel = 0;
    if (PM_BOOL("Markers")) {
	*markFlag = 1;
	*pixelMarks = 0;
	*colorMark = 0;
    }
    if (PM_BOOL("PixelMarkers")) {
	*markFlag = 1;
	*pixelMarks = 1;
	*bigPixel = 0;
    }
    if (PM_BOOL("LargePixels")) {
	*markFlag = 1;
	*pixelMarks = 1;
	*bigPixel = 1;
    }
    if (PM_BOOL("StyleMarkers")) {
	*markFlag = 1;
	*pixelMarks = 0;
	*colorMark = 1;
    }
}
 
#undef DELAY
