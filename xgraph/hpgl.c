/*
 * HPGL Output
 *
 * Tom Quarles
 */

#define MAPX(state,x) ( (x) + P1X + state->clipminX )
#define MAPY(state,y) ( MAXY - (y) + P1Y - state->clipminY)

#include "copyright.h"
#include "xgraph.h"
#include "plotter.h"
#include <stdio.h>
#include <math.h>

static void hpglText();
static void hpglSeg();
static void hpglDot();
static void hpglEnd();

static xgOut hpglInfo =
{
    D_COLOR,			/* device characteristics */
    MAXX,			/* width */
    MAXY,			/* height */
    200,			/* border padding */
    0,				/* extra space around axis labels */
    250,			/* tick length - approx 1/4 inch */
    50,				/* spacing above legend lables */
    0,				/* axis font width */
    0,				/* axis font height */
    0,				/* title font width */
    0,				/* title font height */
    1000000,			/* maximum number of segments */

    hpglText,			/* text output function */
    hpglSeg,			/* segment  drawing function */
    hpglDot,			/* dot/marker drawing function */
    hpglEnd,			/* end of plot function */

    NULL,			/* userInfo */
};

typedef struct {
    double  axis_w;
    double  axis_h;
    double  title_w;
    double  title_h;
    FILE   *plotterFile;
    int     clipminX;
    int     clipminY;
    int     clipmaxX;
    int     clipmaxY;
}       mydata;

/*ARGSUSED*/
int
hpglInit(stream, width, height, title_family, title_size,
	 axis_family, axis_size, flags, outInfo, errmsg)
FILE   *stream;			/* output stream */
int     width;			/* desired width of space in microns */
int     height;			/* desired height in microns */
char   *title_family;		/* name of font for titles */
double  title_size;		/* size of font for titles */
char   *axis_family;		/* name of font for axes */
double  axis_size;		/* size of font for axes */
int     flags;			/* predicate values (ignored) */
xgOut  *outInfo;		/* my structure */
char    errmsg[ERRBUFSIZE];	/* a place to complain to */
{
    mydata *myInfo;

    myInfo = (mydata *) Malloc(sizeof(mydata));
    if (myInfo == NULL)
	return 0;
    *outInfo = hpglInfo;
    outInfo->area_w = MIN(MAXX, width / 25);
    outInfo->area_h = MIN(MAXY, height / 25);
    /* magic formulas:  input sizes are in points = 1/72 inch */
    /* my sizes are in cm */
    /* plotter units are in units of .025mm ~= 1/1016 inch */
    /*
     * have to warn of height 1.5 times larger or get bitten by plotter's
     * internal padding
     */
    /* widths are (arbitrarily) selected to be 2/3 of the height */
    /* (cancels with width factor) */
    myInfo->axis_w = axis_size * .666 * 2.54 / 72.;
    myInfo->axis_h = axis_size * 2.54 / 72.;
    myInfo->title_w = title_size * .666 * 2.54 / 72.;
    myInfo->title_h = title_size * 2.54 / 72.;

    outInfo->axis_pad = axis_size * 1016. * 1.5 / 72.;
    outInfo->axis_width = axis_size * 1016. * 1.5 / 72.;
    outInfo->axis_height = axis_size * 1016. * .666 / 72.;
    outInfo->title_width = title_size * 1016. * 1.5 / 72.;
    outInfo->title_height = title_size * 1016. * .666 / 72.;
    outInfo->user_state = (char *) myInfo;
    myInfo->plotterFile = stream;
    myInfo->clipminX = 0;
    myInfo->clipminY = 0;
    myInfo->clipmaxX = MAXX;
    myInfo->clipmaxY = MAXY;
    fprintf(myInfo->plotterFile, "PG;IN;\n");
    fprintf(myInfo->plotterFile, "DI1,0;\n");
    fprintf(myInfo->plotterFile, "IW%d,%d,%d,%d;\n", MAPX(myInfo, 0),
	    MAPY(myInfo, myInfo->clipmaxY - myInfo->clipminY),
	    MAPX(myInfo, myInfo->clipmaxX - myInfo->clipminX),
	    MAPY(myInfo, 0));
    return (1);
}

static void
hpglText(userState, x, y, text, just, style)
mydata *userState;		/* my state information  */
int     x,
        y;			/* coords of text origin */
char   *text;			/* what to put there */
int     just;			/* how to justify */

 /*
  * where the origin is relative to where the text should go as a function of
  * the various values of just
  * 
  * T_UPPERLEFT     T_TOP       T_UPPERRIGHT T_LEFT          T_CENTER    T_RIGHT
  * T_LOWERLEFT     T_BOTTOM    T_LOWERRIGHT
  * 
  */
int     style;			/* T_AXIS = axis font, T_TITLE = title font */

{
    fprintf(userState->plotterFile, "PU;SP%d;", TEXTCOLOR);
    fprintf(userState->plotterFile, "PA%d,%d;", MAPX(userState, x), MAPY(userState, y));
    switch (style) {
    case T_AXIS:
	fprintf(userState->plotterFile, "SI%f,%f;", userState->axis_w, userState->axis_h);
	break;
    case T_TITLE:
	fprintf(userState->plotterFile, "SI%f,%f;", userState->title_w, userState->title_h);
	break;
    default:
	printf("bad text style %d in hpglText\n", style);
	exit(1);
	break;
    }
    switch (just) {
    case T_UPPERLEFT:
	fprintf(userState->plotterFile, "LO3;\n");
	break;
    case T_TOP:
	fprintf(userState->plotterFile, "LO6;\n");
	break;
    case T_UPPERRIGHT:
	fprintf(userState->plotterFile, "LO9;\n");
	break;
    case T_LEFT:
	fprintf(userState->plotterFile, "LO2;\n");
	break;
    case T_CENTER:
	fprintf(userState->plotterFile, "LO5;\n");
	break;
    case T_RIGHT:
	fprintf(userState->plotterFile, "LO8;\n");
	break;
    case T_LOWERLEFT:
	fprintf(userState->plotterFile, "LO1;\n");
	break;
    case T_BOTTOM:
	fprintf(userState->plotterFile, "LO4;\n");
	break;
    case T_LOWERRIGHT:
	fprintf(userState->plotterFile, "LO7;\n");
	break;
    default:
	printf("bad justification type %d in hpglText\n", just);
	exit(1);
	break;
    }
    fprintf(userState->plotterFile, "LB%s\03;", text);
}



static int penselect[8] =
{PEN1, PEN2, PEN3, PEN4, PEN5, PEN6, PEN7, PEN8};
static int lineselect[8] =
{LINE1, LINE2, LINE3, LINE4, LINE5, LINE6,
 LINE7, LINE8};



static void
hpglSeg(userState, ns, segs, width, style, lappr, color)
mydata *userState;		/* my state information (not used) */
int     ns;			/* number of segments */
XSegment *segs;			/* X array of segments */
int     width;			/* width of lines in pixels */
int     style;			/* L_VAR = dotted, L_AXIS = grid, L_ZERO = axis */
int     lappr;			/* line style */
int     color;			/* line color */
{
    int     i;

    if (style == L_ZERO) {
	fprintf(userState->plotterFile, "SP%d;", PENAXIS);	/* select correct pen */
	fprintf(userState->plotterFile, "LT;");	/* solid line style */
    }
    else if (style == L_AXIS) {
	fprintf(userState->plotterFile, "SP%d;", PENGRID);	/* select correct pen */
	fprintf(userState->plotterFile, "LT;");	/* solid line style */
    }
    else if (style == L_VAR) {
	if ((color < 0) || (color > 7)) {
	    printf("out of range line color %d in hpglLine\n", color);
	    exit(1);
	}
	fprintf(userState->plotterFile, "SP%d;", penselect[color]);	/* select correct pen */
	if ((lappr < 0) || (lappr > 7)) {
	    printf("out of range line style %d in hpglLine\n", lappr);
	    exit(1);
	}
	if (lappr == 0) {
	    fprintf(userState->plotterFile, "LT;");	/* select solid line
							 * type */
	}
	else {
	    fprintf(userState->plotterFile, "LT%d;", lineselect[lappr]);	/* select line type */
	}
    }
    else {
	printf("unknown style %d in hpglLine\n", style);
	exit(1);
    }
    for (i = 0; i < ns; i++) {
	if (!i || ((segs[i].x1 != segs[i - 1].x2) || (segs[i].y1 != segs[i - 1].y2))) {
	    /* MOVE */
	    fprintf(userState->plotterFile, "PU;PA%d,%d;\n", MAPX(userState, segs[i].x1),
		    MAPY(userState, segs[i].y1));
	}
	/* DRAW */
	if (width <= 1) {
	    fprintf(userState->plotterFile, "PD;PA%d,%d;\n", MAPX(userState, segs[i].x2),
		    MAPY(userState, segs[i].y2));
	}
	else {			/* ugly - wide lines -> rectangles */
	    double  frac;
	    int     lx,
	            ly;
	    int     urx,
	            ury,
	            ulx,
	            uly,
	            llx,
	            lly,
	            lrx,
	            lry;

	    frac = (width / 2) / sqrt((double)
				      ((segs[i].x1 - segs[i].x2) *
				       (segs[i].x1 - segs[i].x2)) +
				      ((segs[i].y1 - segs[i].y2) *
				       (segs[i].y1 - segs[i].y2)));
	    lx = frac * (segs[i].y2 - segs[i].y1);
	    ly = -frac * (segs[i].x2 - segs[i].x1);
	    urx = segs[i].x2 + lx;
	    ury = segs[i].y2 + ly;
	    ulx = segs[i].x2 - lx;
	    uly = segs[i].y2 - ly;
	    llx = segs[i].x1 - lx;
	    lly = segs[i].y1 - ly;
	    lrx = segs[i].x1 + lx;
	    lry = segs[i].y1 + ly;
	    fprintf(userState->plotterFile, "PU;PA%d,%d;", MAPX(userState, llx),
		    MAPY(userState, lly));
	    fprintf(userState->plotterFile, "PM0;");
	    fprintf(userState->plotterFile, "PD,PA%d,%D;PA%d,%D;PA%d,%d;\n",
		    MAPX(userState, lrx), MAPY(userState, lry),
		    MAPX(userState, urx), MAPY(userState, ury),
		    MAPX(userState, ulx), MAPY(userState, uly));
	    fprintf(userState->plotterFile, "PM2;FP;EP;");
	}
    }
    fprintf(userState->plotterFile, "PU;");
}

static char *markselect[8] =
{MARK1, MARK2, MARK3, MARK4, MARK5, MARK6,
 MARK7, MARK8};

static void
hpglDot(userState, x, y, style, type, color)
mydata *userState;		/* my state information (not used) */
int     x,
        y;			/* coord of dot */
int     style;			/* type of dot */
int     type;			/* dot style variation */
int     color;			/* color of dot */
{
    /* move to given coord */
    fprintf(userState->plotterFile, "PU;PA%d,%d;\n", MAPX(userState, x), MAPY(userState, y));
    if ((color < 0) || (color > 7)) {
	printf("unknown color %d in hpglDot\n", color);
	exit(1);
    }
    fprintf(userState->plotterFile, "SP%d;", penselect[color]);
    if (style == P_PIXEL) {
	fprintf(userState->plotterFile, "PD;PU;\n");
    }
    else if (style == P_DOT) {
	fprintf(userState->plotterFile, "LT;PM0;CI40;PM2;FT;EP;\n");
    }
    else if (style == P_MARK) {
	if ((type < 0) || (type > 7)) {
	    printf("unknown marker type %d in hpglDot\n", type);
	    exit(1);
	}
	/*
	 * fprintf(userState->plotterFile,"LT;CA5;LO4;SI0.1;LB%s\03;\n",marksele
	 * ct[type]);
	 */
	fprintf(userState->plotterFile, "LT;CS5;LO4;SI0.15;SM%s;PR0,0;SM;CS;\n", markselect[type]);
    }
    else {
	printf("unknown marker style %d in hpglDot\n", style);
	exit(1);
    }
}

static void
hpglEnd(userState)
mydata *userState;		/* my state information (not used) */

{
    fprintf(userState->plotterFile, "SP;PG;IN;\n");
    fflush(userState->plotterFile);
    return;
}
