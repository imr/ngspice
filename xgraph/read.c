/* $Header$ */
/*
 * read.c: Dataset read code
 *
 * Routines:
 *	int ReadData();
 *
 * $Log$
 * Revision 1.1  2004-01-25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.2  1999/12/03 23:17:45  heideman
 * apply xgraph_no_animation.patch
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
/*
 * New dataset reading code
 */

static int setNumber = 0;
static PointList **curSpot = (PointList **) 0;
static PointList *curList = (PointList *) 0;
static int newGroup = 0;
static int redundant_set = 0;

#ifdef DO_DER
extern void Der1();
#endif

static int
rdSet(fn)
char   *fn;			/* Reading from file `fn' */

/*
 * Set up new dataset.  Will return zero if there are too many data sets.
 */
{
    char    setname[100];

    if (!redundant_set) {
	if (setNumber < MAXSETS) {
	    (void) sprintf(setname, "Set %d", setNumber);
	    if ((strcmp(PlotData[setNumber].setName, setname) == 0) && fn) {
		PlotData[setNumber].setName = fn;
	    }
	    curSpot = &(PlotData[setNumber].list);
	    PlotData[setNumber].list = (PointList *) 0;
	    newGroup = 1;
	    setNumber++;
	    redundant_set = 1;
	    return 1;
	}
	else {
	    return 0;
	}
    }
    else {
	return 1;
    }
}

static void
rdSetName(name)
char   *name;			/* New set name */

/*
 * Sets the name of a data set.  Automatically makes a copy.
 */
{
    PlotData[setNumber - 1].setName = STRDUP(name);
}

static void
rdGroup()
/*
 * Set up for reading new group of points within a dataset.
 */
{
    newGroup = 1;
}

static void
rdPoint(xval, yval)
double  xval,
        yval;			/* New point         */

/*
 * Adds a new point to the current group of the current
 * data set.
 */
{
    if (newGroup) {
	*curSpot = (PointList *) Malloc(sizeof(PointList));
	curList = *curSpot;
	curSpot = &(curList->next);
	curList->numPoints = 0;
	curList->allocSize = INITSIZE;
	curList->xvec = (double *) Malloc((unsigned)
					  (INITSIZE * sizeof(double)));
	curList->yvec = (double *) Malloc((unsigned)
					  (INITSIZE * sizeof(double)));
	curList->next = (PointList *) 0;
	newGroup = 0;
    }
    if (curList->numPoints >= curList->allocSize) {
	curList->allocSize *= 2;
	curList->xvec = (double *) Realloc((char *) curList->xvec,
					   (unsigned) (curList->allocSize *
						       sizeof(double)));
	curList->yvec = (double *) Realloc((char *) curList->yvec,
					   (unsigned) (curList->allocSize *
						       sizeof(double)));
    }

    curList->xvec[curList->numPoints] = xval;
    curList->yvec[curList->numPoints] = yval;

    (curList->numPoints)++;
    redundant_set = 0;
}

static int
rdFindMax()
/*
 * Returns the maximum number of items in any one group of any
 * data set.
 */
{
    int     i;
    PointList *list;
    int     max = -1;

    for (i = 0; i < setNumber; i++) {
	for (list = PlotData[i].list; list; list = list->next) {
	    if (list->numPoints > max)
		max = list->numPoints;
	}
    }
    return max;
}


typedef enum line_type {
    EMPTY, COMMENT, SETNAME, DRAWPNT, MOVEPNT, SETPARAM, ERROR
}       LineType;

typedef struct point_defn {
    double  xval,
            yval;
}       Point;

typedef struct parmval_defn {
    char   *name,
           *value;
}       ParmVals;

typedef struct line_info {
    LineType type;
    union val_defn {
	char   *str;		/* SETNAME, ERROR   */
	Point   pnt;		/* DRAWPNT, MOVEPNT */
	ParmVals parm;		/* SETPARAM         */
    }       val;
}       LineInfo;

static  LineType
parse_line(line, result)
char   *line;			/* Line to parse   */
LineInfo *result;		/* Returned result */

/*
 * Parses `line' into one of the types given in the definition
 * of LineInfo.  The appropriate values are filled into `result'.
 * Below are the current formats for each type:
 *   EMPTY:	All white space
 *   COMMENT:	Starts with "#"
 *   SETNAME:	A name enclosed in double quotes
 *   DRAWPNT:	Two numbers optionally preceded by keyword "draw"
 *   MOVEPNT:	Two numbers preceded by keyword "move"
 *   SETPARAM:  Two non-null strings separated by ":"
 *   ERROR:	Not any of the above (an error message is returned)
 * Note that often the values are pointers into the line itself
 * and should be copied if they are to be used over a long period.
 */
{
    char   *first;

    /* Find first non-space character */
    while (*line && isspace(*line))
	line++;
    if (*line) {
	if (*line == '#') {
	    /* comment */
	    result->type = COMMENT;
	}
	else if (*line == '"') {
	    /* setname */
	    result->type = SETNAME;
	    line++;
	    result->val.str = line;
	    while (*line && (*line != '\n') && (*line != '"'))
		line++;
	    if (*line)
		*line = '\0';
	}
	else {
	    first = line;
	    while (*line && !isspace(*line))
		line++;
	    if (*line) {
		*line = '\0';
		if (stricmp(first, "move") == 0) {
		    /* MOVEPNT */
		    if (sscanf(line + 1, "%lf %lf",
			       &result->val.pnt.xval,
			       &result->val.pnt.yval) == 2) {
			result->type = MOVEPNT;
		    }
		    else {
			result->type = ERROR;
			result->val.str = "Cannot read move coordinates";
		    }
		}
		else if (stricmp(first, "draw") == 0) {
		    /* DRAWPNT */
		    if (sscanf(line + 1, "%lf %lf",
			       &result->val.pnt.xval,
			       &result->val.pnt.yval) == 2) {
			result->type = DRAWPNT;
		    }
		    else {
			result->type = ERROR;
			result->val.str = "Cannot read draw coordinates";
		    }
		}
		else if (first[strlen(first) - 1] == ':') {
		    /* SETPARAM */
		    first[strlen(first) - 1] = '\0';
		    result->val.parm.name = first;
		    line++;
		    while (*line && isspace(*line))
			line++;
		    /* may be a \n at end of it */
		    if (line[strlen(line) - 1] == '\n') {
			line[strlen(line) - 1] = '\0';
		    }
		    result->val.parm.value = line;
		    result->type = SETPARAM;
		}
		else if (sscanf(first, "%lf", &result->val.pnt.xval) == 1) {
		    /* DRAWPNT */
		    if (sscanf(line + 1, "%lf", &result->val.pnt.yval) == 1) {
			result->type = DRAWPNT;
		    }
		    else {
			result->type = ERROR;
			result->val.str = "Cannot read second coordinate";
		    }
		}
		else {
		    /* ERROR */
		    result->type = ERROR;
		    result->val.str = "Unknown line type";
		}
	    }
	    else {
		/* ERROR */
		result->type = ERROR;
		result->val.str = "Premature end of line";
	    }
	}
    }
    else {
	/* empty */
	result->type = EMPTY;
    }
    return result->type;
}


int
ReadData(stream, filename)
FILE   *stream;
char   *filename;

/*
 * Reads in the data sets from the supplied stream.  If the format
 * is correct,  it returns the current maximum number of points across
 * all data sets.  If there is an error,  it returns -1.
 */
{
    char    buffer[MAXBUFSIZE];
    LineInfo info;
    int     line_count = 0;
    int     errors = 0;

    if (!rdSet(filename)) {
	(void) fprintf(stderr, "Error in file `%s' at line %d:\n  %s\n",
		       filename, line_count,
		       "Too many data sets - extra data ignored");
	return -1;
    }
    while (fgets(buffer, MAXBUFSIZE, stream)) {
	line_count++;
	switch (parse_line(buffer, &info)) {
	case EMPTY:
	    if (!rdSet(filename)) {
		(void) fprintf(stderr, "Error in file `%s' at line %d:\n  %s\n",
			       filename, line_count,
			       "Too many data sets - extra data ignored");
		return -1;
	    }
	    break;
	case COMMENT:
	    /* nothing */
	    break;
	case SETNAME:
	    rdSetName(info.val.str);
	    break;
	case DRAWPNT:
	    rdPoint(info.val.pnt.xval, info.val.pnt.yval);
	    break;
	case MOVEPNT:
	    rdGroup();
	    rdPoint(info.val.pnt.xval, info.val.pnt.yval);
	    break;
	case SETPARAM:
	    param_reset(info.val.parm.name, info.val.parm.value);
	    break;
	default:
	    if (filename) {
		(void) fprintf(stderr, "Error in file `%s' at line %d:\n  %s\n",
			       filename, line_count, info.val.str);
		errors++;
	    }
	    break;
	}
    }
#ifdef DO_DER
    Der1();
#endif
    if (errors)
	return -1;
    else
	return rdFindMax();
}


