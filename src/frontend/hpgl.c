/**********
Author: Jim Groves
**********/

/*
    HPGL driver
*/

/*
   1000 plotter units / inch  - 1pu = 0.025mm  1pu = 1mil

  SP - select pen
  PU - pen up (PU x,y)
  PD - pen down (PD x,y)
  LT - line type
       0   dots only at plotted points
       1   .   .    .   .    .
       2   ___     ___     ___     ___
       3   ----    ----    ----    ----
       4   ----- . ----- . ----- . -----.
       5   ---- -  ---- -  ---- -
       6   --- - - --- - - --- - - --- - -
       null - solid line
  IN - initialize
  DF - default values (PA, solid line, set 0)
  PA - plot absolute
  SI - absolute character size (SI width, height) in cm

*/

#include "ngspice.h"
#include "cpdefs.h"
#include "graph.h"
#include "ftedbgra.h"
#include "ftedev.h"
#include "fteinput.h"
#include "variable.h"
#include "error.h"
#include "plotting/graphdb.h"
#include "hpgl.h"

#define RAD_TO_DEG      (180.0 / M_PI)
#define DEVDEP(g) (*((GLdevdep *) (g)->devdep))
#define MAX_GL_LINES    9999
#define SOLID 0
#define DOTTED 1

#define gtype         graph->grid.gridtype
#define xoff          dispdev->minx
#define yoff          dispdev->miny
#define XOFF          25      /* printer left margin */
#define YOFF          28      /* printer bottom margin */
#define XTADJ         0       /* printer text adjustment x */
#define YTADJ         0       /* printer text adjustment y */

#define DELXMAX       360     /* printer gridsize divisible by 10, [7-2] */
#define DELYMAX       360     /* printer gridsize divisible by [10-8], [6-2] */

#define FONTWIDTH     6       /* printer default fontwidth */
#define FONTHEIGHT    8       /* printer default fontheight */

typedef struct {
    int lastlinestyle;      /* initial invalid value */
    int lastx, lasty, linecount;
} GLdevdep;

static char *linestyle[] = {
                "",             /* solid */
                "1",        /* was 1 - dotted */
                "",        /* longdashed */
                "3",        /* shortdashed */
                "4",    /* longdotdashed */
                "5",    /* shortdotdashed */
                "1"
                };

static FILE *plotfile;
char psfont[128], psfontsize[32], psscale[32];
static int fontwidth  = FONTWIDTH;
static int fontheight = FONTHEIGHT;
static int   jgmult      = 10;
static int screenflag = 0;
static double tocm = 0.0025;
static double scale;    /* Used for fine tuning */
static int hcopygraphid;

int GL_Init()
{
    if (!cp_getvar("hcopyscale", VT_STRING, psscale)) {
        scale = 1.0;
    } else {
        sscanf(psscale, "%lf", &scale);
        if ((scale <= 0) || (scale > 10))
            scale = 1.0;
    }

    dispdev->numlinestyles = NUMELEMS(linestyle);
    dispdev->numcolors = 6;

    dispdev->width = DELXMAX * scale;
    dispdev->height = DELYMAX * scale;


    screenflag = 0;
    dispdev->minx = XOFF * 1.0;
    dispdev->miny = YOFF * 1.0;

    return(0);

}

/* devdep initially contains name of output file */
int GL_NewViewport(graph)
GRAPH *graph;
{
/*    double scaleps, scalex, scaley; */

    hcopygraphid = graph->graphid;

    if (!(plotfile = fopen(graph->devdep, "w"))) {
      perror(graph->devdep);
      graph->devdep = (char *) NULL;
      return(1);
    }

    if (graph->absolute.width) {
      /* hardcopying from the screen */

      screenflag = 1;

      /* scale to fit on 8 1/2 square */

    }

    /* reasonable values, used in gr_ for placement */
    graph->fontwidth = fontwidth * scale; /* was 12, p.w.h. */
    graph->fontheight = fontheight * scale; /* was 24, p.w.h. */

    graph->absolute.width = dispdev->width;
    graph->absolute.height = dispdev->height;
    /* Also done in gr_init, if called . . . */
    graph->viewportxoff = 16 * fontwidth;
    graph->viewportyoff = 8 * fontheight;

    xoff = XOFF;
    yoff = YOFF;

    /* start file off with a % */
    fprintf(plotfile, "IN;DF;PA;");
    fprintf(plotfile, "SI %f,%f;", tocm*jgmult*fontwidth*scale,tocm*jgmult*fontheight*scale);

#ifdef notdef
    if (!screenflag)
#endif

    graph->devdep = tmalloc(sizeof(GLdevdep));
    DEVDEP(graph).lastlinestyle = -1;
    DEVDEP(graph).lastx = -1;
    DEVDEP(graph).lasty = -1;
    DEVDEP(graph).linecount = 0;
    graph->linestyle = -1;

    return 0;
}

int GL_Close()
{

    /* in case GL_Close is called as part of an abort,
            w/o having reached GL_NewViewport */
    if (plotfile) {
      if (DEVDEP(currentgraph).lastlinestyle != -1) {
        DEVDEP(currentgraph).linecount = 0;
      }
      fclose(plotfile);
      plotfile = NULL;
    }
    /* In case of hardcopy command destroy the hardcopy graph
     * and reset currentgraph to graphid 1, if possible
     */
    if (!screenflag) {
      DestroyGraph(hcopygraphid);
      currentgraph = FindGraph(1);
    }

    return 0;
}

int GL_Clear()
{

    /* do nothing */


    return 0;
}

int GL_DrawLine(x1, y1, x2, y2)
int x1, y1, x2, y2;
{

    /* note: this is not extendible to more than one graph
        => will have to give NewViewport a writeable graph XXX */


    if (DEVDEP(currentgraph).linecount == 0
            || x1 != DEVDEP(currentgraph).lastx
            || y1 != DEVDEP(currentgraph).lasty)
    {
      fprintf(plotfile, "PU;PA %d , %d ;", jgmult*(x1 + xoff), jgmult*(y1 + yoff));
    }
    if (x1 != x2 || y1 != y2) {
        fprintf(plotfile, "PD;PA %d , %d ;", jgmult*(x2 + xoff), jgmult*(y2 + yoff));
        DEVDEP(currentgraph).linecount += 1;
    }

    DEVDEP(currentgraph).lastx = x2;
    DEVDEP(currentgraph).lasty = y2;
    DEVDEP(currentgraph).lastlinestyle = currentgraph->linestyle;

    return 0;
}

/* ARGSUSED */
int GL_Arc(x0, y0, r, theta1, theta2)
int x0, y0, r;
double theta1, theta2;
{
    double x1, y1;
    double angle1, angle2;

    while (theta1 >= theta2)
        theta2 += 2 * M_PI;

    angle1 = (double) (RAD_TO_DEG * theta1);
    angle2 = (double) (RAD_TO_DEG * theta2);
    x1 = (double) x0 + r * cos(theta1);
    y1 = (double) y0 + r * sin(theta1);
/*
    fprintf(plotfile, "%lf %lf moveto ", x1+(double)xoff, y1+(double)yoff);
    fprintf(plotfile, "%d %d %d %lf %lf arc\n", x0+xoff, y0+yoff, r,
        angle1, angle2);
    fprintf(plotfile, "stroke\n");
  */
    DEVDEP(currentgraph).linecount = 0;

    return 0;
}

int GL_Text(text, x, y)
char *text;
int x, y;
{

/*    int savedlstyle; */


    /* move to (x, y) */

    fprintf(plotfile, "PU;PA %d , %d;", jgmult*(x+xoff+XTADJ), jgmult*(y+yoff+YTADJ));
    fprintf(plotfile, "LB %s \x03", text);

    DEVDEP(currentgraph).lastx = -1;
    DEVDEP(currentgraph).lasty = -1;


    return 0;
}

int
GL_SetLinestyle(linestyleid)
int linestyleid;
{

    /* special case
        get it when GL_Text restores a -1 linestyle */
    if (linestyleid == -1) {
      currentgraph->linestyle = -1;
      return 0;
    }

    if (linestyleid < 0 || linestyleid > dispdev->numlinestyles) {
      internalerror("bad linestyleid");
      return 0;
    }

    if (currentgraph->linestyle != linestyleid) {
      fprintf(plotfile, "LT %s ;", linestyle[linestyleid]);
      currentgraph->linestyle = linestyleid;
    }
    return 0;

}

/* ARGSUSED */
int GL_SetColor(colorid)
int colorid;
{
/*va: unused:    static int flag = 0;*/        /* A hack */

    fprintf(plotfile, "SP %d;", colorid);

    return 0;
}

int GL_Update()
{

    fflush(plotfile);

    return 0;
}
