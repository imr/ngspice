/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/


#include <ngspice.h>
#include <graph.h>
#include <ftedev.h>
#include <fteinput.h>
#include <cpdefs.h>     /* for VT_STRING */
#include <ftedefs.h>        /* for mylog() */

#include "display.h"
#include "variable.h"

/* static declarations */
static void gen_DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny);
static int gen_Input(REQUEST *request, RESPONSE *response);
static int nop(void);
static int nodev(void);




#ifndef X_DISPLAY_MISSING
extern int  X11_Init(void), X11_NewViewport(GRAPH *graph), X11_Close(void), X11_Clear(void),
        X11_DrawLine(int x1, int y1, int x2, int y2), X11_Arc(int x0, int y0, int radius, double theta1, double theta2), X11_Text(char *text, int x, int y), X11_DefineColor(int colorid, double red, double green, double blue),
        X11_DefineLinestyle(int linestyleid, int mask), X11_SetLinestyle(int linestyleid), X11_SetColor(int colorid),
        X11_Update(void),
        X11_Input(REQUEST *request, RESPONSE *response);
#endif


#ifdef HAS_WINDOWS	/* Graphic-IO under MS Windows */
extern int WIN_Init(), WIN_NewViewport(), WIN_Close(), WIN_Clear(),
		WIN_DrawLine(), WIN_Arc(), WIN_Text(), WIN_DefineColor(),
		WIN_DefineLinestyle(), WIN_SetLinestyle(), WIN_SetColor(),
		WIN_Update(), WIN_DiagramReady();

extern int WPRINT_Init(), WPRINT_NewViewport(), WPRINT_Close(), WPRINT_Clear(),
		WPRINT_DrawLine(), WPRINT_Arc(), WPRINT_Text(), WPRINT_DefineColor(),
		WPRINT_DefineLinestyle(), WPRINT_SetLinestyle(), WPRINT_SetColor(),
		WPRINT_Update(), WPRINT_DiagramReady();
#endif



extern int  Plt5_Init(void), Plt5_NewViewport(GRAPH *graph), Plt5_Close(void), Plt5_Clear(void),
        Plt5_DrawLine(int x1, int y1, int x2, int y2), Plt5_Arc(int x0, int y0, int radius, double theta1, double theta2), Plt5_Text(char *text, int x, int y),
        Plt5_DefineLinestyle(), Plt5_SetLinestyle(int linestyleid), Plt5_SetColor(int colorid),
        Plt5_Update(void);

extern int  PS_Init(void), PS_NewViewport(GRAPH *graph), PS_Close(void), PS_Clear(void),
        PS_DrawLine(int x1, int y1, int x2, int y2), PS_Arc(int x0, int y0, int r, double theta1, double theta2), PS_Text(char *text, int x, int y),
        PS_DefineLinestyle(), PS_SetLinestyle(int linestyleid), PS_SetColor(int colorid),
        PS_Update(void);

extern int  GL_Init(void), GL_NewViewport(GRAPH *graph), GL_Close(void), GL_Clear(void),
        GL_DrawLine(int x1, int y1, int x2, int y2), GL_Arc(int x0, int y0, int r, double theta1, double theta2), GL_Text(char *text, int x, int y),
        GL_DefineLinestyle(), GL_SetLinestyle(int linestyleid), GL_SetColor(int colorid),
        GL_Update(void);

DISPDEVICE device[] = {

    {"error", 0, 0, 0, 0, 0, 0, nop, nop,
    nop, nop,
    nop, nop, nop, nop, nop,
    nop, nop, nop,
    nop, nop, nop, gen_Input,
    (void *)nop,},

#ifndef X_DISPLAY_MISSING
    {"X11", 0, 0, 1024, 864, 0, 0, X11_Init, X11_NewViewport,
    X11_Close, X11_Clear,
    X11_DrawLine, X11_Arc, X11_Text, X11_DefineColor, X11_DefineLinestyle,
    X11_SetLinestyle, X11_SetColor, X11_Update,
    nodev, nodev, nodev, X11_Input,
    gen_DatatoScreen,},
#endif

#ifdef HAS_WINDOWS	/* Graphic-IO under MS Windows */
	{"Windows", 0, 0, 1000, 1000, 0, 0, WIN_Init, WIN_NewViewport,
	 WIN_Close, WIN_Clear,
	 WIN_DrawLine, WIN_Arc, WIN_Text, WIN_DefineColor, WIN_DefineLinestyle,
	 WIN_SetLinestyle, WIN_SetColor, WIN_Update,
	 nodev, nodev, nodev, gen_Input,
	 gen_DatatoScreen, WIN_DiagramReady},

	// Warning: name "WinPrint" do not change!
	{"WinPrint", 0, 0, 1000, 1000, 0, 0, WPRINT_Init, WPRINT_NewViewport,
	 WPRINT_Close, WPRINT_Clear,
	 WPRINT_DrawLine, WPRINT_Arc, WPRINT_Text, WPRINT_DefineColor, WPRINT_DefineLinestyle,
	 WPRINT_SetLinestyle, WPRINT_SetColor, WPRINT_Update,
	 nodev, nodev, nodev, nodev,
	 gen_DatatoScreen, WPRINT_DiagramReady},

#endif


    {"plot5", 0, 0, 1000, 1000, 0, 0, Plt5_Init, Plt5_NewViewport,
    Plt5_Close, Plt5_Clear,
    Plt5_DrawLine, Plt5_Arc, Plt5_Text, nodev, nodev,
    Plt5_SetLinestyle, Plt5_SetColor, Plt5_Update,
    nodev, nodev, nodev, nodev,
    gen_DatatoScreen,},

    {"postscript", 0, 0, 1000, 1000, 0, 0, PS_Init, PS_NewViewport,
    PS_Close, PS_Clear,
    PS_DrawLine, PS_Arc, PS_Text, nodev, nodev,
    PS_SetLinestyle, PS_SetColor, PS_Update,
    nodev, nodev, nodev, nodev,
    gen_DatatoScreen,},

    {"hpgl", 0, 0, 1000, 1000, 0, 0, GL_Init, GL_NewViewport,
    GL_Close, GL_Clear,
    GL_DrawLine, GL_Arc, GL_Text, nodev, nodev,
    GL_SetLinestyle, GL_SetColor, GL_Update,
    nodev, nodev, nodev, nodev,
    gen_DatatoScreen,},

    {"printf", 0, 0, 24, 80, 0, 0, nodev, nodev,
    nodev, nodev,
    nodev, nodev, nodev, nodev, nodev,
    nodev, nodev, nodev,
    nodev, nodev, nodev, gen_Input,
    (void *)nodev,},

};

DISPDEVICE *dispdev = device + NUMELEMS(device) - 1;

#define XtNumber(arr)       (sizeof(arr) / sizeof(arr[0]))


extern void internalerror (char *message);
extern void externalerror (char *message);

DISPDEVICE *FindDev(char *name)
{
    int i;

    for (i=0; i < XtNumber(device); i++) {
      if (!strcmp(name, device[i].name)) {
        return(&device[i]);
      }
    }
    sprintf(ErrorMessage, "Can't find device %s.", name);
    internalerror(ErrorMessage);
    return(&device[0]);

}

void
DevInit(void)
{
#ifndef X_DISPLAY_MISSING
    char buf[128];   /* va: used with NOT X_DISPLAY_MISSING only */
#endif /* X_DISPLAY_MISSING */

/* note: do better determination */

/*
    dumb tradition that got passed on from gi_interface
    to do compile time determination
*/

    dispdev = NULL;

#ifndef X_DISPLAY_MISSING
    /* determine display type */
    if (getenv("DISPLAY") || cp_getvar("display", VT_STRING, buf)) {
       dispdev = FindDev("X11");
    }
#endif


#ifdef HAS_WINDOWS
	 if (!dispdev) {
      dispdev = FindDev("Windows");
    }
#endif

    if (!dispdev) {
	externalerror(
	 "no graphics interface; please check compiling instructions");
	dispdev = FindDev("error");
    } else if ((*(dispdev->Init))()) {
      fprintf(cp_err,
        "Warning: can't initialize display device for graphics.\n");
      dispdev = FindDev("error");
    }

}

/* NewViewport is responsible for filling in graph->viewport */
int
NewViewport(GRAPH *pgraph)
{

    return (*(dispdev->NewViewport))(pgraph);

}

void DevClose(void)
{

    (*(dispdev->Close))();

}

void DevClear(void)
{

    (*(dispdev->Clear))();

}

void DrawLine(int x1, int y1, int x2, int y2)
{
    (*(dispdev->DrawLine))(x1, y1, x2, y2);

}

void Arc(int x0, int y0, int radius, double theta1, double theta2)
{

    (*(dispdev->Arc))(x0, y0, radius, theta1, theta2);

}

void Text(char *text, int x, int y)
{

    (*(dispdev->Text))(text, x, y);

}

void DefineColor(int colorid, double red, double green, double blue)
{

    (*(dispdev->DefineColor))(colorid, red, green, blue);

}

void DefineLinestyle(int linestyleid, int mask)
{

    (*(dispdev->DefineLinestyle))(linestyleid, mask);

}

void SetLinestyle(int linestyleid)
{

    (*(dispdev->SetLinestyle))(linestyleid);

}

void SetColor(int colorid)
{

    (*(dispdev->SetColor))(colorid);

}

void Update(void)
{

    if (dispdev)
	    (*(dispdev->Update))();

}

/* note: screen coordinates are relative to window
    so need to add viewport offsets */
static void
gen_DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny)
{

    double low, high;

    /* note: may want to cache datawindowsize/viewportsize */ /* done */

    /* note: think this out---Is 1 part of the viewport? Do we handle
        this correctly? */

    /* have to handle several types of grids */

    /* note: we can't compensate for X's demented y-coordinate system here
        since the grid routines use DrawLine w/o calling this routine */
    if ((graph->grid.gridtype == GRID_LOGLOG) ||
            (graph->grid.gridtype == GRID_YLOG)) {
      low = mylog10(graph->datawindow.ymin);
      high = mylog10(graph->datawindow.ymax);
      *screeny = (mylog10(y) - low) / (high - low) * graph->viewport.height
	  + 0.5 + graph->viewportyoff;
    } else {
      *screeny = ((y - graph->datawindow.ymin) / graph->aspectratioy)
            + 0.5 + graph->viewportyoff;
    }

    if ((graph->grid.gridtype == GRID_LOGLOG) ||
            (graph->grid.gridtype == GRID_XLOG)) {
      low = mylog10(graph->datawindow.xmin);
      high = mylog10(graph->datawindow.xmax);
      *screenx = (mylog10(x) - low) / (high - low) * graph->viewport.width
            + 0.5 + graph ->viewportxoff;
    } else {
      *screenx = (x - graph->datawindow.xmin) / graph->aspectratiox
            + 0.5 + graph ->viewportxoff;
    }

}

void DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny)
{

    (*(dispdev->DatatoScreen))(graph, x, y, screenx, screeny);

}

void Input(REQUEST *request, RESPONSE *response)
{

    (*(dispdev->Input))(request, response);

}

static int
gen_Input(REQUEST *request, RESPONSE *response)
{

    switch (request->option) {
      case char_option:
        response->reply.ch = inchar(request->fp);
        response->option = request->option;
        break;
      default:
        /* just ignore, since we don't want a million error messages */
	if (response)
	    response->option = error_option;
        break;
    }
return 0;
}

/* no operation, do nothing */
static int nop(void)
{
    return(1);  /* so NewViewport will fail */
}

static int nodev(void)
{

    sprintf(ErrorMessage,
        "This operation is not defined for display type %s.",
        dispdev->name);
    internalerror(ErrorMessage);
    return(1);

}

void SaveText(GRAPH *graph, char *text, int x, int y)
{

    struct _keyed *keyed;

    keyed = (struct _keyed *) tmalloc(sizeof(struct _keyed));

    if (!graph->keyed) {
      graph->keyed = keyed;
    } else {
      keyed->next = graph->keyed;
      graph->keyed = keyed;
    }

    keyed->text = tmalloc(strlen(text) + 1);
    strcpy(keyed->text, text);

    keyed->x = x;
    keyed->y = y;

    keyed->colorindex = graph->currentcolor;

}

/* if given name of a hardcopy device, finds it and switches devices
   if given NULL, switches back */
int DevSwitch(char *devname)
{

    static DISPDEVICE *lastdev = NULL;

    if (devname != NULL) {
      if (lastdev != NULL) {
        internalerror("DevSwitch w/o changing back");
        return (1);
      }
      lastdev = dispdev;
      dispdev = FindDev(devname);
      if (!strcmp(dispdev->name, "error")) {
        internalerror("no hardcopy device");
        dispdev = lastdev;  /* undo */
        lastdev = NULL;
        return (1);
      }
      (*(dispdev->Init))();
    } else {
      (*(dispdev->Close))();
      dispdev = lastdev;
      lastdev = NULL;
    }
    return(0);

}
