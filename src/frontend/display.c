/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/


#include "ngspice/ngspice.h"
#include "ngspice/graph.h"
#include "ngspice/ftedev.h"
#include "ngspice/fteinput.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"

#ifdef TCL_MODULE
#include "ngspice/tclspice.h"
#endif

#include "display.h"
#include "variable.h"


static void gen_DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny);
static int gen_Input(REQUEST *request, RESPONSE *response);
static int nop(void);
#define NOP ((void *)nop)
static int nodev(void);
#define NODEV ((void *)nodev)

#ifndef X_DISPLAY_MISSING
#include "plotting/x11.h"
#endif

#ifdef HAS_WINGUI      /* Graphic-IO under MS Windows */
#include "wdisp/windisp.h"
//#include "wdisp/winprint.h"
#endif

#include "plotting/plot5.h"
#include "postsc.h"
#include "hpgl.h"
#include "svg.h"

DISPDEVICE device[] = {

    { "error", 0, 0, 0, 0, 0, 0,
      (disp_fn_Init_t *) NOP, (disp_fn_NewViewport_t *) NOP,
      (disp_fn_Close_t *) NOP, (disp_fn_Clear_t *) NOP,
      (disp_fn_DrawLine_t *) NOP, (disp_fn_Arc_t *) NOP, (disp_fn_Text_t *) NOP,
      (disp_fn_DefineColor_t *) NOP, (disp_fn_DefineLinestyle_t *) NOP,
      (disp_fn_SetLinestyle_t *) NOP, (disp_fn_SetColor_t *) NOP, (disp_fn_Update_t *) NOP, (disp_fn_Finalize_t *) NOP,
      (disp_fn_Track_t *) NOP, (disp_fn_MakeMenu_t *) NOP, (disp_fn_MakeDialog_t *) NOP, gen_Input,
      (disp_fn_DatatoScreen_t *) NOP,},

#ifndef X_DISPLAY_MISSING
    { "X11", 0, 0, 1024, 864, 0, 0,
      X11_Init, X11_NewViewport,
      X11_Close, X11_Clear,
      X11_DrawLine, X11_Arc, X11_Text,
      X11_DefineColor, X11_DefineLinestyle,
      X11_SetLinestyle, X11_SetColor, X11_Update,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, X11_Input,
      gen_DatatoScreen,},
#endif

#ifdef HAS_WINGUI      /* Graphic-IO under MS Windows */
    { "Windows", 0, 0, 1000, 1000, 0, 0,
      WIN_Init, WIN_NewViewport,
      WIN_Close, WIN_Clear,
      WIN_DrawLine, WIN_Arc, WIN_Text,
      WIN_DefineColor, WIN_DefineLinestyle,
      WIN_SetLinestyle, WIN_SetColor, WIN_Update,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, gen_Input,
      gen_DatatoScreen, }, /* WIN_DiagramReady */

    /* Warning: name "WinPrint" do not change! */
    { "WinPrint", 0, 0, 1000, 1000, 0, 0,
      WPRINT_Init, WPRINT_NewViewport,
      WPRINT_Close, WPRINT_Clear,
      WPRINT_DrawLine, WPRINT_Arc, WPRINT_Text,
      WPRINT_DefineColor, WPRINT_DefineLinestyle,
      WPRINT_SetLinestyle, WPRINT_SetColor, WPRINT_Update,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, (disp_fn_Input_t *) NODEV,
      gen_DatatoScreen, }, /* WPRINT_DiagramReady */
#endif

#ifdef TCL_MODULE
    { "Tk", 0, 0, 1024, 864, 0, 0,
      sp_Tk_Init, sp_Tk_NewViewport,
      sp_Tk_Close, sp_Tk_Clear,
      sp_Tk_DrawLine, sp_Tk_Arc, sp_Tk_Text,
      sp_Tk_DefineColor, sp_Tk_DefineLinestyle,
      sp_Tk_SetLinestyle, sp_Tk_SetColor, sp_Tk_Update,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, (disp_fn_Input_t *) NODEV,
      gen_DatatoScreen, },
#endif

    { "plot5", 0, 0, 1000, 1000, 0, 0,
      Plt5_Init, Plt5_NewViewport,
      Plt5_Close, Plt5_Clear,
      Plt5_DrawLine, Plt5_Arc, Plt5_Text,
      (disp_fn_DefineColor_t *) NODEV, (disp_fn_DefineLinestyle_t *) NODEV,
      Plt5_SetLinestyle, Plt5_SetColor, Plt5_Update,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, (disp_fn_Input_t *) NODEV,
      gen_DatatoScreen, },

    { "postscript", 0, 0, 1000, 1000, 0, 0,
      PS_Init, PS_NewViewport,
      PS_Close, PS_Clear,
      PS_DrawLine, PS_Arc, PS_Text,
      (disp_fn_DefineColor_t *) NODEV, (disp_fn_DefineLinestyle_t *) NODEV,
      PS_SetLinestyle, PS_SetColor, PS_Update, PS_Finalize,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, (disp_fn_Input_t *) NODEV,
      gen_DatatoScreen, },

    { "svg", 0, 0, 1000, 1000, 0, 0,
      SVG_Init, SVG_NewViewport,
      SVG_Close, SVG_Clear,
      SVG_DrawLine, SVG_Arc, SVG_Text,
      (disp_fn_DefineColor_t*)NODEV, (disp_fn_DefineLinestyle_t*)NODEV,
      SVG_SetLinestyle, SVG_SetColor, SVG_Update, SVG_Finalize,
      (disp_fn_Track_t*)NODEV, (disp_fn_MakeMenu_t*)NODEV, (disp_fn_MakeDialog_t*)NODEV, (disp_fn_Input_t*)NODEV,
      gen_DatatoScreen, },

    { "hpgl", 0, 0, 1000, 1000, 0, 0,
      GL_Init, GL_NewViewport,
      GL_Close, GL_Clear,
      GL_DrawLine, GL_Arc, GL_Text,
      (disp_fn_DefineColor_t *) NODEV, (disp_fn_DefineLinestyle_t *) NODEV,
      GL_SetLinestyle, GL_SetColor, GL_Update,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, (disp_fn_Input_t *) NODEV,
      gen_DatatoScreen, },

    { "BatchMode/PrinterOnly", 0, 0, 24, 80, 0, 0,
      (disp_fn_Init_t *) NODEV, (disp_fn_NewViewport_t *)  NODEV,
      (disp_fn_Close_t *) NOP, (disp_fn_Clear_t *) NODEV,
      (disp_fn_DrawLine_t *) NODEV, (disp_fn_Arc_t *) NODEV, (disp_fn_Text_t *) NODEV,
      (disp_fn_DefineColor_t *) NODEV, (disp_fn_DefineLinestyle_t *) NODEV,
      (disp_fn_SetLinestyle_t *) NODEV, (disp_fn_SetColor_t *) NODEV, (disp_fn_Update_t *) NOP,  (disp_fn_Finalize_t*) NOP,
      (disp_fn_Track_t *) NODEV, (disp_fn_MakeMenu_t *) NODEV, (disp_fn_MakeDialog_t *) NODEV, gen_Input,
      (disp_fn_DatatoScreen_t *) NODEV, },

};


DISPDEVICE *dispdev = device + NUMELEMS(device) - 1;


DISPDEVICE *
FindDev(char *name)
{
    size_t i;

    for (i = 0; i < NUMELEMS(device); i++)
        if (strcmp(name, device[i].name) == 0)
            return (device + i);

    sprintf(ErrorMessage, "Can't find device %s.", name);
    internalerror(ErrorMessage);

    return (device + 0);
}


void
DevInit(void)
{

#ifndef X_DISPLAY_MISSING
    char buf[128];
#endif

    /* note: do better determination */

    /*
      dumb tradition that got passed on from gi_interface
      to do compile time determination
    */

    dispdev = NULL;

#ifndef X_DISPLAY_MISSING
    /* determine display type */
    if (getenv("DISPLAY") || cp_getvar("display", CP_STRING, buf, sizeof(buf)))
        dispdev = FindDev("X11");
#endif


#ifdef HAS_WINGUI
    if (!dispdev)
        dispdev = FindDev("Windows");
#endif


#ifdef TCL_MODULE
    dispdev = FindDev("Tk");
#endif


    if (!dispdev) {

#if !defined(HAS_WINGUI) && !defined(TCL_MODULE) && !defined(SHARED_MODULE) && (defined(_MSC_VER) || defined(__MINGW32__))
        /* console application under MS Windows */
        fprintf
            (cp_err,
             "Warning: no graphics interface!\n"
             " You may use command 'gnuplot'\n"
             " if GnuPlot is installed.\n");
#elif !defined(X_DISPLAY_MISSING)
        externalerror
            ("no graphics interface;\n"
             " please check if X-server is running,\n"
             " or ngspice is compiled properly (see INSTALL)");
#endif

        dispdev = FindDev("error");

    } else if (dispdev->Init()) {
        fprintf(cp_err, "Warning: can't initialize display device for graphics.\n");
        dispdev = FindDev("error");
    }
}


/* NewViewport is responsible for filling in graph->viewport */
int
NewViewport(GRAPH *pgraph)
{
    return dispdev->NewViewport (pgraph);
}


void
DevClose(void)
{
    dispdev->Close();
}


void
DevClear(void)
{
    dispdev->Clear();
}


void
DevDrawLine(int x1, int y1, int x2, int y2, bool isgrid)
{
    dispdev->DrawLine (x1, y1, x2, y2, isgrid);
}


void
DevDrawArc(int x0, int y0, int radius, double theta, double delta_theta, bool isgrid)
{
    dispdev->DrawArc (x0, y0, radius, theta, delta_theta, isgrid);
}


void DevDrawText(const char *text, int x, int y, int angle)
{
    dispdev->DrawText(text, x, y, angle);
}


void
DefineColor(int colorid, double red, double green, double blue)
{
    dispdev->DefineColor (colorid, red, green, blue);
}


void
DefineLinestyle(int linestyleid, int mask)
{
    dispdev->DefineLinestyle (linestyleid, mask);
}


void
SetLinestyle(int linestyleid)
{
    dispdev->SetLinestyle (linestyleid);
}


void
SetColor(int colorid)
{
    dispdev->SetColor(colorid);
}


void
DevUpdate(void)
{
    if (dispdev)
        dispdev->Update();
}

void
DevFinalize(void)
{
    if (dispdev)
        dispdev->Finalize();
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
       since the grid routines use DevDrawLine w/o calling this routine */

    if ((graph->grid.gridtype == GRID_LOGLOG) ||
        (graph->grid.gridtype == GRID_YLOG))
    {
        low  = mylog10(graph->datawindow.ymin);
        high = mylog10(graph->datawindow.ymax);
        *screeny = (int)((mylog10(y) - low) / (high - low) * graph->viewport.height
                         + 0.5 + graph->viewportyoff);
    } else {
        *screeny = (int)(((y - graph->datawindow.ymin) / graph->aspectratioy)
                         + 0.5 + graph->viewportyoff);
    }

    if ((graph->grid.gridtype == GRID_LOGLOG) ||
        (graph->grid.gridtype == GRID_XLOG))
    {
        low  = mylog10(graph->datawindow.xmin);
        high = mylog10(graph->datawindow.xmax);
        *screenx = (int)((mylog10(x) - low) / (high - low) * graph->viewport.width
                         + 0.5 + graph ->viewportxoff);
    } else {
        *screenx = (int)((x - graph->datawindow.xmin) / graph->aspectratiox
                         + 0.5 + graph ->viewportxoff);
    }
}


void
DatatoScreen(GRAPH *graph, double x, double y, int *screenx, int *screeny)
{
    dispdev->DatatoScreen (graph, x, y, screenx, screeny);
}


void
Input(REQUEST *request, RESPONSE *response)
{
    dispdev->Input (request, response);
}


static int
gen_Input(REQUEST *request, RESPONSE *response)
{
    switch (request->option) {
    case char_option:
        if (response)
            response->option = request->option;
        break;
    default:
        /* just ignore, since we don't want a million error messages */
        if (response)
            response->option = error_option;
        break;
    }

    return (0);
}


/* no operation, do nothing */
static int
nop(void)
{
    return (1);  /* so NewViewport will fail */
}


static int
nodev(void)
{
    sprintf(ErrorMessage,
            "This operation is not defined for display type %s.",
            dispdev->name);
    internalerror(ErrorMessage);
    return (1);
}


void
SaveText(GRAPH *graph, char *text, int x, int y)
{
    struct _keyed *keyed = TMALLOC(struct _keyed, 1);

    if (!graph->keyed) {
        graph->keyed = keyed;
    } else {
        keyed->next = graph->keyed;
        graph->keyed = keyed;
    }

    keyed->text = TMALLOC(char, strlen(text) + 1);
    strcpy(keyed->text, text);

    keyed->x = x;
    keyed->y = y;

    keyed->colorindex = graph->currentcolor;
}


/* if given name of a hardcopy device, finds it and switches devices
   if given NULL, switches back */
int
DevSwitch(char *devname)
{
    static DISPDEVICE *lastdev = NULL;

    if (devname) {

        if (lastdev) {
            internalerror("DevSwitch w/o changing back");
            return (1);
        }

        lastdev = dispdev;
        dispdev = FindDev(devname);

        if (!strcmp(dispdev->name, "error")) {
            internalerror("no hardcopy device");
            /* undo */
            dispdev = lastdev;
            lastdev = NULL;
            return (1);
        }

        dispdev->Init();

    } else {

        if (dispdev)
            dispdev->Close();
        dispdev = lastdev;
        lastdev = NULL;

    }

    return (0);
}
