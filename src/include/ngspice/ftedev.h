/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Jeffrey M. Hsu
**********/

/*
    The display device structure.
*/

#ifndef ngspice_FTEDEV_H
#define ngspice_FTEDEV_H

#include "ngspice/bool.h"
#include "ngspice/typedefs.h"

struct graph;
struct request;
struct response;

typedef int  disp_fn_Init_t (void);
typedef int  disp_fn_NewViewport_t (struct graph *);
typedef int  disp_fn_Close_t (void);
typedef int  disp_fn_Clear_t (void);
typedef int  disp_fn_DrawLine_t (int x1, int y1, int x2, int y2, bool isgrid);
typedef int  disp_fn_Arc_t (int x0, int y0, int radius, double theta, double delta_theta, bool isgrid);
typedef int  disp_fn_Text_t (const char *text, int x, int y, int angle);
typedef int  disp_fn_DefineColor_t (int colorid, double red, double green, double blue);
typedef int  disp_fn_DefineLinestyle_t (int linestyleid, int mask);
typedef int  disp_fn_SetLinestyle_t (int linestyleid);
typedef int  disp_fn_SetColor_t (int colorid);
typedef int  disp_fn_Update_t (void);
typedef int  disp_fn_Finalize_t (void);
typedef int  disp_fn_Track_t (void);
typedef int  disp_fn_MakeMenu_t (void);
typedef int  disp_fn_MakeDialog_t (void);
typedef int  disp_fn_Input_t (struct request *request, struct response *response);
typedef void disp_fn_DatatoScreen_t (struct graph *graph, double x, double y, int *screenx, int *screeny);

typedef struct {
    char *name;
    int minx, miny;
    int width, height;      /* in screen coordinate system */
    int numlinestyles, numcolors;   /* number supported */

    disp_fn_Init_t             *Init;
    disp_fn_NewViewport_t      *NewViewport;
    disp_fn_Close_t            *Close;
    disp_fn_Clear_t            *Clear;
    disp_fn_DrawLine_t         *DrawLine;
    disp_fn_Arc_t              *DrawArc;
    disp_fn_Text_t             *DrawText;
    disp_fn_DefineColor_t      *DefineColor;
    disp_fn_DefineLinestyle_t  *DefineLinestyle;
    disp_fn_SetLinestyle_t     *SetLinestyle;
    disp_fn_SetColor_t         *SetColor;
    disp_fn_Update_t           *Update;
    disp_fn_Finalize_t         *Finalize;
    disp_fn_Track_t            *Track;
    disp_fn_MakeMenu_t         *MakeMenu;
    disp_fn_MakeDialog_t       *MakeDialog;
    disp_fn_Input_t            *Input;
    disp_fn_DatatoScreen_t     *DatatoScreen;
} DISPDEVICE;

extern DISPDEVICE *dispdev;


#endif
