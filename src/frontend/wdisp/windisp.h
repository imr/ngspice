/*************
* Header file for windisp.c
* $Id$
************/

#ifndef WINDISP_H
#define WINDISP_H

extern int WIN_Init();
extern int WIN_NewViewport();
extern int WIN_Close();
extern int WIN_Clear();
extern int WIN_DrawLine();
extern int WIN_Arc();
extern int WIN_Text();
extern int WIN_DefineColor();
extern int WIN_DefineLinestyle();
extern int WIN_SetLinestyle();
extern int WIN_SetColor();
extern int WIN_Update();
extern int WIN_DiagramReady();

#endif /* WINDISP_H */
