/*************
* Header file for winprint.c
* $Id$
************/

#ifndef WINPRINT_H
#define WINPRINT_H

extern int WPRINT_Init();
extern int WPRINT_NewViewport();
extern int WPRINT_Close();
extern int WPRINT_Clear();
extern int WPRINT_DrawLine();
extern int WPRINT_Arc();
extern int WPRINT_Text();
extern int WPRINT_DefineColor();
extern int WPRINT_DefineLinestyle();
extern int WPRINT_SetLinestyle();
extern int WPRINT_SetColor();
extern int WPRINT_Update();
extern int WPRINT_DiagramReady();

#endif /* WINPRINT_H */
