/*************
 * Header file for clip.c
 * 1999 E. Rouat
 ************/

#ifndef CLIP_H_INCLUDED
#define CLIP_H_INCLUDED


bool clip_line(int *pX1, int *pY1, int *pX2, int *pY2, int l, int b, int r, int t);
bool clip_to_circle(int *x1, int *y1, int *x2, int *y2, int cx, int cy, int rad);

#endif
