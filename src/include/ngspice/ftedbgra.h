/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Jeffrey M. Hsu
**********/

/*
    External definitions for the graph database module.
*/

extern GRAPH *currentgraph;

extern GRAPH *NewGraph(void);

extern GRAPH *FindGraph(int id);

extern GRAPH *CopyGraph(GRAPH *graph);

extern void RemoveWindow(GRAPH*);
