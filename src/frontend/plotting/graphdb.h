/*************
 * Header file for graphdb.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_GRAPHDB_H
#define ngspice_GRAPHDB_H

int DestroyGraph(int id);
void FreeGraphs(void);
void SetGraphContext(int graphid);
void PushGraphContext(GRAPH *graph);
void PopGraphContext(void);

/* Global variable to indicate that at least one graph exits. Ugly but fast. */

extern bool Have_graph;
#endif
