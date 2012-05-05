/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
    Manage graph data structure.
*/

#include "ngspice/ngspice.h"
#include "ngspice/graph.h"
#include "ngspice/ftedebug.h"
#include "ngspice/fteext.h"

#include "ngspice/ftedbgra.h"
#include "graphdb.h"
#include "../breakp2.h"
#include "../display.h"


/* invariant:  currentgraph contains the current graph */
GRAPH *currentgraph;

/*
 *  We use a linked list rather than a circular one because we
 *    expect few links per list and we don't need to add at the
 *    end of a list (we can add at the beginning).
 */

/* linked list of graphs */
typedef struct listgraph {
      /* we use GRAPH here instead of a pointer to save a tmalloc */
    GRAPH graph;
    struct listgraph *next;
} LISTGRAPH;
#define NEWLISTGRAPH TMALLOC(LISTGRAPH, 1)

#define NUMGBUCKETS 16

typedef struct gbucket {
    LISTGRAPH *list;
} GBUCKET;

static GBUCKET GBucket[NUMGBUCKETS];

/* note: Zero is not a valid id.  This is used in plot() in graf.c. */
static int RunningId = 1;

/* initialize graph structure */
#define SETGRAPH(pgraph, id) (pgraph)->graphid = (id); \
				(pgraph)->degree = 1; \
				(pgraph)->linestyle = -1

/* returns NULL on error */

GRAPH *NewGraph(void)
{

    GRAPH *pgraph;
    LISTGRAPH *list;
    int BucketId = RunningId % NUMGBUCKETS;

    if ((list = NEWLISTGRAPH) == NULL) {
      internalerror("can't allocate a listgraph");
      return(NULL);
    }

    pgraph = &list->graph;
    SETGRAPH(pgraph, RunningId);

    if (!GBucket[BucketId].list) {
      GBucket[BucketId].list = list;
    } else {
      /* insert at front of current list */
      list->next = GBucket[BucketId].list;
      GBucket[BucketId].list = list;
    }

    RunningId++ ;

    return(pgraph);

}

/* Given graph id, return graph */
GRAPH *FindGraph(int id)
{

    LISTGRAPH *list;

    for (list = GBucket[id % NUMGBUCKETS].list;
	    list && list->graph.graphid != id;
	    list = list->next)

	;

    if (list)
	return(&list->graph);
    else
	return(NULL);

}

GRAPH *CopyGraph(GRAPH *graph)
{

    GRAPH *ret;
    struct _keyed *k;
    struct dveclist *link, *newlink;

    ret = NewGraph();
    bcopy(graph, ret, sizeof(GRAPH)); /* va: compatible pointer types */

    ret->graphid = RunningId - 1;   /* restore id */

    /* copy keyed */
    for (ret->keyed = NULL, k = graph->keyed; k; k = k->next) {
      SaveText(ret, k->text, k->x, k->y);
    }

    /* copy dvecs */
    ret->plotdata = NULL;
    for (link = graph->plotdata; link; link = link->next) {
      newlink = TMALLOC(struct dveclist, 1);
      newlink->next = ret->plotdata;
      newlink->vector = vec_copy(link->vector);
      /* vec_copy doesn't set v_color or v_linestyle */
      newlink->vector->v_color = link->vector->v_color;
      newlink->vector->v_linestyle = link->vector->v_linestyle;
      newlink->vector->v_flags |= VF_PERMANENT;
      ret->plotdata = newlink;
    }

    ret->commandline = copy(graph->commandline);
    ret->plotname = copy(graph->plotname);

    return(ret);

}

int
DestroyGraph(int id)
{

    LISTGRAPH *list, *lastlist;
    struct _keyed *k, *nextk;
    struct dveclist *d, *nextd;
    struct dbcomm *db;

    list = GBucket[id % NUMGBUCKETS].list;
    lastlist = NULL;
    while (list) {
      if (list->graph.graphid == id) {  /* found it */

	/* Fix the iplot/trace dbs list */
	for (db = dbs; db && db->db_graphid != id; db = db->db_next)
	    ;

	if (db && (db->db_type == DB_IPLOT
		|| db->db_type == DB_IPLOTALL)) {
	    db->db_type = DB_DEADIPLOT;
	    /* Delete this later */
	    return(0);
	}

	/* adjust bucket pointers */
	if (lastlist) {
	  lastlist->next = list->next;
        } else {
          GBucket[id % NUMGBUCKETS].list = list->next;
        }

        /* run through and de-allocate dynamically allocated keyed list */
        k=list->graph.keyed;
        while (k) {
          nextk = k->next;
          tfree(k->text);
          tfree(k);
          k = nextk;
        }

        /* de-allocate dveclist */
        d = list->graph.plotdata;
        while (d) {
          nextd = d->next;
          tfree(d->vector->v_name);
          if (isreal(d->vector)) {
                tfree(d->vector->v_realdata);
              } else {
                tfree(d->vector->v_compdata);
              }
          tfree(d->vector);
          tfree(d);
          d = nextd;
        }

        tfree(list->graph.commandline);
        tfree(list->graph.plotname);

        /* If device dependent space allocated, free it. */
        if (list->graph.devdep)
          tfree(list->graph.devdep);
        tfree(list);

        return(1);
      }
      lastlist = list;
      list = list->next;
    }

    internalerror("tried to destroy non-existent graph");
    return (0);

}

/* free up all dynamically allocated data structures */
void
FreeGraphs(void)
{

    GBUCKET *gbucket;
    LISTGRAPH *list, *deadl;

    for (gbucket = GBucket; gbucket < &GBucket[NUMGBUCKETS]; gbucket++) {
      list = gbucket->list;
      while (list) {
        deadl = list;
        list = list->next;
        tfree(deadl);
      }
    }

}

void
SetGraphContext(int graphid)
{

    currentgraph = FindGraph(graphid);

}

typedef struct gcstack {
    GRAPH *pgraph;
    struct gcstack *next;
} GCSTACK;
GCSTACK *gcstacktop;
#define NEWGCSTACK TMALLOC(GCSTACK, 1)

/* note: This Push and Pop has tricky semantics.
    Push(graph) will push the currentgraph onto the stack
        and set currentgraph to graph.
    Pop() simply sets currentgraph to the top of the stack and pops stack.
*/
void
PushGraphContext(GRAPH *graph)
{

    GCSTACK *gcstack = NEWGCSTACK;

    if (!gcstacktop) {
      gcstacktop = gcstack;
    } else {
      gcstack->next = gcstacktop;
      gcstacktop = gcstack;
    }
    gcstacktop->pgraph = currentgraph;
    currentgraph = graph;

}

void
PopGraphContext(void)
{

    GCSTACK *dead;

    currentgraph = gcstacktop->pgraph;
    dead = gcstacktop;
    gcstacktop = gcstacktop->next;
    tfree(dead);
}
