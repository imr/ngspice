/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/inpmacs.h"

#include "inppas2.h"
#include "inpxx.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include function prototypes */
#include "ngspice/mifproto.h"
/* gtri - end - wbk - 11/9/90 */
#endif


/* uncomment to trace in this file */
/*#define TRACE*/

/* pass 2 - Scan through the lines.  ".model" cards have processed in
 *  pass1 and are ignored here.  */

void INPpas2(CKTcircuit *ckt, struct card *data, INPtables * tab, TSKtask *task)
{

    struct card *current;
    char c;
    char *groundname = "0";
    char *gname;
    CKTnode *gnode;
    int error;			/* used by the macros defined above */
#ifdef HAS_PROGREP
    int linecount = 0, actcount = 0;
#endif

#ifdef TRACE
    /* SDB debug statement */
    printf("Entered INPpas2 . . . .\n");
#endif

#ifdef XSPICE
    if (!ckt->CKTadevFlag) ckt->CKTadevFlag = 0;
#endif

    error = INPgetTok(&groundname, &gname, 1);
    if (error)
	data->error =
	    INPerrCat(data->error,
		      INPmkTemp
		      ("can't read internal ground node name!\n"));

    error = INPgndInsert(ckt, &gname, tab, &gnode);
    if (error && error != E_EXISTS)
	data->error =
	    INPerrCat(data->error,
		      INPmkTemp
		      ("can't insert internal ground node in symbol table!\n"));
    
#ifdef TRACE 
    printf("Examining this deck:\n");
    for (current = data; current != NULL; current = current->nextcard) {
      printf("%s\n", current->line);
    }
    printf("\n");
#endif


#ifdef HAS_PROGREP
    for (current = data; current != NULL; current = current->nextcard)
        linecount++;
#endif

    for (current = data; current != NULL; current = current->nextcard) {

#ifdef TRACE
	/* SDB debug statement */
	printf("In INPpas2, examining card %s . . .\n", current->line);
#endif

#ifdef HAS_PROGREP
   if (linecount > 0) {     
        SetAnalyse( "Parse", (int) (1000.*actcount/linecount));
        actcount++;
   }
#endif

	c = *(current->line);
	if(islower_c(c))
	    c = toupper_c(c);

	switch (c) {

	case ' ':
	    /* blank line (space leading) */
	case '\t':
	    /* blank line (tab leading) */
	    break;

#ifdef XSPICE
	    /* gtri - add - wbk - 10/23/90 - add case for 'A' devices */

	case 'A':   /* Aname <cm connections> <mname> */
	    MIF_INP2A(ckt, tab, current);
	    ckt->CKTadevFlag = 1; /* an 'A' device is requested */
	    break;

	  /* gtri - end - wbk - 10/23/90 */
#endif

	case 'R':
	    /* Rname <node> <node> [<val>][<mname>][w=<val>][l=<val>] */
	    INP2R(ckt, tab, current);
	    break;

	case 'C':
	    /* Cname <node> <node> <val> [IC=<val>] */
	    INP2C(ckt, tab, current);
	    break;

	case 'L':
	    /* Lname <node> <node> <val> [IC=<val>] */
	    INP2L(ckt, tab, current);
	    break;

	case 'G':
	    /* Gname <node> <node> <node> <node> <val> */
	    INP2G(ckt, tab, current);
	    break;

	case 'E':
	    /* Ename <node> <node> <node> <node> <val> */
	    INP2E(ckt, tab, current);
	    break;

	case 'F':
	    /* Fname <node> <node> <vname> <val> */
	    INP2F(ckt, tab, current);
	    break;

	case 'H':
	    /* Hname <node> <node> <vname> <val> */
	    INP2H(ckt, tab, current);
	    break;

	case 'D':
	    /* Dname <node> <node> <model> [<val>] [OFF] [IC=<val>] */
	    INP2D(ckt, tab, current);
	    break;

	case 'J':
	    /* Jname <node> <node> <node> <model> [<val>] [OFF]
               [IC=<val>,<val>] */
	    INP2J(ckt, tab, current);
	    break;

	case 'Z':
	    /* Zname <node> <node> <node> <model> [<val>] [OFF] 
	       [IC=<val>,<val>] */
	    INP2Z(ckt, tab, current);
	    break;

	case 'M':
	    /* Mname <node> <node> <node> <node> <model> [L=<val>]
	       [W=<val>] [AD=<val>] [AS=<val>] [PD=<val>]
	       [PS=<val>] [NRD=<val>] [NRS=<val>] [OFF] 
	       [IC=<val>,<val>,<val>] */
	    INP2M(ckt, tab, current);
	    break;
#ifdef  OSDI
	case 'N':
	    /* Nname [<node>...]  [<mname>] */
	    INP2N(ckt, tab, current);
	    break;
#endif
	case 'O':
	    /* Oname <node> <node> <node> <node> <model>
	       [IC=<val>,<val>,<val>,<val>] */
	    INP2O(ckt, tab, current);
	    break;

	case 'V':
	    /* Vname <node> <node> [ [DC] <val>] [AC [<val> [<val> ] ] ]
	       [<tran function>] */
	    INP2V(ckt, tab, current);
	    break;

	case 'I':
	    /* Iname <node> <node> [ [DC] <val>] [AC [<val> [<val> ] ] ]
	       [<tran function>] */
	    INP2I(ckt, tab, current);
	    break;

	case 'Q':
	    /* Qname <node> <node> <node> [<node>] <model> [<val>] [OFF]
	       [IC=<val>,<val>] */
	    INP2Q(ckt, tab, current, gnode);
	    break;

	case 'T':
	    /* Tname <node> <node> <node> <node> [TD=<val>] 
	       [F=<val> [NL=<val>]][IC=<val>,<val>,<val>,<val>] */
	    INP2T(ckt, tab, current);
	    break;

	case 'S':
	    /* Sname <node> <node> <node> <node> [<modname>] [IC] */
	    INP2S(ckt, tab, current);
	    break;

	case 'W':
	    /* Wname <node> <node> <vctrl> [<modname>] [IC] */
	    /* CURRENT CONTROLLED SWITCH */
	    INP2W(ckt, tab, current);
	    break;

	case 'U':
	    /* Uname <node> <node> <model> [l=<val>] [n=<val>] */
	    INP2U(ckt, tab, current);
	    break;

	/* Kspice addition - saj */
	case 'P': 
	    /* Pname <node> <node> ... <gnd> <node> <node> ... <gnd> [<modname>] */
	    /* R=<vector> L=<matrix> G=<vector> C=<matrix> len=<val> */
	    INP2P(ckt, tab, current);
	    break;
	case 'Y':   
	    /* Yname <node> <node> R=<val> L=<val> G=<val> C=<val> len=<val> */
	    INP2Y(ckt, tab, current);
	    break;
	/* end Kspice */

	case 'K':
	    /* Kname Lname Lname <val> */
	    INP2K(ckt, tab, current);
	    break;

	case '*': case '$':
	    /* *<anything> - a comment - ignore */
	    break;

	case 'B':
	    /* Bname <node> <node> [V=expr] [I=expr] */
	    /* Arbitrary source. */
	    INP2B(ckt, tab, current);
	    break;

	case '.':   /* .<something> Many possibilities */
	    if (INP2dot(ckt,tab,current,task,gnode))
	    return;
	    break;

	case '\0':
	    break;

	default:
	    /* the un-implemented device */
	    LITERR(" unknown device type - error \n");
	    break;
	}
  }

    return;
}
