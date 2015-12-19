/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 *	A variant on the "zeroin" method.  This is a bit convoluted.
 */

#include "ngspice/ngspice.h"
#include "ngspice/pzdefs.h"
#include "ngspice/complex.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"


#ifdef PZDEBUG
#  ifndef notdef
#    define DEBUG(N)	if (Debug >= (unsigned) (N))
static unsigned int	Debug = 3;
#  else
#    define DEBUG(N)	if (0)
#  endif
#endif

/* static function definitions */

static int CKTpzStrat(PZtrial **set);
static int CKTpzStep(int strat, PZtrial **set);
static int CKTpzRunTrial(CKTcircuit *ckt, PZtrial **new_trialp, PZtrial **set);
static int CKTpzVerify(PZtrial **set, PZtrial *new_trial);
static void clear_trials(int mode);
static void check_flat(PZtrial *a, PZtrial *b);
void CKTpzUpdateSet(PZtrial **set, PZtrial *new);
void zaddeq(double *a, int *amag, double x, int xmag, double y, int ymag);
void CKTpzReset(PZtrial **set);


#ifdef PZDEBUG
static void show_trial(PZtrial *new_trial, char x);
#endif

#define NITER_LIM	200

#define	SHIFT_LEFT	2
#define	SHIFT_RIGHT	3
#define	SKIP_LEFT	4
#define	SKIP_RIGHT	5
#define	INIT		6

#define	GUESS		7
#define	SPLIT_LEFT	8
#define	SPLIT_RIGHT	9

#define	MULLER		10
#define	SYM		11
#define	SYM2		12
#define	COMPLEX_INIT	13
#define	COMPLEX_GUESS	14
#define	QUIT		15

#define	NEAR_LEFT	4
#define	MID_LEFT	5
#define	FAR_LEFT	6
#define	NEAR_RIGHT	7
#define	FAR_RIGHT	8
#define	MID_RIGHT	9

#ifdef PZDEBUG
static char *snames[ ] = {
	"none",
	"none",
	"shift left",
	"shift right",
	"skip left",
	"skip right",
	"init",
	"guess",
	"split left",
	"split right",
	"Muller",
	"sym 1",
	"sym 2",
	"complex_init",
	"complex_guess",
	"quit",
	"none"
	};
#endif

#define sgn(X)	((X) < 0 ? -1 : (X) == 0 ? 0 : 1)

#define	ISAROOT		2
#define	ISAREPEAT	4
#define	ISANABERRATION	8
#define	ISAMINIMA	16

extern double	NIpzK;
extern int	NIpzK_mag;

int	CKTpzTrapped;

static int	NZeros, NFlat, Max_Zeros;
static PZtrial	*ZeroTrial, *Trials;
static int	Seq_Num;
static double	Guess_Param;
static double	High_Guess, Low_Guess;
static int	Last_Move, Consec_Moves;
static int	NIter, NTrials;
static int	Aberr_Num;

int PZeval(int strat, PZtrial **set, PZtrial **new_trial_p);
static PZtrial *pzseek(PZtrial *t, int dir);
static int alter(PZtrial *new, PZtrial *nearto, double abstol, double reltol);

int
CKTpzFindZeros(CKTcircuit *ckt, PZtrial **rootinfo, int *rootcount)
{
    PZtrial *new_trial;
    PZtrial *neighborhood[3];
    int    strat;
    int    error;

    NIpzK = 0.0;
    NIpzK_mag = 0;
    High_Guess = -1.0;
    Low_Guess = 1.0;
    ZeroTrial = NULL;
    Trials = NULL;
    NZeros = 0;
    NFlat = 0;
    Max_Zeros = SMPmatSize(ckt->CKTmatrix);
    NIter = 0;
    error = OK;
    CKTpzTrapped = 0;
    Aberr_Num = 0;
    NTrials = 0;
    ckt->CKTniState |= NIPZSHOULDREORDER; /* Initial for LU fill-ins */

    Seq_Num = 1;

    CKTpzReset(neighborhood);

    do {

	while ((strat = CKTpzStrat(neighborhood)) < GUESS && !CKTpzTrapped)
	    if (!CKTpzStep(strat, neighborhood)) {
		strat = GUESS;
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr, "\t\tGuess\n");
#endif
		break;
	    }

	NIter += 1;
	
	/* Evaluate current strategy */
	error = PZeval(strat, neighborhood, &new_trial);
	if (error != OK)
	    return error;

	error = CKTpzRunTrial(ckt, &new_trial, neighborhood);
	if (error != OK)
	    return error;

	if (new_trial->flags & ISAROOT) {
	    if (CKTpzVerify(neighborhood, new_trial)) {
		NIter = 0;
		CKTpzReset(neighborhood);
	    } else
		/* XXX Verify fails ?!? */
		CKTpzUpdateSet(neighborhood, new_trial);
	} else if (new_trial->flags & ISANABERRATION) {
	    CKTpzReset(neighborhood);
	    Aberr_Num += 1;
	    tfree(new_trial);
	} else if (new_trial->flags & ISAMINIMA) {
	    neighborhood[0] = NULL;
	    neighborhood[1] = new_trial;
	    neighborhood[2] = NULL;
	} else {
	    CKTpzUpdateSet(neighborhood, new_trial);	/* Replace a value */
	}

	if (SPfrontEnd->IFpauseTest()) {
	    SPfrontEnd->IFerrorf (ERR_WARNING, "Pole-Zero analysis interrupted; %d trials, %d roots\n", Seq_Num, NZeros);
	    error = E_PAUSE;
	    break;
	}
    } while (High_Guess - Low_Guess < 1e40
	    && NZeros < Max_Zeros
	    && NIter < NITER_LIM && Aberr_Num < 3
	    && High_Guess - Low_Guess < 1e35	/* XXX Should use mach const */
	    && (!neighborhood[0] || !neighborhood[2] || CKTpzTrapped
	    || neighborhood[2]->s.real - neighborhood[0]->s.real < 1e22));
	    /* XXX ZZZ */

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr,
	"Finished: NFlat %d, NZeros: %d, NTrials %d, Guess %g to %g, aber %d\n",
	NFlat, NZeros, NTrials, Low_Guess, High_Guess, Aberr_Num);
#endif

    if (NZeros >= Seq_Num - 1) {
	/* Short */
	clear_trials(ISAROOT);
	*rootinfo = NULL;
	*rootcount = 0;
	MERROR(E_SHORT, "The input signal is shorted on the way to the output");
    } else
	clear_trials(0);

    *rootinfo = Trials;
    *rootcount = NZeros;

    if (Aberr_Num > 2) {
	SPfrontEnd->IFerrorf (ERR_WARNING, "Pole-zero converging to numerical aberrations; giving up after %d trials", Seq_Num);
    }

    if (NIter >= NITER_LIM) {
	SPfrontEnd->IFerrorf (ERR_WARNING, "Pole-zero iteration limit reached; giving up after %d trials", Seq_Num);
    }

    return error;
}

/* PZeval: evaluate an estimation function (given by 'strat') for the next
    guess (returned in a PZtrial) */

/* XXX ZZZ */
int
PZeval(int strat, PZtrial **set, PZtrial **new_trial_p)
{
    int		error;
    PZtrial	*new_trial;

    new_trial = TMALLOC(PZtrial, 1);
    new_trial->multiplicity = 0;
    new_trial->count = 0;
    new_trial->seq_num = Seq_Num++;

    switch (strat) {
    case GUESS:
	if (High_Guess < Low_Guess)
	    Guess_Param = 0.0;
	else if (Guess_Param > 0.0) {
	    if (High_Guess > 0.0)
		Guess_Param = High_Guess * 10.0;
	    else
		Guess_Param = 1.0;
	} else {
	    if (Low_Guess < 0.0)
		Guess_Param = Low_Guess * 10.0;
	    else
		Guess_Param = -1.0;
	}
	if (High_Guess < Guess_Param)
	    High_Guess = Guess_Param;
	if (Low_Guess > Guess_Param)
	    Low_Guess = Guess_Param;
	new_trial->s.real = Guess_Param;
	if (set[1])
	    new_trial->s.imag = set[1]->s.imag;
	else
	    new_trial->s.imag = 0.0;
	error = OK;
	break;

    case SYM:
    case SYM2:
	error = NIpzSym(set, new_trial);

	if (CKTpzTrapped == 1) {
	    if (new_trial->s.real < set[0]->s.real
		    || new_trial->s.real > set[1]->s.real) {
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr,
		    "FIXED UP BAD Strat: %s (%d) was (%.15g,%.15g)\n",
		    snames[strat], CKTpzTrapped,
		    new_trial->s.real, new_trial->s.imag);
#endif
		new_trial->s.real = (set[0]->s.real + set[1]->s.real) / 2.0;
	    }
	} else if (CKTpzTrapped == 2) {
	    if (new_trial->s.real < set[1]->s.real
		    || new_trial->s.real > set[2]->s.real) {
#ifdef PZDEBUG
		DEBUG(1) fprintf(stderr,
		    "FIXED UP BAD Strat: %s (%d) was (%.15g,%.15g)\n",
		    snames[strat], CKTpzTrapped,
		    new_trial->s.real, new_trial->s.imag);
#endif
		new_trial->s.real = (set[1]->s.real + set[2]->s.real) / 2.0;
	    }
	} else if (CKTpzTrapped == 3) {
	    if (new_trial->s.real <= set[0]->s.real
		|| (new_trial->s.real == set[1]->s.real
		&& new_trial->s.imag == set[1]->s.imag)
		|| new_trial->s.real >= set[2]->s.real) {
#ifdef PZDEBUG
		DEBUG(1)
		    fprintf(stderr,
			"FIXED UP BAD Strat: %s (%d), was (%.15g %.15g)\n",
			snames[strat], CKTpzTrapped,
			new_trial->s.real, new_trial->s.imag);
#endif
		new_trial->s.real = (set[0]->s.real + set[2]->s.real) / 2.0;
		if (new_trial->s.real == set[1]->s.real) {
#ifdef PZDEBUG
		    DEBUG(1)
			fprintf(stderr, "Still off!");
#endif
		    if (Last_Move == MID_LEFT || Last_Move == NEAR_RIGHT)
			new_trial->s.real = (set[0]->s.real + set[1]->s.real)
			    / 2.0;
		    else
			new_trial->s.real = (set[1]->s.real + set[2]->s.real)
			    / 2.0;
		}
	    }
	}

	break;

    case COMPLEX_INIT:
	/* Not automatic */
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "\tPZ minima at: %-30g %d\n",
	    NIpzK, NIpzK_mag);
#endif

	new_trial->s.real = set[1]->s.real;

	/* NIpzK is a good idea, but the value gets trashed 
	 * due to the numerics when zooming in on a minima.
	 * The key is to know when to stop taking new values for NIpzK
	 * (which I don't).  For now I take the first value indicated
	 * by the NIpzSym2 routine.  A "hack".
	 */

	if (NIpzK != 0.0 && NIpzK_mag > -10) {
	    while (NIpzK_mag > 0) {
		NIpzK *= 2.0;
		NIpzK_mag -= 1;
	    }
	    while (NIpzK_mag < 0) {
		NIpzK /= 2.0;
		NIpzK_mag += 1;
	    }
	    new_trial->s.imag = NIpzK;
	} else
	    new_trial->s.imag = 10000.0;

	/*
	 * Reset NIpzK so the same value doesn't get used again.
	 */

	NIpzK = 0.0;
	NIpzK_mag = 0;
	error = OK;
	break;

    case COMPLEX_GUESS:
	if (!set[2]) {
	    new_trial->s.real = set[0]->s.real;
	    new_trial->s.imag = 1.0e8;
	} else {
	    new_trial->s.real = set[0]->s.real;
	    new_trial->s.imag = 1.0e12;
	}
	error = OK;
	break;

    case MULLER:
	error = NIpzMuller(set, new_trial);
	break;

    case SPLIT_LEFT:
	new_trial->s.real = (set[0]->s.real + 2 * set[1]->s.real) / 3.0;
	error = OK;
	break;

    case SPLIT_RIGHT:
	new_trial->s.real = (set[2]->s.real + 2 * set[1]->s.real) / 3.0;
	error = OK;
	break;

    default:
	MERROR(E_PANIC, "Step type unknown");
	break;
    }

    *new_trial_p = new_trial;
    return error;
}

/* CKTpzStrat: given three points, determine a good direction or method for
    guessing the next zero */

/* XXX ZZZ what is a strategy for complex hunting? */
int CKTpzStrat(PZtrial **set)
{
    int	suggestion;
    double a, b;
    int	a_mag, b_mag;
    double k1, k2;
    int	new_trap;

    new_trap = 0;

    if (set[1] && (set[1]->flags & ISAMINIMA)) {
	suggestion = COMPLEX_INIT;
    } else if (set[0] && set[0]->s.imag != 0.0) {
	if (!set[1] || !set[2])
	    suggestion = COMPLEX_GUESS;
	else
	    suggestion = MULLER;
    } else if (!set[0] || !set[1] || !set[2]) {
	suggestion = INIT;
    } else {
	if (sgn(set[0]->f_def.real) != sgn(set[1]->f_def.real)) {
	    /* Zero crossing between s[0] and s[1] */
	    new_trap = 1;
	    suggestion = SYM2;
	} else if (sgn(set[1]->f_def.real) != sgn(set[2]->f_def.real)) {
	    /* Zero crossing between s[1] and s[2] */
	    new_trap = 2;
	    suggestion = SYM2;
	} else {

	    zaddeq(&a, &a_mag, set[1]->f_def.real, set[1]->mag_def,
		-set[0]->f_def.real, set[0]->mag_def);
	    zaddeq(&b, &b_mag, set[2]->f_def.real, set[2]->mag_def,
		-set[1]->f_def.real, set[1]->mag_def);

	    if (!CKTpzTrapped) {

		k1 = set[1]->s.real - set[0]->s.real;
		k2 = set[2]->s.real - set[1]->s.real;
		if (a_mag + 10 < set[0]->mag_def
			&& a_mag + 10 < set[1]->mag_def
			&& b_mag + 10 < set[1]->mag_def
			&& b_mag + 10 < set[2]->mag_def) {
		    if (k1 > k2)
			suggestion = SKIP_RIGHT;
		    else
			suggestion = SKIP_LEFT;
		} else if (sgn(a) != -sgn(b)) {
		    if (a == 0.0)
			suggestion = SKIP_LEFT;
		    else if (b == 0.0)
			suggestion = SKIP_RIGHT;
		    else if (sgn(a) == sgn(set[1]->f_def.real))
			suggestion = SHIFT_LEFT;
		    else
			suggestion = SHIFT_RIGHT;
		} else if (sgn(a) == -sgn(set[1]->f_def.real)) {
		    new_trap = 3;
		    /*  minima in magnitude above the x axis */
		    /* Search for exact mag. minima, look for complex pair */
		    suggestion = SYM;
		} else if (k1 > k2)
			suggestion = SKIP_RIGHT;
		    else
			suggestion = SKIP_LEFT;
	    } else {
		new_trap = 3; /* still */
		/* XXX ? Are these tests needed or is SYM safe all the time? */
		if (sgn(a) != sgn(b)) {
		    /*  minima in magnitude */
		    /* Search for exact mag. minima, look for complex pair */
		    suggestion = SYM;
		} else if (a_mag > b_mag || (a_mag == b_mag
			   && fabs(a) > fabs(b)))
		    suggestion = SPLIT_LEFT;
		else
		    suggestion = SPLIT_RIGHT;
	    }
	}
	if (Consec_Moves >= 3 && CKTpzTrapped == new_trap) {
	    new_trap = CKTpzTrapped;
	    if (Last_Move == MID_LEFT || Last_Move == NEAR_RIGHT)
		suggestion = SPLIT_LEFT;
	    else if (Last_Move == MID_RIGHT || Last_Move == NEAR_LEFT)
		suggestion = SPLIT_RIGHT;
	    else
		abort( );	/* XXX */
	    Consec_Moves = 0;
	}
    }

    CKTpzTrapped = new_trap;
#ifdef PZDEBUG
    DEBUG(1) {
	if (set[0] && set[1] && set[2])
	    fprintf(stderr, "given %.15g %.15g / %.15g %.15g / %.15g %.15g\n",
		set[0]->s.real, set[0]->s.imag, set[1]->s.real, set[1]->s.imag,
		set[2]->s.real, set[2]->s.imag);
	fprintf(stderr, "suggestion(%d/%d/%d | %d): %s\n",
		NFlat, NZeros, Max_Zeros, CKTpzTrapped, snames[suggestion]);
    }
#endif
    return suggestion;
}

/* CKTpzRunTrial: eval the function at a given 's', fold in deflation */

int 
CKTpzRunTrial(CKTcircuit *ckt, PZtrial **new_trialp, PZtrial **set)
{
    PZAN *job = (PZAN *) ckt->CKTcurJob;

    PZtrial	*match, *new_trial;
    PZtrial	*p, *prev;
    SPcomplex	def_frac, diff_frac;
    double	reltol, abstol;
    int		def_mag, diff_mag, error = 0;
    int		i;
    int		pretest, shifted, was_shifted;
    int		repeat;

    new_trial = *new_trialp;

    if (new_trial->s.imag < 0.0)
	new_trial->s.imag *= -1.0;

    /* Insert the trial into the list of Trials, while calculating
	the deflation factor from previous zeros */

    pretest = 0;
    shifted = 0;
    repeat = 0;

    do {

	def_mag = 0;
	def_frac.real = 1.0;
	def_frac.imag = 0.0;
	was_shifted = shifted;
	shifted = 0;

	prev = NULL;
	match = NULL;

	for (p = Trials; p != NULL; p = p->next) {

	    C_SUBEQ(diff_frac,p->s,new_trial->s);

	    if (diff_frac.real < 0.0
		|| (diff_frac.real == 0.0 && diff_frac.imag < 0.0)) {
		prev = p;
	    }

	    if (p->flags & ISAROOT) {
		abstol = 1e-5;
		reltol = 1e-6;
	    } else {
		abstol = 1e-20;
		reltol = 1e-12;
	    }

	    if (diff_frac.imag == 0.0 &&
		fabs(diff_frac.real) / (fabs(p->s.real) + abstol/reltol)
		< reltol) {

#ifdef PZDEBUG
		    DEBUG(1) {
			fprintf(stderr,
			"diff_frac.real = %10g, p->s = %10g, nt = %10g\n",
				diff_frac.real, p->s.real, new_trial->s.real);
			fprintf(stderr, "ab=%g,rel=%g\n", abstol, reltol);
		    }
#endif
		if (was_shifted || p->count >= 3
		    || !alter(new_trial, set[1], abstol, reltol)) {
		    /* assume either a root or minima */
		    p->count = 0;
		    pretest = 1;
		    break;
		} else
		    p->count += 1;	/* try to shift */

		shifted = 1;	/* Re-calculate deflation */
		break;

	    } else {
		if (!CKTpzTrapped)
		    p->count = 0;
		if (p->flags & ISAROOT) {
		    diff_mag = 0;
		    C_NORM(diff_frac,diff_mag);
		    if (diff_frac.imag != 0.0) {
			C_MAG2(diff_frac);
			diff_mag *= 2;
		    }
		    C_NORM(diff_frac,diff_mag);

		    for (i = p->multiplicity; i > 0; i--) {
			C_MUL(def_frac,diff_frac);
			def_mag += diff_mag;
			C_NORM(def_frac,def_mag);
		    }
		} else if (!match)
		    match = p;
	    }
	}

    } while (shifted);

    if (pretest) {

#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "Pre-test taken\n");

	/* XXX Should catch the double-zero right off
	 * if K is 0.0
	 * instead of forcing a re-converge */
	DEBUG(1) {
	    fprintf(stderr, "NIpzK == %g, mag = %d\n", NIpzK, NIpzK_mag);
	    fprintf(stderr, "over at %.30g %.30g (new %.30g %.30g, %x)\n",
		p->s.real, p->s.imag, new_trial->s.real, new_trial->s.imag,
		p->flags);
	}
#endif
	if (!(p->flags & ISAROOT) && CKTpzTrapped == 3
		&& NIpzK != 0.0 && NIpzK_mag > -10) {
#ifdef notdef
//	    if (p->flags & ISAROOT) {
//	        /* Ugh! muller doesn't work right */
//	        new_trial->flags = ISAMINIMA;
//	        new_trial->s.imag = scalb(NIpzK, (int) (NIpzK_mag / 2));
//	        pretest = 0;
//	    } else {
#endif
	    p->flags |= ISAMINIMA;
	    tfree(new_trial);
	    *new_trialp = p;
	    repeat = 1;
	} else if (p->flags & ISAROOT) {
#ifdef PZDEBUG
	    DEBUG(1) fprintf(stderr, "Repeat at %.30g %.30g\n",
		p->s.real, p->s.imag);
#endif
	    *new_trialp = p;
	    p->flags |= ISAREPEAT;
	    p->multiplicity += 1;
	    repeat = 1;
	} else {
	    /* Regular zero, as precise as we can get it */
	    error = E_SINGULAR;
	}

    }

    if (!repeat) {
	if (!pretest) {
	    /* Run the trial */
	    ckt->CKTniState |= NIPZSHOULDREORDER;	/* XXX */
	    if (!(ckt->CKTniState & NIPZSHOULDREORDER)) {
		CKTpzLoad(ckt, &new_trial->s);
#ifdef PZDEBUG
		DEBUG(3) {
		    printf("Original:\n");
		    SMPprint(ckt->CKTmatrix, stdout);
		}
#endif
		error = SMPcLUfac(ckt->CKTmatrix, ckt->CKTpivotAbsTol);
		if (error == E_SINGULAR) {
#ifdef PZDEBUG
		    DEBUG(1) printf("Needs reordering\n");
#endif
		    ckt->CKTniState |= NIPZSHOULDREORDER;
		} else if (error != OK)
		    return error;
	    }
	    if (ckt->CKTniState & NIPZSHOULDREORDER) {
		CKTpzLoad(ckt, &new_trial->s);
		error = SMPcReorder(ckt->CKTmatrix, 1.0e-30,
		    0.0 /* 0.1 Piv. Rel. */,
		    &(job->PZnumswaps));
	    }

	    if (error != E_SINGULAR) {
		ckt->CKTniState &= ~NIPZSHOULDREORDER;
#ifdef PZDEBUG
		DEBUG(3) {
		    printf("Factored:\n");
		    SMPprint(ckt->CKTmatrix, NULL);
		}
#endif
		error = SMPcDProd(ckt->CKTmatrix, &new_trial->f_raw,
		    &new_trial->mag_raw);
	    }
	}

	if (error == E_SINGULAR || (new_trial->f_raw.real == 0.0
	    && new_trial->f_raw.imag == 0.0)) {
	    new_trial->f_raw.real = 0.0;
	    new_trial->f_raw.imag = 0.0;
	    new_trial->mag_raw = 0;
	    new_trial->f_def.real = 0.0;
	    new_trial->f_def.imag = 0.0;
	    new_trial->mag_def = 0;
	    new_trial->flags = ISAROOT;
	    /*printf("SMP Det: Singular\n");*/
	} else if (error != OK)
	    return error;
	else {

	    /* PZnumswaps is either 0 or 1 */
	    new_trial->f_raw.real *=  job->PZnumswaps;
	    new_trial->f_raw.imag *=  job->PZnumswaps;
	    
#ifdef PZDEBUG	    
	    printf("SMP Det: (%g,%g)^%d\n", new_trial->f_raw.real,
		    new_trial->f_raw.imag, new_trial->mag_raw);
#endif	    

	    new_trial->f_def.real = new_trial->f_raw.real;
	    new_trial->f_def.imag = new_trial->f_raw.imag;
	    new_trial->mag_def = new_trial->mag_raw;

	    C_DIV(new_trial->f_def,def_frac);
	    new_trial->mag_def -= def_mag;
	    C_NORM(new_trial->f_def,new_trial->mag_def);
	}

	/* Link into the rest of the list */
	if (prev) {
	    new_trial->next = prev->next;
	    if (prev->next)
		prev->next->prev = new_trial;
	    prev->next = new_trial;
	} else {
	    if (Trials)
		Trials->prev = new_trial;
	    else
		ZeroTrial = new_trial;
	    new_trial->next = Trials;
	    Trials = new_trial;
	}
	new_trial->prev = prev;

	NTrials += 1;

	if (!(new_trial->flags & ISAROOT)) {
	    if (match)
		check_flat(match, new_trial);
	    else
		NFlat = 1;
	}
    }

#ifdef PZDEBUG
    show_trial(new_trial, '*');
#endif

    return OK;
}

/* Process a zero; inc. zero count, deflate other trials */

int 
CKTpzVerify(PZtrial **set, PZtrial *new_trial)
{
    PZtrial	*next;
    int		diff_mag;
    SPcomplex	diff_frac;
    double	tdiff;

    PZtrial *t, *prev;

    NG_IGNORE(set);

    NZeros += 1;
    if (new_trial->s.imag != 0.0)
	NZeros += 1;
    NFlat = 0;

    if (new_trial->multiplicity == 0) {
	new_trial->flags |= ISAROOT;
	new_trial->multiplicity = 1;
    }

    prev = NULL;

    for (t = Trials; t; t = next) {

	next = t->next;

	if (t->flags & ISAROOT) {
	    prev = t;
	    /* Don't need to bother */
	    continue;
	}

	C_SUBEQ(diff_frac,new_trial->s,t->s);
	if (new_trial->s.imag != 0.0)
	    C_MAG2(diff_frac);

	tdiff = diff_frac.real;
	/* Note that Verify is called for each time the root is found, so
	 * multiplicity is not significant
	 */
	if (diff_frac.real != 0.0) {
	    diff_mag = 0;
	    C_NORM(diff_frac,diff_mag);
	    diff_mag *= -1;
	    C_DIV(t->f_def,diff_frac);
	    C_NORM(t->f_def,diff_mag);
	    t->mag_def += diff_mag;
	}

	if (t->s.imag != 0.0
		|| fabs(tdiff) / (fabs(new_trial->s.real) + 200) < 0.005) {
	    if (prev)
		prev->next = t->next;
	    if (t->next)
		t->next->prev = prev;
	    NTrials -= 1;
#ifdef PZDEBUG
	    show_trial(t, '-');
#endif
	    if (t == ZeroTrial) {
		if (t->next)
		    ZeroTrial = t->next;
		else if (t->prev)
		    ZeroTrial = t->prev;
		else
		    ZeroTrial = NULL;
	    }
	    if (t == Trials) {
		Trials = t->next;
	    }
	    tfree(t);
	} else {

	    if (prev)
		check_flat(prev, t);
	    else
		NFlat = 1;

	    if (t->flags & ISAMINIMA)
		t->flags &= ~ISAMINIMA;

	    prev = t;
#ifdef PZDEBUG
	    show_trial(t, '+');
#endif
	}

    }

    return 1;	/* always ok */
}

/* pzseek: search the trial list (given a starting point) for the first
 *	non-zero entry; direction: -1 for prev, 1 for next, 0 for next
 *	-or- first.  Also, sets "Guess_Param" at the next reasonable
 *	value to guess at if the search falls of the end of the list
 */

static PZtrial *
pzseek(PZtrial *t, int dir)
{
    Guess_Param = dir;
    if (t == NULL)
	return NULL;

    if (dir == 0 && !(t->flags & ISAROOT) && !(t->flags & ISAMINIMA))
	return t;

    do {
	if (dir >= 0)
	    t = t->next;
	else
	    t = t->prev;
    } while (t && ((t->flags & ISAROOT) || (t->flags & ISAMINIMA)));

    return t;
}

static void
clear_trials(int mode)
{
    PZtrial *t, *next, *prev;

    prev = NULL;

    for (t = Trials; t; t = next) {
	next = t->next;
	if (mode || !(t->flags & ISAROOT)) {
	    tfree(t);
	} else {
	    if (prev)
		prev->next = t;
	    else
		Trials = t;
	    t->prev = prev;
	    prev = t;
	}
    }

    if (prev)
	prev->next = NULL;
    else
	Trials = NULL;
}

void
CKTpzUpdateSet(PZtrial **set, PZtrial *new)
{
    int	this_move;

    this_move = 0;

    if (new->s.imag != 0.0) {
	set[2] = set[1];
	set[1] = set[0];
	set[0] = new;
    } else if (!set[1])
	set[1] = new;
    else if (!set[2] && new->s.real > set[1]->s.real) {
	set[2] = new;
    } else if (!set[0]) {
	set[0] = new;
    } else if (new->flags & ISAMINIMA) {
	set[1] = new;
    } else if (new->s.real < set[0]->s.real) {
	set[2] = set[1];
	set[1] = set[0];
	set[0] = new;
	this_move = FAR_LEFT;
    } else if (new->s.real < set[1]->s.real) {
	if (!CKTpzTrapped || new->mag_def < set[1]->mag_def
	    || (new->mag_def == set[1]->mag_def
	    && fabs(new->f_def.real) < fabs(set[1]->f_def.real))) {
		/* Really should check signs, not just compare fabs( ) */
	    set[2] = set[1];	/* XXX = set[2]->prev :: possible opt */
	    set[1] = new;
	    this_move = MID_LEFT;
	} else {
	    set[0] = new;
	    this_move = NEAR_LEFT;
	}
    } else if (new->s.real < set[2]->s.real) {
	if (!CKTpzTrapped || new->mag_def < set[1]->mag_def
	    || (new->mag_def == set[1]->mag_def
	    && fabs(new->f_def.real) < fabs(set[1]->f_def.real))) {
		/* Really should check signs, not just compare fabs( ) */
	    set[0] = set[1];
	    set[1] = new;
	    this_move = MID_RIGHT;
	} else {
	    set[2] = new;
	    this_move = NEAR_RIGHT;
	}
    } else {
	set[0] = set[1];
	set[1] = set[2];
	set[2] = new;
	this_move = FAR_RIGHT;
    }

    if (CKTpzTrapped && this_move == Last_Move)
	Consec_Moves += 1;
    else
	Consec_Moves = 0;
    Last_Move = this_move;
}

void
zaddeq(double *a, int *amag, double x, int xmag, double y, int ymag)
{
    /* Balance magnitudes . . . */
    if (xmag > ymag) {
	*amag = xmag;
	if (xmag > 50 + ymag)
	    y = 0.0;
	else
	    for (xmag -= ymag; xmag > 0; xmag--)
		y /= 2.0;
    } else {
	*amag = ymag;
	if (ymag > 50 + xmag)
	    x = 0.0;
	else
	    for (ymag -= xmag; ymag > 0; ymag--)
		x /= 2.0;
    }

    *a = x + y;
    if (*a == 0.0)
	*amag = 0;
    else {
	while (fabs(*a) > 1.0) {
	    *a /= 2.0;
	    *amag += 1;
	}
	while (fabs(*a) < 0.5) {
	    *a *= 2.0;
	    *amag -= 1;
	}
    }
}

#ifdef PZDEBUG
static void
show_trial(PZtrial *new_trial, char x)
{
    DEBUG(1) {
        if(new_trial) {
            fprintf(stderr, "%c (%3d/%3d) %.15g %.15g :: %.30g %.30g %d\n", x,
                    NIter, new_trial->seq_num, new_trial->s.real, new_trial->s.imag,
                    new_trial->f_def.real, new_trial->f_def.imag, new_trial->mag_def);
            if (new_trial->flags & ISANABERRATION)
                fprintf(stderr, "*** numerical aberration ***\n");
        } else {
            fprintf(stderr, "%c (%3d/---) new_trial = nil\n", x, NIter);
        }
    }
}
#endif

static void
check_flat(PZtrial *a, PZtrial *b)
{
    int		diff_mag;
    SPcomplex	diff_frac;
    double	mult;

    diff_mag = a->mag_def - b->mag_def;
    if (abs(diff_mag) <= 1) {
	if (diff_mag == 1)
	    mult = 2.0;
	else if (diff_mag == -1)
	    mult = 0.5;
	else
	    mult = 1.0;
	C_SUBEQ(diff_frac, mult * a->f_def, b->f_def);
	C_MAG2(diff_frac);
	if (diff_frac.real < 1.0e-20)
	    NFlat += 1;
    }
	/* XXX else NFlat = ?????? */
}

/* XXX ZZZ */
int
CKTpzStep(int strat, PZtrial **set)
{
    switch (strat) {
    case INIT:
	if (!set[1]) {
	    set[1] = pzseek(ZeroTrial, 0);
	} else if (!set[2])
	    set[2] = pzseek(set[1], 1);
	else if (!set[0])
	    set[0] = pzseek(set[1], -1);
	break;

    case SKIP_LEFT:
	set[0] = pzseek(set[0], -1);
	break;

    case SKIP_RIGHT:
	set[2] = pzseek(set[2], 1);
	break;

    case SHIFT_LEFT:
	set[2] = set[1];
	set[1] = set[0];
	set[0] = pzseek(set[0], -1);
	break;

    case SHIFT_RIGHT:
	set[0] = set[1];
	set[1] = set[2];
	set[2] = pzseek(set[2], 1);
	break;

    }
    if (!set[0] || !set[1] || !set[2])
	return 0;
    else
	return 1;
}

void
CKTpzReset(PZtrial **set)
{
    CKTpzTrapped = 0;
    Consec_Moves = 0;

    set[1] = pzseek(ZeroTrial, 0);
    if (set[1] != NULL) {
	set[0] = pzseek(set[1], -1);
	set[2] = pzseek(set[1], 1);
    } else {
	set[0] = NULL;
	set[2] = NULL;
    }
}

static int
alter(PZtrial *new, PZtrial *nearto, double abstol, double reltol)
{
    double	p1, p2;

#ifdef PZDEBUG
    DEBUG(1) {
	fprintf(stderr, "ALTER from: %.30g %.30g\n",
		new->s.real, new->s.imag);
	if (nearto->prev)
	    fprintf(stderr, "nt->prev %g\n", nearto->prev->s.real);
	if (nearto->next)
	    fprintf(stderr, "nt->next %g\n", nearto->next->s.real);
    }
#endif

    if (CKTpzTrapped != 2) {
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "not 2\n");
#endif
	p1 = nearto->s.real;
	if (nearto->flags & ISAROOT)
	    p1 -= 1e-6 * nearto->s.real + 1e-5;
	if (nearto->prev) {
	    p1 += nearto->prev->s.real;
#ifdef PZDEBUG
	    DEBUG(1) fprintf(stderr, "p1 %g\n", p1);
#endif
	} else
	    p1 -= 10.0 * (fabs(p1) + 1.0);

	p1 /= 2.0;
    } else
	p1 = nearto->s.real;

    if (CKTpzTrapped != 1) {
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "not 1\n");
#endif
	p2 = nearto->s.real;
	if (nearto->flags & ISAROOT)
	    p2 += 1e-6 * nearto->s.real + 1e-5; /* XXX Would rather use pow(2)*/
	if (nearto->next) {
	    p2 += nearto->next->s.real;
#ifdef PZDEBUG
	    DEBUG(1) fprintf(stderr, "p2 %g\n", p2);
#endif
	} else
	    p2 += 10.0 * (fabs(p2)+ 1.0);

	p2 /= 2.0;
    } else
	p2 = nearto->s.real;

    if ((nearto->prev &&
	 fabs(p1 - nearto->prev->s.real) /
	 fabs(nearto->prev->s.real) + abstol/reltol < reltol)
	||
	(nearto->next &&
	 fabs(p2 - nearto->next->s.real) /
	 fabs(nearto->next->s.real) + abstol/reltol < reltol)) {

#ifdef PZDEBUG
	DEBUG(1)
		fprintf(stderr, "Bailed out\n");
#endif

	return 0;
	}

    if (CKTpzTrapped != 2 && nearto->s.real - p1 > p2 - nearto->s.real) {
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "take p1\n");
#endif
	new->s.real = p1;
    } else {
#ifdef PZDEBUG
	DEBUG(1) fprintf(stderr, "take p2\n");
#endif
	new->s.real = p2;
    }

#ifdef PZDEBUG
    DEBUG(1) fprintf(stderr, "ALTER to  : %.30g %.30g\n",
	new->s.real, new->s.imag);
#endif
    return 1;

}
