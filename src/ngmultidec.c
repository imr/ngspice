/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet Roychowdury
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ngspice/spmatrix.h"

#define THRSH 0.01
#define ABS_THRSH 0
#define DIAG_PIVOTING	1

#undef DEBUG_LEVEL1

/* `-u' option showing the usage help is changed to `-h'. -- ro */

extern void usage(char **argv);
extern void comments(double r,double l,double g,double c,double ctot,double cm,double lm,double k,char *name,int num, double len);
extern double phi(int i, double arg);
extern void spErrorMessage(MatrixPtr, FILE*, char*);

int
main (int argc, char **argv)
{
  int ch;
  int errflg=0,i,j;
  double l,c,ctot,r=0.0,g=0.0,k=0.0,lm=0.0,cm=0.0,len;
  unsigned gotl=0,gotc=0,gotr=0,gotg=0,gotk=0,gotcm=0,gotlen=0;
  unsigned gotname=0, gotnum=0;
  char *name = "";
  double **matrix, **inverse;
  double *tpeigenvalues, *gammaj;
  char *options;
  int num, node;
  char **pname, *s;
  int use_opt;
  char *optarg;

  pname = argv;
  argv++;
  argc--;

  ch = 0;
  while (argc > 0) {
    s = *argv++;
    argc--;
    while ((ch = *s++) != '\0') {
      if (*s)
	optarg = s;
      else if (argc)
	optarg = *argv;
      else
	optarg = NULL;
      use_opt = 0;

      switch (ch) {
      case 'o':
	name = TMALLOC(char, strlen(optarg));
	(void) strcpy(name,optarg);
	gotname=1;
	use_opt = 1;
	break;
      case 'l':
	sscanf(optarg,"%lf",&l);
	gotl=1;
	use_opt = 1;
	break;
      case 'c':
	sscanf(optarg,"%lf",&c);
	gotc=1;
	use_opt = 1;
	break;
      case 'r':
	sscanf(optarg,"%lf",&r);
	use_opt = 1;
	gotr=1;
	break;
      case 'g':
	sscanf(optarg,"%lf",&g);
	use_opt = 1;
	gotg=1;
	break;
      case 'k':
	sscanf(optarg,"%lf",&k);
	use_opt = 1;
	gotk=1;
	break;
      case 'x':
	sscanf(optarg,"%lf",&cm);
	use_opt = 1;
	gotcm=1;
	break;
      case 'L':
	sscanf(optarg,"%lf",&len);
	use_opt = 1;
	gotlen=1;
	break;
      case 'n':
	sscanf(optarg,"%d",&num);
	use_opt = 1;
	gotnum=1;
	break;
      case 'h':
	usage(pname);
	exit(1);
	break;
      case '-':
	break;
      default:
	usage(pname);
	exit(2);
	break;
      }
      if (use_opt) {
	if (optarg == s)
	  s += strlen(s);
	else if (optarg) {
	  argc--;
	  argv++;
	}
      }
    }
  }

  if (errflg) {
    usage(argv);
    exit (2);
  }

  if (gotl + gotc + gotname + gotnum + gotlen < 5) {
    fprintf(stderr,"l, c, model_name, number_of_conductors and length must be specified.\n");
    fprintf(stderr,"%s -u for details.\n",pname[0]);
    fflush(stdout);
    exit(1);
  }

  if (fabs(k) >= 1.0) {
    fprintf(stderr,"Error: |k| must be less than 1.0\n");
    fflush(stderr);
    exit(1);
  }

  if (num == 1) {
    fprintf(stdout,"* single conductor line\n");
    fflush(stdout);
    exit(1);
  }

  lm = l*k;
  switch(num) {

  case 1: ctot = c; break;
  case 2: ctot = c + cm; break;
  default: ctot = c + 2*cm; break;
  }

  comments(r,l,g,c,ctot,cm,lm,k,name,num,len);

  matrix = TMALLOC(double *, num + 1);
  inverse = TMALLOC(double *, num + 1);
  tpeigenvalues = TMALLOC(double, num + 1);

  for (i=1;i<=num;i++) {
    matrix[i] = TMALLOC(double, num + 1);
    inverse[i] = TMALLOC(double, num + 1);
  }

  for (i=1;i<=num;i++) {
    tpeigenvalues[i] = -2.0 * cos(M_PI*i/(num+1));
  }

  for (i=1;i<=num;i++) {
    for (j=1;j<=num;j++) {
      matrix[i][j] = phi(i-1,tpeigenvalues[j]);
    }
  }
  gammaj = TMALLOC(double, num + 1);

  for (j=1;j<=num;j++) {
    gammaj[j] = 0.0;
    for (i=1;i<=num;i++) {
      gammaj[j] += matrix[i][j] * matrix[i][j];
    }
    gammaj[j] = sqrt(gammaj[j]);
  }

  for (j=1;j<=num;j++) {
    for (i=1;i<=num; i++) {
      matrix[i][j] /= gammaj[j];
    }
  }

  tfree(gammaj);

  /* matrix = M set up */

  {
    MatrixPtr othermatrix;
    double *rhs, *solution;
    double *irhs, *isolution;
    int errflg, err, singular_row, singular_col;
    double *elptr;

    rhs = TMALLOC(double, num + 1);
    irhs = TMALLOC(double, num + 1);
    solution = TMALLOC(double, num + 1);
    isolution = TMALLOC(double, num + 1);

    othermatrix = spCreate(num,0,&errflg);

    for (i=1;i<=num;i++) {
      for (j=1; j<=num; j++) {
	elptr = spGetElement(othermatrix,i,j);
	*elptr = matrix[i][j];
      }
    }

#ifdef DEBUG_LEVEL1
    (void) spPrint(othermatrix,0,1,0);
#endif

    for (i=1;i<=num;i++) rhs[i] = 0.0;
    rhs[1]=1.0;

    err =
      spOrderAndFactor(othermatrix,rhs,THRSH,ABS_THRSH,DIAG_PIVOTING);

    spErrorMessage(othermatrix,stderr,NULL);

    switch(err) {

    case spNO_MEMORY:
      fprintf(stderr,"No memory in spOrderAndFactor\n");
      fflush(stderr);
      exit(1);
    case spSINGULAR:
      (void)
	spWhereSingular(othermatrix,&singular_row,&singular_col);
      fprintf(stderr,"Singular matrix: problem in row %d and col %d\n", singular_row, singular_col);
      fflush(stderr);
      exit(1);
    default: break;
    }

    for (i=1;i<=num;i++) {
      for (j=1;j<=num;j++) {
	rhs[j] = (j==i?1.0:0.0);
	irhs[j] = 0.0;
      }
      (void) spSolveTransposed(othermatrix,rhs,solution, irhs, isolution);
      for (j=1;j<=num;j++) {
	inverse[i][j] = solution[j];
      }
    }

    tfree(rhs);
    tfree(solution);
  }

  /* inverse = M^{-1} set up */

  fprintf(stdout,"\n");
  fprintf(stdout,"* Lossy line models\n");

  options = "rel=1.2 nocontrol";
  for (i=1;i<=num;i++) {
    fprintf(stdout,".model mod%d_%s ltra %s r=%0.12g l=%0.12g g=%0.12g c=%0.12g len=%0.12g\n",
	    i,name,options,r,l+tpeigenvalues[i]*lm,g,ctot-tpeigenvalues[i]*cm,len);
    /*i,name,options,r,l+tpeigenvalues[i]*lm,g,ctot+tpeigenvalues[i]*cm,len);*/
  }


  fprintf(stdout,"\n");
  fprintf(stdout,"* subcircuit m_%s - modal transformation network for %s\n",name,name);
  fprintf(stdout,".subckt m_%s", name);
  for (i=1;i<= 2*num; i++) {
    fprintf(stdout," %d",i);
  }
  fprintf(stdout,"\n");
  for (j=1;j<=num;j++) fprintf(stdout,"v%d %d 0 0v\n",j,j+2*num);

  for (j=1;j<=num;j++) {
    for (i=1; i<=num; i++) {
      fprintf(stdout,"f%d 0 %d v%d %0.12g\n",
	      (j-1)*num+i,num+j,i,inverse[j][i]);
    }
  }

  node = 3*num+1;
  for (j=1;j<=num;j++) {
    fprintf(stdout,"e%d %d %d %d 0 %0.12g\n", (j-1)*num+1,
	    node, 2*num+j, num+1, matrix[j][1]);
    node++;
    for (i=2; i<num; i++) {
      fprintf(stdout,"e%d %d %d %d 0 %0.12g\n", (j-1)*num+i,
	      node,node-1,num+i,matrix[j][i]);
      node++;
    }
    fprintf(stdout,"e%d %d %d %d 0 %0.12g\n", j*num,j,node-1,
	    2*num,matrix[j][num]);
  }
  fprintf(stdout,".ends m_%s\n",name);

  fprintf(stdout,"\n");
  fprintf(stdout,"* Subckt %s\n", name);
  fprintf(stdout,".subckt %s",name);
  for (i=1;i<=2*num;i++) {
    fprintf(stdout," %d",i);
  }
  fprintf(stdout,"\n");

  fprintf(stdout,"x1");
  for (i=1;i<=num;i++)  fprintf(stdout," %d", i);
  for (i=1;i<=num;i++)  fprintf(stdout," %d", 2*num+i);
  fprintf(stdout," m_%s\n",name);

  for (i=1;i<=num;i++) 
    fprintf(stdout,"o%d %d 0 %d 0 mod%d_%s\n",i,2*num+i,3*num+i,i,name);

  fprintf(stdout,"x2");
  for (i=1;i<=num;i++)  fprintf(stdout," %d", num+i);
  for (i=1;i<=num;i++)  fprintf(stdout," %d", 3*num+i);
  fprintf(stdout," m_%s\n",name);

  fprintf(stdout,".ends %s\n",name);

  tfree(tpeigenvalues);
  for (i=1;i<=num;i++) {
    tfree(matrix[i]);
    tfree(inverse[i]);
  }
  tfree(matrix);
  tfree(inverse);
  tfree(name);

  return EXIT_NORMAL; 
}

void
usage(char **argv)
{

fprintf(stderr,"Purpose: make subckt. for coupled lines using uncoupled simple lossy lines\n");
fprintf(stderr,"Usage: %s -l<line-inductance> -c<line-capacitance>\n",argv[0]);
fprintf(stderr,"	   -r<line-resistance> -g<line-conductance> \n");
fprintf(stderr,"	   -k<inductive coeff. of coupling> \n");
fprintf(stderr,"	   -x<line-to-line capacitance> -o<subckt-name> \n");
fprintf(stderr,"	   -n<number of conductors> -L<length> -h\n");
fprintf(stderr,"Example: %s -n4 -l9e-9 -c20e-12 -r5.3 -x5e-12 -k0.7 -otest -L5.4\n\n",argv[0]);

fprintf(stderr,"See \"Efficient Transient Simulation of Lossy Interconnect\",\n");
fprintf(stderr,"J.S. Roychowdhury and D.O. Pederson, Proc. DAC 91 for details\n");
fprintf(stderr,"\n");
fflush(stderr);
}

void
comments(double r,double l,double g,double c,double ctot,double cm,double lm,double k,char *name,int num, double len)
{

fprintf(stdout,"* Subcircuit %s\n",name);
fprintf(stdout,"* %s is a subcircuit that models a %d-conductor transmission line with\n",name,num);
fprintf(stdout,"* the following parameters: l=%g, c=%g, r=%g, g=%g,\n",l,c,r,g);
fprintf(stdout,"* inductive_coeff_of_coupling k=%g, inter-line capacitance cm=%g,\n",k,cm);
fprintf(stdout,"* length=%g. Derived parameters are: lm=%g, ctot=%g.\n",len,lm,ctot);
fprintf(stdout,"* \n");
fprintf(stdout,"* It is important to note that the model is a simplified one - the\n");
fprintf(stdout,"* following assumptions are made: 1. The self-inductance l, the\n");
fprintf(stdout,"* self-capacitance ctot (note: not c), the series resistance r and the\n");
fprintf(stdout,"* parallel capacitance g are the same for all lines, and 2. Each line\n");
fprintf(stdout,"* is coupled only to the two lines adjacent to it, with the same\n");
fprintf(stdout,"* coupling parameters cm and lm. The first assumption implies that edge\n");
fprintf(stdout,"* effects have to be neglected. The utility of these assumptions is\n");
fprintf(stdout,"* that they make the sL+R and sC+G matrices symmetric, tridiagonal and\n");
fprintf(stdout,"* Toeplitz, with useful consequences (see \"Efficient Transient\n");
fprintf(stdout,"* Simulation of Lossy Interconnect\", by J.S.  Roychowdhury and\n");
fprintf(stdout,"* D.O Pederson, Proc. DAC 91).\n\n");
fprintf(stdout,"* It may be noted that a symmetric two-conductor line is\n");
fprintf(stdout,"* represented accurately by this model.\n\n");
fprintf(stdout,"* Subckt node convention:\n");
fprintf(stdout,"* \n");
fprintf(stdout,"*            |--------------------------|\n");
fprintf(stdout,"*      1-----|                          |-----n+1\n");
fprintf(stdout,"*      2-----|                          |-----n+2\n");
fprintf(stdout,"*         :  |   n-wire multiconductor  |  :\n");
fprintf(stdout,"*         :  |          line            |  :\n");
fprintf(stdout,"*    n-1-----|(node 0=common gnd plane) |-----2n-1\n");
fprintf(stdout,"*      n-----|                          |-----2n\n");
fprintf(stdout,"*            |--------------------------|\n\n");
fflush(stdout);
}

double 
phi(int i, double arg)
{
	double	rval;

	switch (i) {

	case 0:
		rval = 1.0;
		break;
	case 1:
		rval = arg;
		break;
	default:
		rval = arg*phi(i-1,arg) - phi(i-2,arg);
	}
	return rval;
}
