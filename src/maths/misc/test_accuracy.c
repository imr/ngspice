/* Paolo Nenzi 2002 - This program tests some machine
 * dependent variables.
 */

/* Nota:
 *
 * Compilare due volte nel seguente modo:
 *
 * gcc test_accuracy.c -o test_64accuracy -lm
 * gcc -DIEEEDOUBLE test_accuracy.c -o test_53accuracy -lm
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fpu_control.h>

int main (void)
{
 fpu_control_t prec;
 double acc=1.0;
 double xl = 0.0;
 double xu = 1.0;
 double xh, x1, x2;
 double xhold =0.0;
 

#ifdef IEEEDOUBLE 
 _FPU_GETCW(prec);
 prec &= ~_FPU_EXTENDED;
  prec |= _FPU_DOUBLE;
 _FPU_SETCW(prec);
#endif 
 
 for( ; (acc + 1.0) > 1.0 ; ) {
	acc *= 0.5;	
    }
    acc *= 2.0;
 printf("Accuracy: %e\n", acc);  
 printf("------------------------------------------------------------------\n"); 
 
 xh = 0.5 * (xl + xu);

    for( ; (xu-xl > (2.0 * acc * (xu + xl))); ) {
	
	 x1 = 1.0 / ( 1.0 + (0.5 * xh) );	 
	 x2 = xh / ( exp(xh) - 1.0 );
	 
	if( (x1 - x2) <= (acc * (x1 + x2))) {
	    xl = xh;
	    xhold = xh;
	} else {
	    xu = xh;
	    xhold = xh;
	}
	xh = 0.5 * (xl + xu);
/*	if (xhold == xh) break; */
    }
printf("xu-xl: %e \t cond: %e \t xh: %e\n", (xu-xl), (2.0 * acc * (xu + xl)), xh);

exit(1);

}
