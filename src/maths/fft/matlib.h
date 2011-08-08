/* a few routines from a vector/matrix library */

void xpose(double *indata, long iRsiz, double *outdata, long oRsiz, long Nrows, long Ncols);
/* not in-place matrix transpose	*/
/* INPUTS */
/* *indata = input data array	*/
/* iRsiz = offset to between rows of input data array	*/
/* oRsiz = offset to between rows of output data array	*/
/* Nrows = number of rows in input data array	*/
/* Ncols = number of columns in input data array	*/
/* OUTPUTS */
/* *outdata = output data array	*/

void cxpose(double *indata, long iRsiz, double *outdata, long oRsiz, long Nrows, long Ncols);
/* not in-place complex matrix transpose	*/
/* INPUTS */
/* *indata = input data array	*/
/* iRsiz = offset to between rows of input data array	*/
/* oRsiz = offset to between rows of output data array	*/
/* Nrows = number of rows in input data array	*/
/* Ncols = number of columns in input data array	*/
/* OUTPUTS */
/* *outdata = output data array	*/

void cvprod(double *a, double *b, double *out, long N);
/* complex vector product, can be in-place */
/* product of complex vector *a times complex vector *b */
/* INPUTS */
/* N vector length */
/* *a complex vector length N complex numbers */
/* *b complex vector length N complex numbers */
/* OUTPUTS */
/* *out complex vector length N */
