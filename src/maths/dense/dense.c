
#include "dense.h"
#include "denseinlines.h"
#include "ngspice/ngspice.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "ngspice/bool.h"
#include "ngspice/iferrmsg.h"

cplx cdet(CMat* M);
double det(Mat* M);
void resizecmat(CMat* A, int r, int c);

void showmat(Mat* A) {
	if (A->row > 0 && A->col > 0) {

		printf("[");
		for (int i = 0; i < A->row; i++) {
			for (int j = 0; j < A->col; j++) {
				if (j < A->col) {
					printf("%f\t", A->d[i][j]);
				}
				else {
					printf("%f", A->d[i][j]);
				}
			}
			if (i < A->row) {
				printf("\n");
			}
			else {
				printf("]\n");
			}
		}
		printf("\n");
	}
	else {
		printf("[]");
	}
}

void showcmat(CMat* A) {
	if (A->row > 0 && A->col > 0) {

		printf("[");
		for (int i = 0; i < A->row; i++) {
			for (int j = 0; j < A->col; j++) {
				if (j < A->col) {
					printf("%f+i%f\t", A->d[i][j].re, A->d[i][j].im);
				}
				else {
					printf("%f+i%f\t", A->d[i][j].re, A->d[i][j].im);
				}
			}
			if (i < A->row) {
				printf("\n");
			}
			else {
				printf("]\n");
			}
		}
		printf("\n");
	}
	else {
		printf("[]");
	}
}

CMat* newcmat(int r, int c, double dr, double di) {
	CMat* M = (CMat*)tmalloc(sizeof(CMat));

	if (M == NULL) return NULL;
	
	M->row = r; M->col = c;
	M->d = (cplx**)tmalloc(sizeof(cplx*) * (size_t)r);
	if (M->d == NULL) {
		tfree(M);  return NULL;
	}

	for (int i = 0; i < r; i++)
		M->d[i] = (cplx*)tmalloc(sizeof(cplx) * (size_t)c);

	for (int i = 0; i < M->row; i++) {
		for (int j = 0; j < M->col; j++) {
			M->d[i][j].re = dr;
			M->d[i][j].im = di;
		}
	}
	return M;
}

void init(Mat* A, double d)
{
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			A->d[i][j] = d;
		}
	}
}
void cinit(CMat* A, double dr, double di)
{
	cplx ci;
	ci.re = dr;
	ci.im = di;
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			A->d[i][j] = ci;
		}
	}
}



CMat* newcmatnoinit(int r, int c) {
	CMat* M = (CMat*)tmalloc(sizeof(CMat));
	if (M == NULL) return NULL;
	M->row = r; M->col = c;

	M->d = (cplx**)tmalloc(sizeof(cplx*) * (size_t)r);
	for (int i = 0; i < r; i++)
		M->d[i] = (cplx*)tmalloc(sizeof(cplx) * (size_t)c);
	return M;
}

Mat* newmat(int r, int c, double d) {
	Mat* M = (Mat*)tmalloc(sizeof(Mat));
	if (M == NULL) return NULL;

	M->row = r; M->col = c;
	M->d = (double**)tmalloc(sizeof(double*) * (size_t)r);
	for (int i = 0; i < r; i++)
		M->d[i] = (double*)tmalloc(sizeof(double) * (size_t)c);

	for (int i = 0; i < M->row; i++) {
		for (int j = 0; j < M->col; j++) {
			M->d[i][j] = d;
		}
	}
	return M;
}

Mat* newmatnoinit(int r, int c) {
	Mat* M = (Mat*)tmalloc(sizeof(Mat));
	if (M == NULL) return NULL;

	M->row = r; M->col = c;
	M->d = (double**)tmalloc(sizeof(double*) * (size_t)r);
	for (int i = 0; i < r; i++)
		M->d[i] = (double*)tmalloc(sizeof(double) * (size_t)c);

	return M;
}


void resizemat(Mat* A, int r, int c)
{
	if (A == NULL) return;
	if ((A->row == r) && (A->col == c))
		return;

	for (int ri = 0; ri < A->row; ri++)
		tfree(A->d[ri]);

	if (A->d != NULL)
		tfree(A->d);
	
	A->row = r; A->col = c;
	A->d = (double**)tmalloc(sizeof(double*) * (size_t)r);

	if (A->d == NULL) return;

	for (int i = 0; i < r; i++)
		A->d[i] = (double*)tmalloc(sizeof(double) * (size_t)c);
}

void resizecmat(CMat* A, int r, int c)
{
	if (A == NULL) return;
	if ((A->row == r) && (A->col == c))
		return;

	for (int ri = 0; ri < A->row; ri++)
		tfree(A->d[ri]);

	if (A->d != NULL)
		tfree(A->d);

	A->row = r; A->col = c;
	A->d = (cplx**)tmalloc(sizeof(cplx*) * (size_t)r);

	if (A->d == NULL) return;

	for (int i = 0; i < r; i++)
		A->d[i] = (cplx*)tmalloc(sizeof(cplx) * (size_t)c);
}


void freecmat(CMat* A) {
	if (A == NULL) return;

	for (int r = 0; r < A->row; r++)
		tfree(A->d[r]);

	if (A->d!=NULL)
		tfree(A->d);

	tfree(A);
}

void freemat(Mat* A) {
	if (A == NULL) return;

	for (int r = 0; r < A->row; r++)
		tfree(A->d[r]);

	if (A->d!=NULL)
		tfree(A->d);

	tfree(A);
}

CMat* ceye(int n) {
	CMat* I = newcmat(n, n, 0,0);
	for (int i = 0; i < n; i++) {
		I->d[i][i].re = 1;
	}
	return I;
}

Mat* eye(int n) {
	Mat* I = newmat(n, n, 0);
	for (int i = 0; i < n; i++) {
		I->d[i][i] = 1;
	}
	return I;
}
Mat* zeros(int r, int c) {
	Mat* Z = newmat(r, c, 0);
	return Z;
}

CMat* czeros(int r, int c) {
	CMat* Z = newcmat(r, c, 0,0);
	return Z;
}


Mat* ones(int r, int c) {
	Mat* O = newmat(r, c, 1);
	return O;
}

CMat* cones(int r, int c) {
	CMat* O = newcmat(r, c, 1,0);
	return O;
}

Mat* randm(int r, int c, double l, double u) {
	Mat* R = newmatnoinit(r, c);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			double rx = ((double)rand()) / ((double)RAND_MAX);
			R->d[i][j] = l + (u - l) * rx;
		}
	}
	return R;
}


CMat* randcm(int r, int c, double l, double u) {
	CMat* R = newcmatnoinit(r, c);

	double ks = u - l;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {

			setcplx(&(R->d[i][j]),
				l + ks * ((double)rand()) / ((double)RAND_MAX),
				l + ks * ((double)rand()) / ((double)RAND_MAX));
		}
	}
	return R;
}



double get(Mat* M, int r, int c) {	
	double d = M->d[r][c];
	return d;
}

cplx getcplx(CMat* M, int r, int c)
{
	return  M->d[r][c];
}

void set(Mat* M, int r, int c, double d) {
	M->d[r][c] = d;
}

void setc(CMat* M, int r, int c, cplx d)
{
	M->d[r][c] = d;
}

Mat* scalarmultiply(Mat* M, double c) {
	Mat* B = newmatnoinit(M->row, M->col);

	for (int i = 0; i < M->row; i++) {
		for (int j = 0; j < M->col; j++) {
			B->d[i][j] = M->d[i][j] * c;
		}
	}
	return B;
}


CMat* cscalarmultiply(CMat* M, double c) {
	CMat* B = newcmatnoinit(M->row, M->col);

	for (int i = 0; i < M->row; i++) {
		for (int j = 0; j < M->col; j++) {
			cmultd(&(B->d[i][j]), M->d[i][j], c);
		}
	}
	return B;
}


CMat* complexmultiply(CMat* M, cplx c) {
	CMat* B = newcmatnoinit(M->row, M->col);

	for (int i = 0; i < M->row; i++) {
		for (int j = 0; j < M->col; j++) {
			cmultc(&(B->d[i][j]), M->d[i][j], c);
		}
	}
	return B;
}

int complexmultiplydest(CMat* M, cplx c, CMat* dest) {
	
	for (int i = 0; i < M->row; i++) {
		for (int j = 0; j < M->col; j++) {
			cmultc(&(dest->d[i][j]), M->d[i][j], c);
		}
	}
	return (OK);
}

Mat* sum(Mat* A, Mat* B) {
	int r = A->row;
	int c = A->col;
	Mat* C = newmatnoinit(r, c);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			C->d[i][j] = A->d[i][j] + B->d[i][j];
		}
	}
	return C;
}

CMat* csum(CMat* A, CMat* B) {
	int r = A->row;
	int c = A->col;
	CMat* C = newcmatnoinit(r, c);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			caddc(&(C->d[i][j]), (A->d[i][j]), (B->d[i][j]));
		}
	}
	return C;
}
Mat* minus(Mat* A, Mat* B) {
	int r = A->row;
	int c = A->col;
	Mat* C = newmatnoinit(r, c);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			C->d[i][j] = A->d[i][j] - B->d[i][j];
		}
	}
	return C;
}


CMat* cminus(CMat* A, CMat* B) {
	int r = A->row;
	int c = A->col;
	CMat* C = newcmatnoinit(r, c);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			csubc(&(C->d[i][j]),
				A->d[i][j],
				B->d[i][j]);
		}
	}
	return C;
}


Mat* submat(Mat* A, int r1, int r2, int c1, int c2) {
	Mat* B = newmatnoinit(r2 - r1 + 1, c2 - c1 + 1);

	for (int i = r1, id =0 ; i <= r2; i++, id++) {
		for (int j = c1, jd=0; j <= c2; j++, jd++) {
			B->d[id][jd] = A->d[i][j];
		}
	}
	return B;
}


void submat2(Mat* A, Mat* B, int r1, int r2, int c1, int c2) {

	for (int i = r1, id = 0; i <= r2; i++, id++) {
		for (int j = c1, jd = 0; j <= c2; j++, jd++) {
			B->d[id][jd] = A->d[i][j];
		}
	}
}

CMat* subcmat(CMat* A, int r1, int r2, int c1, int c2) {
	CMat* B = newcmatnoinit(r2 - r1 + 1, c2 - c1 + 1);
	int k = 0;
	for (int i = r1; i <= r2; i++) {
		for (int j = c1; j <= c2; j++) {
			B->d[k++] = A->d[i * A->col + j];
		}
	}
	return B;
}


void subcmat2(CMat* A, CMat* B, int r1, int r2, int c1, int c2) {
	int k = 0;
	for (int i = r1; i <= r2; i++) {
		for (int j = c1; j <= c2; j++) {
			
			B->d[k++] = A->d[i* A->col + j];
		}
	}
}

Mat* multiply(Mat* A, Mat* B) {
	int r1 = A->row;
	int r2 = B->row;
	int c1 = A->col;
	int c2 = B->col;
	if (r1 == 1 && c1 == 1) {
		Mat* C = scalarmultiply(B, A->d[0][0]);
		return C;
	}
	else if (r2 == 1 && c2 == 1) {
		Mat* C = scalarmultiply(A, B->d[0][0]);
		return C;
	}
	Mat* C = newmatnoinit(r1, c2);
	for (int i = 0; i < r1; i++) {
		for (int j = 0; j < c2; j++) {
			double de = 0;
			for (int k = 0; k < r2; k++) {
				de += A->d[i][k] * B->d[k][j];
			}
			C->d[i][j] = de;
		}
	}
	return C;
}


CMat* cmultiply(CMat* A, CMat* B) {
	int r1 = A->row;
	int r2 = B->row;
	int c1 = A->col;
	int c2 = B->col;
	if (r1 == 1 && c1 == 1) {
		CMat* C = complexmultiply(B, A->d[0][0]);
		return C;
	}
	else if (r2 == 1 && c2 == 1) {
		CMat* C = complexmultiply(A, B->d[0][0]);
		return C;
	}
	CMat* C = newcmatnoinit(r1, c2);
	for (int i = 0; i < r1; i++) {
		for (int j = 0; j < c2; j++) {
			cplx de;
			de.re = de.im = 0;
			for (int k = 0; k < r2; k++) {
				caddc(&de, de, cmultco(A->d[i][k], B->d[k][j]));
			}
			C->d[i][j] = de;
		}
	}
	return C;
}


int cmultiplydest(CMat* A, CMat* B, CMat* dest) {
	int r1 = A->row;
	int r2 = B->row;
	int c1 = A->col;
	int c2 = B->col;
	// r2 must be = c1
	if (r1 == 1 && c1 == 1) {
		complexmultiplydest(B, A->d[0][0],dest);
		return (OK);
	}
	else if (r2 == 1 && c2 == 1) {
		complexmultiplydest(A, B->d[0][0],dest);
		return (OK);
	}

	for (int i = 0; i < r1; i++) {
		for (int j = 0; j < c2; j++) {
			cplx de;
			de.re = de.im = 0;
			for (int k = 0; k < r2; k++) {
				caddc(&de, de, cmultco(A->d[i][k], B->d[k][j]));
			}
			dest->d[i][j] = de;
		}
	}
	return (OK);
}
//-----------------------------------------
Mat* removerow(Mat* A, int r) {
	Mat* B = newmatnoinit(A->row - 1, A->col);
	int rowdest = 0;
	for (int i = 0; i < A->row; i++) {
		if (i!=r)
		{
			for (int j = 0; j < A->col; j++)
					B->d[rowdest][j] = A->d[i][j];

			rowdest++;
		}
	}
	return B;
}
Mat* removecol(Mat* A, int c) {
	Mat* B = newmatnoinit(A->row, A->col - 1);
	int coldest = 0 ;
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			if (j != c) {
				B->d[i][coldest] = A->d[i][j];
				coldest++;
			}
		}
	}
	return B;
}
void removerow2(Mat* A, Mat* B, int r) {
	int rowdest = 0;
	for (int i = 0; i < A->row; i++) {
		if (i != r)
		{
			for (int j = 0; j < A->col; j++)
				B->d[rowdest][j] = A->d[i][j];

			rowdest++;
		}
	}
	return;
}
void removecol2(Mat* A, Mat* B, int c) {
	int coldest = 0;
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			if (j != c) {
				B->d[i][coldest] = A->d[i][j];
				coldest++;
			}
		}
	}
	return;
}




CMat* cremoverow(CMat* A, int r) {
	CMat* B = newcmatnoinit(A->row - 1, A->col);

	int rowdest = 0;
	for (int i = 0; i < A->row; i++) {
		if (i != r)
		{
			for (int j = 0; j < A->col; j++)
				B->d[rowdest][j] = A->d[i][j];

			rowdest++;
		}
	}
	return B;
}
CMat* cremovecol(CMat* A, int c) {
	CMat* B = newcmatnoinit(A->row, A->col - 1);
	for (int i = 0; i < A->row; i++) {
		int coldest = 0;
		for (int j = 0; j < A->col; j++) {
			if (j != c) {
				B->d[i][coldest] = A->d[i][j];
				coldest++;
			}
		}
	}
	return B;
}
void cremoverow2(CMat* A, CMat* B, int r) {
	int rowdest = 0;
	for (int i = 0; i < A->row; i++) {
		if (i != r)
		{
			for (int j = 0; j < A->col; j++)
				B->d[rowdest][j] = A->d[i][j];

			rowdest++;
		}
	}
	return;
}
void cremovecol2(CMat* A, CMat* B, int c) {
	for (int i = 0; i < A->row; i++) {
		int coldest = 0;
		for (int j = 0; j < A->col; j++) {
			if (j != c) {
				B->d[i][coldest] = A->d[i][j];
				coldest++;
			}
		}
	}
	return;
}



Mat* transpose(Mat* A) {
	Mat* B = newmatnoinit(A->col, A->row);
	int k = 0;
	for (int i = 0; i < A->col; i++) {
		for (int j = 0; j < A->row; j++, k++) {
			B->d[j][i] = A->d[i][j];
		}
	}
	return B;
}

CMat* ctranspose(CMat* A) {
	CMat* B = newcmatnoinit(A->col, A->row);
	int k = 0;
	for (int i = 0; i < A->col; i++) {
		for (int j = 0; j < A->row; j++, k++) {
			B->d[j][i] = A->d[i][j];
		}
	}
	return B;
}




double det(Mat* M) {
	int r = M->row;
	int c = M->col;
	if (r == 1 && c == 1) {
		double d = M->d[0][0];
		return d;
	}
	Mat* M1 = removerow(M, 1);
	Mat* M2 = newmatnoinit(M->row - 1, M->col - 1);
	double d = 0, si = +1;
	for (int j = 0; j < M->col; j++) {
		double cx = M->d[0][j];
		removecol2(M1, M2, j);
		d += si * det(M2) * cx;
		si *= -1;
	}
	freemat(M1);
	freemat(M2);
	return d;
}


cplx cdet(CMat* M) {
	int r = M->row;
	int c = M->col;
	if (r == 1 && c == 1) {
		cplx d = M->d[0][0];
		return d;
	}
	CMat* M1 = cremoverow(M, 0);
	CMat* M2 = newcmatnoinit(M->row - 1, M->col - 1);
	cplx d; double si = +1;
	d.re = d.im = 0;

	for (int j = 0; j < M->col; j++) {
		cplx cx = M->d[0][j];
		cremovecol2(M1, M2, j);
		cplx d2 = cmultdo(cmultco(cdet(M2), cx), si);
		caddc(&d, d, d2);
		si = -si;
	}
	freecmat(M1);
	freecmat(M2);
	return d;
}


double trace(Mat* A) {
	double d = 0;
	for (int i = 0; i < A->row; i++) {
		d += A->d[i][i];
	}
	return d;
}


cplx ctrace(CMat* A) {
	cplx d;
	d.re = d.im = 0;
	for (int i = 0; i < A->row; i++) {
		d=caddco(d, A->d[i][i]);
	}
	return d;
}


Mat* adjoint(Mat* A) {
	Mat* B = newmatnoinit(A->row, A->col);
	Mat* A1 = newmatnoinit(A->row - 1, A->col);
	Mat* A2 = newmatnoinit(A->row - 1, A->col - 1);
	for (int i = 0; i < A->row; i++) {
		removerow2(A, A1, i);
		for (int j = 0; j < A->col; j++) {
			removecol2(A1, A2, j);
			double si = (i + j) % 2 ? -1.0 : 1.0;
			B->d[i][j] = det(A2) * si;
		}
	}
	Mat* C = transpose(B);
	freemat(A1);
	freemat(A2);
	freemat(B);
	return C;
}



extern CMat* ctransposeconj(CMat* source)
{
	CMat* dest = newcmatnoinit(source->col, source->row);
	for (int i = 0; i < dest->row; i++)
		for (int j = 0; j < dest->col; j++)
			dest->d[i][j] = conju(source->d[j][i]);
	return dest;
}

CMat* cadjoint(CMat* A) {
	CMat* B = newcmatnoinit(A->row, A->col);
	CMat* A1 = newcmatnoinit(A->row - 1, A->col);
	CMat* A2 = newcmatnoinit(A->row - 1, A->col - 1);
	for (int i = 0; i < A->row; i++) {
		cremoverow2(A, A1, i);
		for (int j = 0; j < A->col; j++) {
			cremovecol2(A1, A2, j);
			double si = (i + j) % 2 ? -1.0 : 1.0;
			B->d[i][j] = cmultdo( cdet(A2) , si);
		}
	}
	CMat* C = ctranspose(B);
	freecmat(A1);
	freecmat(A2);
	freecmat(B);
	return C;
}

Mat* inverse(Mat* A) {
	Mat* B = adjoint(A);
	double de = det(A);
	Mat* C = scalarmultiply(B, 1 / de);
	freemat(B);
	return C;
}

CMat* cinverse(CMat* A) {
	CMat* B = cadjoint(A);
	cplx de = cinv(cdet(A));

	CMat* C = complexmultiply(B, de);
	freecmat(B);
	return C;
}


void cinversedest(CMat* A, CMat* dest) {
	CMat* B = cadjoint(A);
	cplx de = cinv(cdet(A));

	complexmultiplydest(B, de, dest);
	freecmat(B);
	return;
}

	

Mat* copyvalue(Mat* A) {
	Mat* B = newmatnoinit(A->row, A->col);

	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			B->d[i][j] = A->d[i][j];
		}
	}
	return B;
}


CMat* copycvalue(CMat* A) {
	CMat* B = newcmatnoinit(A->row, A->col);

	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			B->d[i][j] = A->d[i][j];
		}
	}
	return B;
}

int copycvaluedest(CMat* A,CMat* dest) {

	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			dest->d[i][j] = A->d[i][j];
		}
	}
	return (OK);
}


Mat* triinverse(Mat* A) {
	Mat* B = newmatnoinit(A->row, A->col);
	for (int i = 0; i < B->row; i++) {
		for (int j = i; j < B->col; j++) {
			if (i == j) {
				B->d[i][j] = 1 / A->d[i][j];
			}
			else {
				B->d[i][j] = -A->d[i][j] / A->d[j][j];
			}
		}
	}
	return B;
}

CMat* ctriinverse(CMat* A) {
	CMat* B = newcmatnoinit(A->row, A->col);
	for (int i = 0; i < B->row; i++) {
		for (int j = i; j < B->col; j++) {
			if (i == j) {
				B->d[i][j] = cinv(A->d[i][j]);
			}
			else {
				B->d[i][j] = cmultdo( cmultco(A->d[i][j], cinv( A->d[j][j])), -1.0);
			}
		}
	}
	return B;
}


Mat* rowechelon(Mat* A) {
	if (A->row == 1) {
		for (int j = 0; j < A->col; j++) {
			if (A->d[j] != 0) {
				Mat* B = scalarmultiply(A, 1 / A->d[0][j]);
				return B;
			}
		}
		Mat* B = newmat(1, A->col, 0);
		return B;
	}


	Mat* B = copyvalue(A);
	int ind1 = B->col;
	int ind2 = 0;
	for (int i = 0; i < B->row; i++) {
		for (int j = 0; j < B->col; j++) {
			if (B->d[i][j] != 0 && j < ind1) {
				ind1 = j;
				ind2 = i;
				break;
			}
		}
	}//TODOHERE
	// Swap columns
	if (ind2 > 0) {
		for (int j = 0; j < B->col; j++) {
			double temp = B->d[0][j];
			B->d[0][j] = B->d[ind2][j];
			B->d[ind2][j] = temp;
		}
	}

	if (B->d[0][0] != 0) {
		double coeff = B->d[0][0];
		for (int j = 0; j < B->col; j++) {
			B->d[0][j] /= coeff;
		}

		for (int i = 1; i < B->row; i++) {
			coeff = B->d[i][0];

			for (int j = 0; j < B->col; j++) {
				B->d[i][j] -= coeff * B->d[0][j];
			}
		}
	}
	else {
		double coeff = 0;
		for (int j = 0; j < B->col; j++) {
			if (B->d[0][j] != 0 && coeff == 0) {
				coeff = B->d[0][j];
				B->d[0][j] = 1;
			}
			else if (B->d[0][j] != 0) {
				B->d[0][j] /= coeff;
			}
		}
	}

	Mat* B1 = removerow(B, 1);
	Mat* B2 = removecol(B1, 1);
	Mat* Be = rowechelon(B2);

	for (int i = 0, iB=1; i <= Be->row; i++, iB++) {
		for (int j = 0, jB=1; j <= Be->col; j++, jB++) {

			B->d[iB][jB] = Be->d[i][j];
		}
	}
	freemat(B1);
	freemat(B2);
	freemat(Be);
	return B;
}



CMat* crowechelon(CMat* A) {
	if (A->row == 1) {
		for (int j = 0; j < A->col; j++) {
			if (cmodinv(A->d[0][j]) != 0) {
				CMat* B = complexmultiply(A,cinv(A->d[0][j]));
				return B;
			}
		}
		CMat* B = newcmat(1, A->col, 0,0);
		return B;
	}


	CMat* B = copycvalue(A);
	int ind1 = B->col;
	int ind2 = 0;

	for (int i = 0; i < B->row; i++) {
		for (int j = 0; j < B->col; j++) {
			if (!ciszero(B->d[i][j]) && j < ind1) {
				ind1 = j;
				ind2 = i;
				break;
			}
		}
	}
	// Swap columns
	if (ind2 > 0) {
		for (int j = 0; j < B->col; j++) {
			cplx temp = B->d[0][j];
			B->d[0][j] = B->d[ind2][j];
			B->d[ind2][j] = temp;
		}
	}

	if (!ciszero(B->d[0][0])) {
		cplx coeff = cinv(B->d[0][0]);
		for (int j = 0; j < B->col; j++) {
			B->d[0][j] = cmultco(B->d[0][j], coeff);
		}

		for (int i = 1; i < B->row; i++) {
			coeff = B->d[i][0];

			for (int j = 0; j < B->col; j++) {
				B->d[i][j] = csubco(B->d[i][j], cmultco(B->d[0][j], coeff));
			}
		}
	}
	else {
		cplx coeff; coeff.re = coeff.im = 0.0;
		for (int j = 0; j < B->col; j++) {
			if (!ciszero(B->d[0][j]) && ciszero(coeff)) {
				coeff = cinv(B->d[0][j]);
				B->d[0][j].re = 1; B->d[0][j].im = 0.0;
			}
			else 
				B->d[0][j] = cmultco(B->d[0][j], coeff);
		}
	}

	CMat* B1 = cremoverow(B, 0);
	CMat* B2 = cremovecol(B1, 0);
	CMat* Be = crowechelon(B2);

	for (int i = 0, iB = 1; i <= Be->row; i++, iB++) {
		for (int j = 0, jB = 1; j <= Be->col; j++, jB++) {

			B->d[iB][jB] = Be->d[i][j];
		}
	}
	freecmat(B1);
	freecmat(B2);
	freecmat(Be);
	return B;
}


Mat* hconcat(Mat* A, Mat* B) {
	Mat* C = newmatnoinit(A->row, A->col + B->col);
	for (int i = 0; i < A->row; i++) {
		int k = 0;
		for (int j = 0; j < A->col; j++, k++) {
			C->d[i][k] = A->d[i][j];
		}
		for (int j = 0; j < B->col; j++, k++) {
			C->d[i][k] = B->d[i][j];
		}
	}
	return C;
}

CMat* chconcat(CMat* A, CMat* B) {
	CMat* C = newcmatnoinit(A->row, A->col + B->col);

	for (int i = 0; i < A->row; i++) {
		int k = 0;
		for (int j = 0; j < A->col; j++, k++) {
			C->d[i][k] = A->d[i][j];
		}
		for (int j = 0; j < B->col; j++, k++) {
			C->d[i][k] = B->d[i][j];
		}
	}
	return C;
}
Mat* vconcat(Mat* A, Mat* B) {
	Mat* C = newmatnoinit(A->row + B->row, A->col);
	int k = 0;
	for (int i = 0; i < A->row; i++, k++) {
		for (int j = 0; j < A->col; j++) {
			C->d[k][j] = A->d[i][j];
		}
	}
	for (int i = 0; i < B->row; i++, k++) {
		for (int j = 0; j < B->col; j++) {
			C->d[k][j] = B->d[i][j];
		}
	}
	return C;
}

CMat* cvconcat(CMat* A, CMat* B) {
	CMat* C = newcmatnoinit(A->row + B->row, A->col);
	int k = 0;
	for (int i = 0; i < A->row; i++, k++) {
		for (int j = 0; j < A->col; j++) {
			C->d[k][j] = A->d[i][j];
		}
	}
	for (int i = 0; i < B->row; i++, k++) {
		for (int j = 0; j < B->col; j++) {
			C->d[k][j] = B->d[i][j];
		}
	}
	return C;
}

double norm(Mat* A) {
	double d = 0;

	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			double d0 = A->d[i][j];
			d += d0*d0;
		}
	}
	d = sqrt(d);
	return d;
}

double cnorm(CMat* A) {
	double d = 0;
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			d += cmodinv(A->d[i][j]);
		}
	}
	d = sqrt(d);
	return d;
}


Mat* nullmat(Mat* A) {
	Mat* RM = rowechelon(A);
	int k = RM->row;
	for (int i = RM->row-1; i >= 0; i--) {
		bool flag = FALSE;
		for (int j = 0; j < RM->col; j++) {
			if (RM->d[i][j] != 0) {
				flag = TRUE;
				break;
			}
		}
		if (flag) {
			k = i;
			break;
		}
	}
	Mat* RRM = submat(RM, 0, k-1, 0, RM->col-1);
	freemat(RM);
	int nn = RRM->col - RRM->row;
	if (nn == 0) {
		Mat* N = newmat(0, 0, 0);
		return N;
	}
	Mat* R1 = submat(RRM, 0, RRM->row-1, 0, RRM->row-1);
	Mat* R2 = submat(RRM, 0, RRM->row-1, RRM->row, RRM->col-1);
	freemat(RRM);
	Mat* I = eye(nn);
	Mat* T1 = multiply(R2, I);
	freemat(R2);
	Mat* R3 = scalarmultiply(T1, -1);
	freemat(T1);
	Mat* T2 = triinverse(R1);
	freemat(R1);
	Mat* X = multiply(T2, R3);
	freemat(T2);
	freemat(R3);
	Mat* N = vconcat(X, I);
	freemat(I);
	freemat(X);
	for (int j = 0; j < N->col; j++) {
		double de = 0;
		for (int i = 0; i < N->row; i++) {
			de += N->d[i][j] * N->d[i][j];
		}
		de = sqrt(de);
		for (int i = 1; i <= N->row; i++) {
			N->d[i][j] /= de;
		}

	}
	return N;
}

MatList* lu(Mat* A) {
	if (A->row == 1) {
		MatList* ml = (MatList*)tmalloc(sizeof(MatList));
		ml->mat = newmat(1, 1, A->d[0][0]);
		ml->next = (MatList*)tmalloc(sizeof(MatList));
		ml->next->mat = newmat(1, 1, 1);
		return ml;
	}
	double a = A->d[0][0];
	double c = 0;
	if (a != 0) {
		c = 1 / a;
	}
	Mat* w = submat(A, 0, 0, 1, A->col-1);
	Mat* v = submat(A, 1, A->row-1, 0, 0);
	Mat* Ab = submat(A, 1, A->row-1, 1, A->col-1);
	Mat* T1 = multiply(v, w);
	Mat* T2 = scalarmultiply(T1, -c);
	Mat* T3 = sum(Ab, T2);
	MatList* mlb = lu(T3);
	freemat(T1);
	freemat(T2);
	freemat(T3);
	freemat(Ab);
	Mat* L = newmat(A->row, A->col, 0);
	Mat* U = newmat(A->row, A->col, 0);
	for (int i = 0; i < A->row; i++) {
		for (int j = 0; j < A->col; j++) {
			if (i == 0 && j == 0) {
				L->d[i][j] = 1;
				U->d[i][j] = a;
			}
			else if (i == 0 && j > 0) {
				U->d[i][j] = w->d[0][j - 1];
			}
			else if (i > 0 && j == 0) {
				L->d[i][j] = c * v->d[0][i - 1];
			}
			else {
				L->d[i][j] = mlb->mat->d[i-1][j-1];
				U->d[i][j] = mlb->next->mat->d[i-1][j-1];
			}
		}
	}
	MatList* ml = (MatList*)tmalloc(sizeof(MatList));
	ml->mat = L;
	ml->next = (MatList*)tmalloc(sizeof(MatList));;
	ml->next->mat = U;
	freemat(w);
	freemat(v);
	tfree(mlb);
	return ml;
}

double innermultiply(Mat* a, Mat* b) {
	double d = 0;
	int n = a->row;
	if (a->col > n) {
		n = a->col;
	}
	for (int i = 0; i <= n; i++) {
		d += a->d[0][i] * b->d[0][i];
	}
	return d;
}

MatList* qr(Mat* A) {
	int r = A->row;
	int c = A->col;
	Mat* Q = newmat(r, r, 0);
	Mat* R = newmat(r, c, 0);
	Mat* ek = newmat(r, 1, 0);
	Mat* uj = newmat(r, 1, 0);
	Mat* aj = newmat(r, 1, 0);

	for (int j = 0; j < r; j++) {
		submat2(A, aj, 0, r-1, j, j);
		for (int k = 0; k < r; k++) {
			uj->d[0][k] = aj->d[0][k];
		}
		for (int k = 0; k < j - 1; k++) {
			submat2(Q, ek, 0, r-1, k, k);
			double proj = innermultiply(aj, ek);
			for (int l = 0; l < ek->row; l++) {
				ek->d[0][l] *= proj;

			}
			uj = minus(uj, ek);
		}
		double nuj = norm(uj);
		for (int i = 0; i < r; i++) {
			Q->d[i][j] = uj->d[0][i] / nuj;
		}
		for (int j1 = j-1; j1 < c; j1++) {
			R->d[j][j1] = innermultiply(uj, submat(A, 0, r-1, j1, j1)) / nuj;
		}
	}
	MatList* ml = (MatList*)tmalloc(sizeof(MatList));
	ml->mat = Q;
	ml->next = (MatList*)tmalloc(sizeof(MatList));;
	ml->next->mat = R;
	freemat(ek);
	freemat(uj);
	freemat(aj);
	return ml;
}

