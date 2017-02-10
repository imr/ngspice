/* I/O Redirection for Spice 3F4 under Win32s
	Autor: Wolfgang Muees
	Stand: 21.05.95
*/

#ifndef ngspice_WSTDIO_H
#define ngspice_WSTDIO_H
#include <stdio.h>				/* original definitions */

#undef getc					/* old macros removed */
#undef putc
#undef ungetc
#undef getchar
#undef putchar
#undef feof
#undef ferror

/* -------------------------------<forwards>----------------------------------*/

int	f_c_l_o_s_e( FILE * __stream);
int	f_f_l_u_s_h( FILE * __stream);
int	fg_e_t_c( FILE * __stream);
int	f_g_e_t_p_o_s( FILE * __stream, fpos_t * __pos);
char * fg_e_t_s(char * __s, int __n, FILE * __stream);
int    fp_r_i_n_t_f(FILE * __stream, const char * __format, ...);
int    fp_u_t_c(int __c, FILE * __stream);
int    fp_u_t_s(const char * __s, FILE * __stream);
size_t f_r_e_a_d(void * __ptr, size_t __size, size_t __n, FILE * __stream);
FILE * f_r_e_o_p_e_n(const char * __path, const char * __mode, FILE * __stream);
int    fs_c_a_n_f(FILE * __stream, const char * __format, ...);
int    f_s_e_e_k(FILE * __stream, long __offset, int __whence);
int    f_s_e_t_p_o_s(FILE * __stream, const fpos_t*__pos);
long   f_t_e_l_l(FILE * __stream);
size_t f_w_r_i_t_e(const void * __ptr, size_t __size, size_t __n, FILE * __stream);
char * g_e_t_s(char * __s);
void   p_e_r_r_o_r(const char * __s);
int    p_r_i_n_t_f(const char * __format, ...);
int    p_u_t_s(const char * __s);
int    s_c_a_n_f(const char * __format, ...);
int    ung_e_t_c(int __c, FILE * __stream);
int    vfp_r_i_n_t_f(FILE * __stream, const char * __format, void * __arglist);
/*int   vfs_c_a_n_f(FILE * __stream, const char * __format, void * __arglist);*/
int    vp_r_i_n_t_f(const char * __format, void * __arglist);
/*int   vs_c_a_n_f(const char * __format, void * __arglist); */
#ifdef _MSC_VER 
#if _MSC_VER < 1500
/* VC++ 6.0, VC++ 2005 */
_CRTIMP int __cdecl read(int fd,  void * __buf, unsigned int __n);
#else
/* VC++ 2008 */
_CRTIMP int __cdecl read(int fd, _Out_bytecap_(_MaxCharCount) void * __buf, _In_ unsigned int __n);
#endif
#else
int    r_e_a_d(int fd, char * __buf, int __n);
#endif
int    g_e_t_c(FILE * __fp);
int    g_e_t_char(void);
int    p_u_t_char(const int __c);
int    p_u_t_c(const int __c, FILE * __fp);
int    f_e_o_f(FILE * __fp);
int    f_e_r_r_o_r(FILE * __fp);
int    fp_u_t_char(int __c);

/* ------------------------------<New macros>---------------------------------*/

#define fclose		f_c_l_o_s_e
#define fflush 	f_f_l_u_s_h
#define fgetc  	fg_e_t_c
#define fgetpos	f_g_e_t_p_o_s
#define fgets		fg_e_t_s
#define fprintf	fp_r_i_n_t_f
#define fputc		fp_u_t_c
#define fputs		fp_u_t_s
#define fread		f_r_e_a_d
/* #define freopen	f_r_e_o_p_e_n    hvogt  10.05.2000 */
#define fscanf		fs_c_a_n_f
#define fseek		f_s_e_e_k
#define fsetpos	f_s_e_t_p_o_s
#define ftell		f_t_e_l_l
#define fwrite		f_w_r_i_t_e
#define gets		g_e_t_s
#define perror		p_e_r_r_o_r
#define printf		p_r_i_n_t_f
#define puts		p_u_t_s
#define scanf		s_c_a_n_f
#define ungetc		ung_e_t_c
#define vfprintf	vfp_r_i_n_t_f
/*#define vfscanf	vfs_c_a_n_f*/
#define vprintf	vp_r_i_n_t_f
/*#define vscanf	vs_c_a_n_f*/
#define read      r_e_a_d
#define getc		g_e_t_c
#define getchar	g_e_t_char
#define putchar	p_u_t_char
#define putc		p_u_t_c
#define feof		f_e_o_f
#define ferror		f_e_r_r_o_r
#define fputchar	fp_u_t_char


/*----------------------------------------------------------------------------*/

#endif

