/* I/O Redirection for Spice 3F4 under Win32s
 *      Autor: Wolfgang Muees
 *      Stand: 21.05.95
 */

#ifndef ngspice_WSTDIO_H
#define ngspice_WSTDIO_H

#include <stdio.h>                              /* original definitions */

#undef getc                                     /* old macros removed */
#undef putc
#undef ungetc
#undef getchar
#undef putchar
#undef feof
#undef ferror

/* -------------------------------<forwards>---------------------------------- */

int    win_x_fclose(FILE *stream);
int    win_x_fflush(FILE *stream);
int    win_x_fgetc(FILE *stream);
int    win_x_fgetpos(FILE *stream, fpos_t *pos);
char  *win_x_fgets(char *s, int n, FILE *stream);
int    win_x_fprintf(FILE *stream, const char *format, ...);
int    win_x_fputc(int c, FILE *stream);
int    win_x_fputs(const char *s, FILE *stream);
size_t win_x_fread(void *ptr, size_t size, size_t n, FILE *stream);
FILE  *win_x_freopen(const char *path, const char *mode, FILE *stream);
int    win_x_fscanf(FILE *stream, const char *format, ...);
int    win_x_fseek(FILE *stream, long offset, int whence);
int    win_x_fsetpos(FILE *stream, const fpos_t*pos);
long   win_x_ftell(FILE *stream);
size_t win_x_fwrite(const void *ptr, size_t size, size_t n, FILE *stream);
char  *win_x_gets(char *s);
void   win_x_perror(const char *s);
int    win_x_printf(const char *format, ...);
int    win_x_puts(const char *s);
int    win_x_scanf(const char *format, ...);
int    win_x_ungetc(int c, FILE *stream);
int    win_x_vfprintf(FILE *stream, const char *format, void *arglist);
/* int    win_x_vfscanf(FILE *stream, const char *format, void *arglist); */
int    win_x_vprintf(const char *format, void *arglist);
/* int    win_x_vscanf(const char *format, void *arglist); */

int    win_x_getc(FILE *fp);
int    win_x_getchar(void);
int    win_x_putchar(const int c);
int    win_x_putc(const int c, FILE *fp);
int    win_x_feof(FILE *fp);
int    win_x_ferror(FILE *fp);
int    win_x_fputchar(int c);

/* ------------------------------<New macros>--------------------------------- */

#define fclose          win_x_fclose
#define fflush          win_x_fflush
#define fgetc           win_x_fgetc
#define fgetpos         win_x_fgetpos
#define fgets           win_x_fgets
#define fprintf         win_x_fprintf
#define fputc           win_x_fputc
#define fputs           win_x_fputs
#define fread           win_x_fread
/* #define freopen      win_x_freopen    hvogt  10.05.2000 */
#define fscanf          win_x_fscanf
#define fseek           win_x_fseek
#define fsetpos         win_x_fsetpos
#define ftell           win_x_ftell
#define fwrite          win_x_fwrite
#define gets            win_x_gets
#define perror          win_x_perror
#define printf          win_x_printf
#define puts            win_x_puts
#define scanf           win_x_scanf
#define ungetc          win_x_ungetc
#define vfprintf        win_x_vfprintf
/* #define vfscanf         win_x_vfscanf */
#define vprintf         win_x_vprintf
/* #define vscanf          win_x_vscanf */
#define read            win_x_read
#define getc            win_x_getc
#define getchar         win_x_getchar
#define putchar         win_x_putchar
#define putc            win_x_putc
#define feof            win_x_feof
#define ferror          win_x_ferror
#define fputchar        win_x_fputchar

/* --------------------------------------------------------------------------- */

#endif
