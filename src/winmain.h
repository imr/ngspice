/* Forwards and external function declarations
   for winmain.c
*/

/* Forward definition of main() */
int xmain(int argc, char *argv[]);

/* forward of Update function */
#ifdef __CYGWIN__
static char* rlead(char*);
#endif

void winmessage(char*);

void WaitForIdle(void);
static void WaitForMessage(void);
static void ClearInput(void);
void SetSource(char *Name);
void SetAnalyse(char *, int);
static void AdjustScroller(void);
static void _DeleteFirstLine(void);
static void AppendChar(char c);
static void AppendString(const char *Line);
static void DisplayText(void);

static int w_getch(void);
static int w_putch(int c);

static void Main_OnSize(HWND hwnd, UINT state, int cx, int cy);
static void PostSpiceCommand(const char *const cmd);
static LRESULT CALLBACK MainWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK StringWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK TextWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static void Element_OnPaint(HWND hwnd);
static LRESULT CALLBACK ElementWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static int MakeArgcArgv(char *cmdline, int *argc, char ***argv);

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
int    win_x_read(int fd, char *buf, int n);
int    win_x_getc(FILE *fp);
int    win_x_getchar(void);
int    win_x_putchar(const int c);
int    win_x_putc(const int c, FILE *fp);
int    win_x_feof(FILE *fp);
int    win_x_ferror(FILE *fp);
int    win_x_fputchar(int c);
