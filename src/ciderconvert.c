#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static unsigned int numvars = 0;
static int numdims = 1;
static int numpoints = -1;
static int xdim = -1;
static int ydim = -1;

static void reset(void) {
    numvars = 0;
    numdims = 1;
    numpoints = -1;
    xdim = -1;
    ydim = -1;
}

static int matched(char *tomatch, char *data)
{
    return !strncmp(tomatch, data, strlen(tomatch));
}

static int convert_bin(FILE *fp, int ndim, unsigned int numv, int nump, int xn, int yn) {
    double *dbuf = 0;
    size_t sizeb = numv * sizeof(double);
    size_t nread;
    int currpt = 0;
    unsigned int idx;
    int retval = 0;
    int currx = 1, curry = 1;

    dbuf = malloc(sizeb);
    if (!dbuf) {
        return 1;
    }
    printf("Values:\n");
    while ((nread = fread(dbuf, sizeb, 1, fp)) == 1) {
        if (ndim == 2) {
            currpt++;
            if (curry > yn) {
                curry = 1;
                currx++;
            }
            if (currx > xn) {
                fprintf(stderr, "ERROR currx %d > xn %d\n", currx, xn);
                free(dbuf);
                return 2;
            }
            for (idx = 0; idx < numv; idx++) {
                if (idx == 0) {
                    printf("%d %d", curry, currx);
                }
                printf("\t%e\n", dbuf[idx]);
            }
            curry++;
        } else if (ndim == 1) {
            for (idx = 0; idx < numv; idx++) {
                if (idx == 0) {
                    printf("%d", currpt);
                }
                printf("\t%e\n", dbuf[idx]);
            }
            currpt++;
        }
    }
    if (currpt != nump) {
        fprintf(stderr, "ERROR currpt %d != nump %d\n", currpt, nump);
        retval = 3;
    }
    if (dbuf) {
        free(dbuf);
    }
    return retval;
}

static int convert_file(FILE *fp)
{
    int retval = 0, num = 0, found_binary = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t nread;
    char str1[256];
    char str2[256];

    while ((nread = getline(&line, &len, fp)) != -1) {
        if (matched("Binary:", line)) {
            found_binary = 1;
            break;
        }
        if (matched("Title:", line)) {
            reset();
        } else if (matched("No. Variables:", line)) {
            num = sscanf(line, "%s %s %d", str1, str2, &numvars);
            if (num != 3) {
                printf("num %d\n", num);
                printf("???? %s", line);
                return 1;
            }
        } else if (matched("No. Points:", line)) {
            num = sscanf(line, "%s %s %d", str1, str2, &numpoints);
            if (num != 3) {
                printf("num %d\n", num);
                printf("???? %s", line);
                return 2;
            }
        } else if (matched("Dimensions:", line)) {
            num = sscanf(line, "%s %d,%d", str1, &xdim, &ydim);
            if (num != 3) {
                printf("num %d\n", num);
                printf("???? %s", line);
                return 3;
            }
            numdims = 2;
        }
        printf("%s", line);
    }
    if (found_binary) {
        if (numdims == 1) {
            if (numvars > 0 && numpoints > 0) {
                retval = convert_bin(fp,numdims,numvars,numpoints,0,0);
            }
        } else if (numdims == 2) {
            if (numvars > 0 && xdim > 0 && ydim > 0
            && (numpoints == xdim * ydim)) {
                retval = convert_bin(fp,numdims,numvars,numpoints,xdim,ydim);
            }
        }
    }
    reset();
    return retval;
}

static void usage(char *prog)
{
    printf("Usage: %s -help | cider_save_data_file\n", prog);
    printf("  Converts the input binary cider_save_data_file to ascii.\n");
    printf("  If the input file is already converted to ascii, it is copied.\n");
    printf("  The output is written to stdout.\n");
}

int main(int argc, char** argv)
{
    FILE *fp = NULL;
    int retval = 0;

    if (argc < 2) {
        printf("No Cider save file name!\n");
        usage(argv[0]);
        return 1;
    } else if (argc == 2) {
        if (!strcmp(argv[1], "-help")) {
            usage(argv[0]);
            return 1;
        } else {
            fp = fopen(argv[1], "rb");
            if (fp != NULL) {
                retval = convert_file(fp);
            } else {
                printf("Cannot open %s\n",argv[1]);
                return 1;
            }
            fclose(fp);
        }
    } else {
        usage(argv[0]);
        return 1;
    }
    return retval;
}
