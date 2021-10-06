#!/bin/sh
# Configuration script for ngspice.
#
# This script performs initial configuration of ngspice source
# package.
#
#
# temp-adms.ac: modified configure.ac if --adms is selected
# for temporary use by autoconf, will be deleted automatically
# configure.ac stays untouched

PROJECT=ngspice

# ADMS variables

ADMSDIR=src/spicelib/devices/adms
XMLPATH=src/spicelib/devices/adms/admst
ADMSXML=${ADMSXML:-admsXml}
ADMS=0

# Exit variable
DIE=0


# Check for Mac OS X
uname -a | grep -q "Darwin"
if [ $? -eq 0 ]; then
    LIBTOOLIZE=glibtoolize
else
    LIBTOOLIZE=libtoolize
fi

help()
{
    echo
    echo "$PROJECT autogen.sh help"
    echo
    echo "--adms     -a: enables adms feature"
    echo "--help     -h: print this file"
    echo "--version  -v: print version"
    echo
}

version()
{
    echo
    echo "$PROJECT autogen.sh 1.0"
    echo
}

error_and_exit()
{
    echo "Error: $1"
    if [ "$ADMS" -eq 1 ]; then
        rm -f temp-adms.ac
    fi
    exit 1
}

check_awk()
{
    (awk --version) < /dev/null > /dev/null 2>&1 ||
    (awk -W version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have awk installed to compile $PROJECT with --adms."
	exit 1
    }
}

check_autoconf()
{
    (autoconf --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have autoconf installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/automake/"
	echo "(newest stable release is recommended)"
	DIE=1
    }

    ($LIBTOOLIZE --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have libtool installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/libtool/"
	echo "(newest stable release is recommended)"
	DIE=1
    }

    (automake --version) < /dev/null > /dev/null 2>&1 || {
	echo
	echo "You must have automake installed to compile $PROJECT."
	echo "See http://www.gnu.org/software/automake/"
	echo "(newest stable release is recommended)"
	DIE=1
    }
}


check_adms()
{
    ($ADMSXML --version) < /dev/null > /dev/null 2>&1 || {
        echo
	echo "You must have admsXml installed to compile adms models."
	echo "See https://sourceforge.net/projects/mot-adms/"
	echo "(version 2.3.6, tested for ngspice under MINGW on MS Windows)"
        DIE=1
    }
}


# check if verilog-a files exist in every adms device directory
check_adms_va()
{
    echo
    # get the devices directories from configure.ac
    admsdirs=`awk '$1 ~ /#VLAMKF/ { print $2 }' < configure.ac`
    admsdirs=`echo $admsdirs | sed "s/\/Makefile//g"`

    for adms_dir in $admsdirs ; do
        FOK=0
        if [ -d "$adms_dir" ]; then
                    ls $adms_dir/admsva/*.va  > /dev/null 2>&1
                    exitcode=$?
                    if [ $exitcode -ne 0 ]; then
                       FOK=1
                    fi
        else
           FOK=1
        fi
        if [ "$FOK" -eq 1 ]; then
            echo "Error: No *.va file found in $adms_dir/admsva"
            echo "Please download patch file ng-adms-va.tar.gz from"
            echo "http://ngspice.sourceforge.net/experimental/ngspice-adms-va.7z"
            echo "and expand it into the ngspice directory"
            echo
            DIE=1
        fi
    done
}

case "$1" in
    "--adms" | "-a")
        check_adms
        check_adms_va
        ADMS=1
        ;;

    "--help" | "-h")
        help
        exit 0
        ;;

    "--version" | "-v")
        version
        exit 0
        ;;

    *)
        ;;
esac


check_autoconf

if [ "$DIE" -eq 1 ]; then
    exit 1
fi

[ -f "DEVICES" ] || {
    echo "You must run this script in the top-level $PROJECT directory"
    exit 1
}

# only for --adms:
if [ "$ADMS" -gt 0 ]; then

    check_awk

    # add adms related Makefile entries to a configure.ac style file for
    #   autoconf and automake

    # Find all lines with "#VLAMKF" and put the second token of each line
    #   into a shell variable
    adms_Makefiles=`awk '$1 ~ /#VLAMKF/ { print "./" $2 }' < configure.ac`

    # just the same, but escape newlines with '\' for the following sed expression
    znew=`awk '$1 ~ /#VLAMKF/ { print " " $2 "\\\\" }' < configure.ac`

    # Find "tests/vbic/Makefile" and insert the list of Makefiles
    # some sed's fail to process the '\n' escape on the RHS,
    #   thus use an escaped plain newline
    sed \
        -e "s,tests\\/vbic\\/Makefile,&\\
$znew
 ," \
        configure.ac > temp-adms.ac

    for adms_dir in `ls $ADMSDIR` ; do
        if [ -d "$ADMSDIR/$adms_dir" ]; then

            case "$adms_dir" in

                "admst")
#                    echo "Skipping admst dir"
                    ;;

                *)
                    echo "Entering into directory: $adms_dir"
                    echo "-->"$ADMSDIR/$adms_dir
                    (
                        cd $ADMSDIR/$adms_dir
                        $ADMSXML `ls admsva/*.va` -Iadmsva -xv -x \
                            -e ../admst/ngspiceVersion.xml \
                            -e ../admst/ngspiceMakefile.am.xml
                    )
                    ;;
            esac
        fi
    done

fi

echo "Running $LIBTOOLIZE"
$LIBTOOLIZE --copy --force \
    || error_and_exit "$LIBTOOLIZE failed"

echo "Running aclocal $ACLOCAL_FLAGS"
aclocal $ACLOCAL_FLAGS --force -I m4 \
    || error_and_exit "aclocal failed"

# optional feature: autoheader
(autoheader --version) < /dev/null > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Running autoheader"
  autoheader --force \
      || error_and_exit "autoheader failed"
fi

echo "Running automake -Wall --copy --add-missing"
automake  -Wall --copy --add-missing \
    || error_and_exit "automake failed"

if [ "$ADMS" -gt 0 ]; then
    echo "Running automake for adms"
    automake  -Wall --copy --add-missing $adms_Makefiles \
        || error_and_exit "automake failed"
fi

echo "Running autoconf"
if [ "$ADMS" -gt 0 ]; then
    autoconf --force temp-adms.ac > configure \
        || error_and_exit "autoconf failed, with adms"
    rm -f temp-adms.ac
    chmod +x configure
else
    autoconf --force \
        || error_and_exit "autoconf failed"
fi

echo "Success."
exit 0
