#!/bin/sh
# Configuration script for ngspice. 
#
# This script performs initial configuration of ngspice source 
# package.
#
#
# $Id$
#
# temp-adms.ac: modified configure.ac if --adms is selected
# for temporary use by autoconf, will be deleted automatically
# configure.ac stays untouched

PROJECT=ngspice
TEST_TYPE=-f
FILE=DEVICES

# ADMS variables

ADMSDIR=src/spicelib/devices/adms
XMLPATH=src/spicelib/devices/adms/admst
ADMSXML=admsXml
ADMS=0

# Exit variable
DIE=0


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

end_on_error()
{
if test "$ADMS" -eq 1; then
#  cp -p temp-adms.ac configure.err
  rm -f temp-adms.ac
fi

exit 1
}

check_awk()
{
(awk --version) < /dev/null > /dev/null 2>&1 || {
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

(libtoolize --version) < /dev/null > /dev/null 2>&1 || {
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
(admsXml --version)   < /dev/null > /dev/null 2>&1 || {
        echo
	echo "You must have admsXml installed to compile adms models."
	echo "See http://mot-adms.sourceforge.net"
	echo "(newest stable release is recommended)"
        DIE=1
}
}

case "$1" in
    "--adms" | "-a")
    check_adms 
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

if test "$DIE" -eq 1; then
	exit 1
fi

test $TEST_TYPE $FILE || {
	echo "You must run this script in the top-level $PROJECT directory"
	exit 1
}

# only for --adms:
if test "$ADMS" -eq 1; then

check_awk

#  sed 's/tests\/vbic\/Makefile/tests\/vbic\/Makefile\
#                src\/spicelib\/devices\/adms\/ekv\/Makefile\
#               src\/spicelib\/devices\/adms\/hicum0\/Makefile\
#                 src\/spicelib\/devices\/adms\/hicum2\/Makefile\
#                 src\/spicelib\/devices\/adms\/mextram\/Makefile\
#                 src\/spicelib\/devices\/adms\/psp102\/Makefile/g' configure.temp >configure.ac
  
  # automake and autoconf need these entries in configure.ac for adms enabled  
  z=""
  znew=""
  # Find all lines with "#VLAMKF" and put the second token of each line into shell variable z
  # as input to additional automake call for the adms directories
  z=`cat configure.ac | awk -v z=${z} '$1 ~ /#VLAMKF/{ z=$2; print "./"z }' `
  # same as above, sed requires \ at line endings, to be added to temp-adms.ac used by autoconf
  znew=`cat configure.ac | awk -v z=${znew} '$1 ~ /#VLAMKF/{ znew=$2; print "             "znew"\\\" }' `
 
 # Find "tests/vbic/Makefile" and replace by tests/vbic/Makefile plus contents of variable z
  sed -e "
  s,tests\\/vbic\\/Makefile,tests\\/vbic\\/Makefile\\
  $znew ," configure.ac >temp-adms.ac
  
  currentdir=`pwd`
  
  for adms_dir in `ls $ADMSDIR`
  do
    if [ -d "$ADMSDIR/$adms_dir" ]; then
     
     case "$adms_dir" in
        "CVS")
        echo "Skipping CVS" ;;
        
        "admst")
        echo "Skipping scripts dir" ;;
        
        *)
        echo "Entering into directory: $adms_dir"
        echo "-->"$ADMSDIR/$adms_dir
        cd $ADMSDIR/$adms_dir
        file=`ls admsva/*.va`
        $ADMSXML $file -Iadmsva -xv -e ../admst/ngspiceVersion.xml \
        -e ../admst/ngspiceMakefile.am.xml
        
        cd $currentdir
        ;;
     esac
    fi 
  done

fi

echo "Running aclocal $ACLOCAL_FLAGS"
aclocal $ACLOCAL_FLAGS
if [ $? -ne 0 ]; then  echo "aclocal failed"; end_on_error ; fi

echo "Running libtoolize"
libtoolize --copy --force
if [ $? -ne 0 ];then  echo "libtoolize failed"; end_on_error ; fi

# optional feature: autoheader
(autoheader --version)  < /dev/null > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "Running autoheader"
  autoheader
  if [ $? -ne 0 ]; then  echo "autoheader failed"; end_on_error ; fi
fi

echo "Running automake -Wall --copy --add-missing"
automake  -Wall --copy --add-missing
if [ $? -ne 0 ]; then  echo "automake failed"; end_on_error ; fi

if test "$ADMS" -eq 1; then
echo "Running automake for adms"
automake  -Wall --copy --add-missing $z
if [ $? -ne 0 ]; then  echo "automake failed"; end_on_error ; fi
fi

echo "Running autoconf"
if test "$ADMS" -eq 1; then
  autoconf temp-adms.ac > configure
  rm -f temp-adms.ac
else
autoconf
fi
if [ $? -ne 0 ]; then  echo "autoconf failed"; end_on_error ; fi

echo "Success."

exit 0
