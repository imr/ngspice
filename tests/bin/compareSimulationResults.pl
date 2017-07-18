#!/bin/sh
eval 'exec perl -S -x -w $0 ${1+"$@"}'
#!perl

#
# compareSimulationResults.pl: program to do a toleranced comparison of compact model simulation results
#
#  Rel  Date            Who             Comments
# ====  ==========      =============   ========
#  1.0  04/13/06        Colin McAndrew  Initial version
#

sub usage() {
    print "
$prog: compare simulation results between two files

Usage: $prog [options] refFile simFile

Files:
    refFile                reference results file
    simFile                simulated results file

Options:
    -c CLIP                match numbers n1 and n2 with abs(n1)<CLIP and abs(n2)<CLIP
    -n NDIGIT              match numbers n1 and n2 if they are within 1 of the NDIGITth digit
    -r REL                 match any numbers n1 and n2 with abs(n1-n2)/(0.5*(abs(n1)+abs(n2)+abs(n1-n2)))<REL
    -d                     debug mode
    -h                     print this help message
    -i                     print info on file formats and structure
    -v                     verbose mode
";
} # End of usage

sub info() {
    print "
This program numerically compares simulation results and returns a string
that indicates the result of the comparision. Possible return values are:
ERROR: cannot open file nameOfFile
FAIL        (probably from some simulation failure)
FAIL        (simulation output quantities differ)
FAIL        (number of results is different)
FAIL        (non-numeric results)
DIFFER      (max rel error is relErr)
MATCH       (within specified tolerances)
MATCH       (exact)

It is expected that each file is a columnar list of simulation results,
with the first line being a title line that indicates column contents,
and every other line being numerical simulation results.

The comparisons are done in the order
exact
clip
nDigits
relErr
and passing one test means the results are considered to be the same.

Different tolerancing should be used for different types of simulations,
and because of this mixing different types of simulation results in
the one file is not recommended. Reasonable tolerances are:
Quantity       Clip        nDigtis     relTol
DC current     1.0e-13     6           1.0e-06
AC conductance 1.0e-20     6           1.0e-06
   capacitance 1.0e-20     6           1.0e-06
noise          1.0e-30     5           1.0e-05
Note that these numbers should be adjusted based on the precision
to which the numbers to be compared are printed.

If no tolerances are specified, then the refFile name must contain
one of the strings \"Dc\", \"Ac\", or \"Noise\" and the above values
are used as default tolerances.

The UNIX utilities spiff and ndiff so toleranced numerical comparisons,
but they seem not generally available now. Hence this simplified
numerical comparison program is provided for verifying simulation
results against reference test results. Also, clipping and comparison
to a certain number of digits are more relevant for comparison of
numbers printed in output (as compared to results held in memory),
and these are not considered in other toleranced numerical comparisons.
";
} # End of info

#
# Set program names and variables
#

$\="\n";
$,=" ";
$debug=0;
$verbose=0;
@prog=split("/",$0);
$prog=$prog[$#prog];
$number='[+-]?\d+[\.]?\d*[eE][+-]?\d+|[+-]?[\.]\d+[eE][+-]?\d+|[+-]?\d+[\.]?\d*|[+-]?[\.]\d+';

for (;;) {
    if (!defined($ARGV[0])) {
        last;
    } elsif ($ARGV[0]  =~ /^-c/i) {
        shift(@ARGV);
        die("ERROR: no clip value specified for -c option, stopped") if ($#ARGV < 0);
        $clip=$ARGV[0];
        die("ERROR: clip must be a positive number, stopped") if ($clip !~ /^$number$/ || $clip <= 0);
    } elsif ($ARGV[0]  =~ /^-n/i) {
        shift(@ARGV);
        die("ERROR: no number of digits value specified for -n option, stopped") if ($#ARGV < 0);
        $nDigits=$ARGV[0];
        die("ERROR: nDigits must be a positive integer, stopped") if ($nDigits !~ /^[1-9][0-9]*$/);
    } elsif ($ARGV[0]  =~ /^-r/i) {
        shift(@ARGV);
        die("ERROR: no relTol value specified for -r option, stopped") if ($#ARGV < 0);
        $relTol=$ARGV[0];
        die("ERROR: relTol must be a positive number, stopped") if ($relTol !~ /^$number$/ || $relTol <= 0);
    } elsif ($ARGV[0]  =~ /^-d/i) {
        $debug=1;$verbose=1;
    } elsif ($ARGV[0] =~ /^-h/i) {
        &usage();exit(0);
    } elsif ($ARGV[0] =~ /^-i/i) {
        &usage();&info();exit(0);
    } elsif ($ARGV[0] =~ /^-v/i) {
        $verbose=1;
    } elsif ($ARGV[0] =~ /^-/) {
        &usage();
        die("ERROR: unknown flag $ARGV[0], stopped");
    } else {
        last;
    }
    shift(@ARGV);
}
if ($#ARGV<1) {
    &usage();exit(0);
}

if (!defined($clip)) {
    if ($ARGV[0] =~ /Dc/i) {
        $clip=1.0e-13;
    } elsif ($ARGV[0] =~ /Ac/i) {
        $clip=1.0e-20;
    } elsif ($ARGV[0] =~ /Noise/i) {
        $clip=1.0e-30;
    } else {
        die("ERROR: must specify -c CLIP value if file is not Dc, Ac, or noise, stopped");
    }
}
if (!defined($relTol)) {
    if ($ARGV[0] =~ /Dc/i) {
        $relTol=1.0e-6;
    } elsif ($ARGV[0] =~ /Ac/i) {
        $relTol=1.0e-6;
    } elsif ($ARGV[0] =~ /Noise/i) {
        $relTol=1.0e-5;
    } else {
        die("ERROR: must specify -r RELTOL value if file is not Dc, Ac, or noise, stopped");
    }
}
if (!defined($nDigits)) {
    if ($ARGV[0] =~ /Dc/i) {
        $nDigits=6;
    } elsif ($ARGV[0] =~ /Ac/i) {
        $nDigits=6;
    } elsif ($ARGV[0] =~ /Noise/i) {
        $nDigits=5;
    } else {
        die("ERROR: must specify -n NDIGITS value if file is not Dc, Ac, or noise, stopped");
    }
}

if ($ARGV[0] =~ /reference/i) {
    $reference="reference";
} else {
    ($reference=$ARGV[0])=~s/^.*\.//;
}
($variant=$ARGV[1])=~s/^.*\.//;
$result=&compareResults($ARGV[0],$ARGV[1],$clip,$nDigits,$relTol);
printf("     variant: %-20s(compared to: %-9s) %s\n",$variant,$reference,$result);

sub compareResults {
    use strict;
    my($refFile,$simFile,$clip,$nDigits,$relTol)=@_;
    my(@Ref,@Sim,$i,$j,$relErr,$maxRelErr,$absErr,$maxAbsErr);
    my(@RefRes,@SimRes,@ColNames,$matchType,$mag,$lo,$hi);

    return("ERROR: cannot open file $refFile") if (!open(IF,"$refFile"));
    while (<IF>) {chomp;push(@Ref,$_)}
    close(IF);
    return("ERROR: cannot open file $simFile") if (!open(IF,"$simFile"));
    while (<IF>) {chomp;push(@Sim,$_)}
    close(IF);
    if ($main::verbose && $#Ref != $#Sim) {print STDERR "Reference points: $#Ref\nSimulated points: $#Sim"}
    return("FAIL        (probably from some simulation failure)") if ($#Ref != $#Sim || $#Sim<1);
    if ($main::verbose && $Ref[0] ne $Sim[0]) {print STDERR "Reference quantities: $Ref[0]\nSimulated quantities: $Sim[0]"}
    return("FAIL        (simulation output quantities differ)")   if ($Ref[0] ne $Sim[0]);
    $maxAbsErr=0;$maxRelErr=0;$matchType=0;
    @ColNames=split(/\s+/,$Ref[0]);
    for ($j=1;$j<=$#Ref;++$j) {
        @RefRes=split(/\s+/,$Ref[$j]);
        @SimRes=split(/\s+/,$Sim[$j]);
        if ($main::verbose && $#RefRes != $#SimRes) {print STDERR "Line $j: Ref data: $#RefRes\tSim data: $#SimRes"}
        return("FAIL        (number of quantities simulated are different)") if ($#RefRes != $#SimRes);
        for ($i=1;$i<=$#RefRes;++$i) { # ignore first column, this is the sweep variable
            if ($RefRes[$i] !~ /^$main::number$/ || $SimRes[$i] !~ /^$main::number$/) {
                return("FAIL        (non-numeric results");
            }
            next if ($RefRes[$i] == $SimRes[$i]);
            $matchType=1 if ($matchType<1);
            next if (abs($RefRes[$i]) < $clip && abs($SimRes[$i]) < $clip);
            #next if (abs($RefRes[$i]) < $clip || abs($SimRes[$i]) < $clip);
            if ($RefRes[$i]*$SimRes[$i] <= 0.0) {
                $matchType=2 if ($matchType<2);
                $absErr=abs($RefRes[$i]-$SimRes[$i]);
                $relErr=$absErr/(0.5*(abs($RefRes[$i])+abs($SimRes[$i])+$absErr));
                $maxRelErr=$relErr if ($relErr > $maxRelErr);
                if ($main::verbose) {print STDERR $ColNames[$i],$RefRes[$i],$SimRes[$i],100*$relErr."\%"}
                next;
            }
            $lo=abs($RefRes[$i]);
            if (abs($SimRes[$i]) < $lo) {
                $hi=$lo;
                $lo=abs($SimRes[$i]);
            } else {
                $hi=abs($SimRes[$i]);
            }
            $mag=int(log($lo)/log(10))+1;
            if ($lo < 1) {$mag-=1}
            $lo=int(0.5+$lo*10**($nDigits+1-$mag));
            $hi=int(0.5+$hi*10**($nDigits+1-$mag));
            next if (abs($lo-$hi)<=10);
            $absErr=abs($RefRes[$i]-$SimRes[$i]);
            $relErr=$absErr/(0.5*(abs($RefRes[$i])+abs($SimRes[$i])+$absErr));
            next if ($relErr<$relTol);
            if ($main::verbose) {print STDERR $ColNames[$i],$RefRes[$i],$SimRes[$i],100*$relErr."\%"}
            $matchType=2 if ($matchType<2);
            $maxRelErr=$relErr if ($relErr > $maxRelErr);
        }
    }
    if ($matchType==0) {
        return("MATCH       (exact)");
    } elsif ($matchType==1) {
        return("MATCH       (within specified tolerances)");
    } elsif ($matchType==2) {
        $mag=int(log($maxRelErr)/log(10));
        $i=$maxRelErr/10**$mag;
        $maxRelErr=100*10**$mag*int(0.5+1e3*$i)/1e3;
        return("DIFFER      (max rel error is $maxRelErr\%)");
    }
}
