eval '(exit $?0)' && 
  eval 'exec perl -S $0 ${1+"$@"}' && 
  eval 'exec perl -S $0 $argv:q'
  if 0;
# $Id$ 
# Copyright 2000 by John Sheahan <john@reptechnic.com.au>
#
#  This program is free software; you may redistribute and/or modify it under
#  the terms of the GNU General Public License Version 2 as published by the
#  Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY, without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for complete details.
#
# measure utility for spice
# call to 'meas' inserted into spicedeck by spicepp
# preceeded by write of a data and control file
# copyright John Sheahan taudelta@integritynet.com.au
$ver='$id$ ';

# data is in meas.data in pwd
############# control information if=nfo is in meas.ctl
# output is appended to spice.log
$data="meas.data";
$ctl="meas.ctl";
$log="spice.log";

$date = `date`;   chop($date); 
$user=$ENV{"USER"};

open CTL, $ctl || die "cannot open control file $ctl";
$_=<CTL>;
if (/^\.measu?r?e? (tran|dc|ac) (\S+) trig\s+(\S+) (.*)targ\s+(\S+)(.*)/) {
  $name=$2;
  $trig=$3;
  $trigstr=$4;
  $targ=$5;
  $targstr=$6;
  if ($trigstr =~ /\bval=(\S+)/) { $trigval=&unit($1)}
  else { print "ERROR: no trigger value\n";  }
  if ($trigstr =~ /\b(fall|rise|count)=(\S+)/) {$trigdrn=$1 ; $trigcount=&unit($2); } 
  else { print "ERROR: no trigger rise/ fall / cross\n"; }
  if ($trigstr =~ /\btd=(\S+)/) {$trigdelay=&unit($1)}
  else {$trigdelay=0}
  if ($targstr =~ /\bval=(\S+)/) { $targval=&unit($1)}
  else { print "ERROR: no target value\n";  }
  if ($targstr =~ /\b(fall|rise|count)=(\S+)/) {$targdrn=$1 ; $targcount=&unit($2); } 
  else { print "ERROR: no target rise/ fall / cross\n"; }
  if ($targstr =~ /\btd=(\S+)/) {$targdelay=&unit($1)}
  else {$targdelay=0}
  /^\.measu?r?e? (tran|dc|ac) \S+ (.*)/;  $remainder=$2;
}
else {
  print "ERROR cannot parse command line\n"; 
}
close CTL;
#($name, $a_level,$a_drn,$b_level,$b_drn)=split; 

open DATA, $data || die "cannot open data file $data";
$title=<DATA>;
while (<DATA>) {
  if (/^(\d+)\s+([0-9\.e\+\-]+)\s+([0-9\.e\+\-]+)\s+([0-9\.e\+\-]+)/) {
    $time[$1]=$2;
    $a[$1]=$3;
    $b[$1]=$4;
  }
}
close DATA;

$first=&get_index($trigdelay,@time);
if ($trigdrn eq 'rise') { 
  $aindex = &get_rise($trigval,$trigcount,$first,@a);
}
elsif ($trigdrn eq 'fall') {
  $aindex = &get_fall($trigval,$trigcount,$first,@a);
}
else {
  $aindex = &get_cross($trigval,$trigcount,$first,@a);
}
$atime = &get_time($aindex,@time);
#print "atime=$atime aindex=$aindex first=$first tdelay=$trigdelay count=$trigcount drn=$trigdrn\n";

$first=&get_index($targdelay,@time);
if ($targdrn eq 'rise') { 
  $bindex = &get_rise($targval,$targcount,$first,@b);
}
elsif ($targdrn eq 'fall') {
  $bindex = &get_fall($targval,$targcount,$first,@b);
}
else {
  $bindex = &get_cross($targval,$targcount,$first,@b);
}
$btime = &get_time($bindex,@time);
#print "btime=$btime bindex=$bindex\n";
#print "btime=$btime bindex=$bindex first=$first tdelay=$targdelay count=$targcount drn=$targdrn\n";
$ans=$btime-$atime;

open LOG, ">>$log" || die "cannot open logfile $log";
print LOG "$title";
print LOG "$user $date\n";
print LOG "$name=$ans trig=$atime targ=$btime $remainder\n\n"; 
print "$user $date\n";
print "$name=$ans trig=$atime targ=$btime $remainder\n"; 
close LOG;


################### get_fall ################
sub get_fall {
  $level=shift(@_);
  $count=shift(@_);
  $first=shift(@_);
  for ($i=$first;$i<@_-1;$i++) {
    if (($_[$i] >= $level) && ($_[$i+1] < $level)) {
      $count--;
      return $i + (($_[$i]-$level)/($_[$i]-$_[$i+1])) if ($count==0);
   }
  }
  return undef;
}
sub get_rise {
  $level=shift(@_);
  $count=shift(@_);
  $first=shift(@_);
  for ($i=$first;$i<@_-1;$i++) {
    if (($_[$i] <= $level) && ($_[$i+1] > $level)) {
      $count--;
      return $i + (($level-$_[$i])/($_[$i+1]-$_[$i])) if ($count==0);
   }
  }
  return undef;
}
sub get_cross {
  $level=shift(@_);
  $count=shift(@_);
  $first=shift(@_);
  for ($i=$first;$i<@_-1;$i++) {
    if ((($_[$i] <= $level) && ($_[$i+1] > $level)) || 
        (($_[$i] >= $level) && ($_[$i+1] < $level))) {
      $count--;
      return $i + (($level-$_[$i])/($_[$i+1]-$_[$i])) if ($count==0);
   }
  }
  return undef;
}
# return the interpolated time corresponding to the (fractional) index
sub get_time {
  $level=shift(@_);
  $index=int($level);
  $time = $_[$index] + (($_[$index+1] - $_[$index]) * ($level-$index));
  return $time;
}

# return the first index at or after the given time 
sub get_index {
  $time=shift(@_);
  for ($i=0;$i<@_-1;$i++) {
    return $i  if ($_[$i] >= $time) 
  }
  return undef;
}


sub unit {
  if ($_[0] =~ /([0-9e\+\-\.]+)(t|g|meg|k|mil|m|u|n|p|f)?(v|a|s)?/) {
    if    ($2 eq 't')   { $mult = 1e12 } 
    elsif ($2 eq 'g')   { $mult = 1e9  } 
    elsif ($2 eq 'meg') { $mult = 1e6  } 
    elsif ($2 eq 'k')   { $mult = 1e3  } 
    elsif ($2 eq 'm')   { $mult = 1e-3   } 
    elsif ($2 eq 'u')   { $mult = 1e-6   } 
    elsif ($2 eq 'n')   { $mult = 1e-9   } 
    elsif ($2 eq 'p')   { $mult = 1e-12  } 
    elsif ($2 eq 'f')   { $mult = 1e-15  } 
    elsif ($2 eq 'mil') { $mult = 25.4e-6} 
    else                { $mult = 1  } 
    return $1 * $mult;
  }
  else { 
    print "ERROR: cannot result value $_[0]\n";
  }
  return $_;  # maybe perl does it better??
}
