#use warnings
#use strict

my $va_code = '../vacode/hicumL2V2p33.va';
my %ref_para = ();

open(FILE, "<$va_code");

while (<FILE>) {
	my ($dummy, $name, $value) = /parameter\s*(real|integer)\s*(\w+)\s*=\s*([+-]?\d+(\.\d+)?([Ee][+-]?\d+)?)/;
	
	$ref_para{$name} = $value;
}

my @para_names = keys %ref_para;

close(FILE);

my @parameters = ('npn_1D','npn_cornoise','npn_full','npn_full_sh','npn_full_subcoupl','npn_full_subtran','npn_internal','npn_lat_nqs','npn_vert_nqs');

foreach my $set (@parameters) {
	open(FILE,"<$set");
	
	my %act_para = ();
	
	while (<FILE>) {
		my ($name, $value) = /\+\s?(\w+)\s?=\s?\(\s?([+-]?\d+(\.\d+)?([Ee][+-]?\d+)?)\s?\)/;
		
		$act_para{$name} = $value;
	}
	
	my @neq = grep { $ref_para{$_} != $act_para{$_} } @para_names;
	my @eq = grep { $ref_para{$_} == $act_para{$_} } @para_names;
	
	my $percentage = ($#neq/$#para_names)*100;
	
	print "$set : $percentage\n";
	#print join(' ', @eq), "\n";
	
	close(FILE);
}
