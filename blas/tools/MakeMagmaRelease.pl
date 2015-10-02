#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my $major; 
my $minor; 
my $micro; 

# default options
my $DIRNAME;
my $BASENAME;
#my $svn    = "https://icl.cs.utk.edu/svn/magma/branches/sc_release";
my $svn    = "https://icl.cs.utk.edu/svn/magma/trunk";
#my $svninst= "https://icl.cs.utk.edu/svn/plasma/plasma-installer";
my $user   = "";
my $rc = 0;

my @file2delete = (
    "Makefile.gen",
    "tools",
    "quark",
    "docs",
    "include/Makefile",
    "make.inc.cumin",
    "make.inc.disco",
    "make.inc.ig",
    "make.inc.ig.pgi",
    "Release-ToDo.txt",
    "BugsToFix.txt",
    #"src/magma_zf77.cpp",
    #"src/magma_zf77pgi.cpp",
    "src/zgeqrf_mc.cpp",
    "src/zgeqrf-v2.cpp",
    "src/zgeqrf-v3.cpp",
    "src/zgetrf_mc.cpp",
    "src/zpotrf_mc.cpp",
    "testing/fortran2.cpp",
    "testing/*.txt",
    "testing/testing_zgetrf_f.f",
    "testing/testing_zgetrf_gpu_f.cuf",
    "testing/testing_zgeqrf_mc.cpp",
    "testing/testing_zgeqrf-v2.cpp",
    "testing/testing_zpotrf_mc.cpp",
    "testing/testing_zgetrf_mc.cpp",
    "testing/testing_zswap.cpp",
   );

my $RELEASE_PATH;
my %opts;
my $NUMREL = "";

sub myCmd {
    my ($cmd) = @_ ;
    my $err = 0;

    print "---------------------------------------------------------------\n";
    print $cmd."\n";
    print "---------------------------------------------------------------\n";
    $err = system($cmd);
    if ($err != 0) {
    	print "Error during execution of the following command:\n$cmd\n";
    	exit;
    }    
}

sub MakeRelease {

    my $numversion = $major.'.'.$minor.'.'.$micro;
    my $cmd;

    if ( $rc > 0 ) {
	$numversion = $numversion."-rc".$rc;
    }

    $RELEASE_PATH = $ENV{ PWD}."/magma_".$numversion;

    # Sauvegarde du rep courant
    my $dir = `pwd`; chop $dir;

    $cmd = 'svn export --force '.$NUMREL.' '.$user.' '.$svn.' '.$RELEASE_PATH;
    myCmd($cmd);
    
    chdir $RELEASE_PATH;

    # Change version in plasma.h
    #myCmd("sed -i 's/PLASMA_VERSION_MAJOR[ ]*[0-9]/PLASMA_VERSION_MAJOR $major/' include/plasma.h"); 
    #myCmd("sed -i 's/PLASMA_VERSION_MINOR[ ]*[0-9]/PLASMA_VERSION_MINOR $minor/' include/plasma.h");
    #myCmd("sed -i 's/PLASMA_VERSION_MICRO[ ]*[0-9]/PLASMA_VERSION_MICRO $micro/' include/plasma.h");

    # Change the version in comments 
    #myCmd("find -type f -exec sed -i 's/\@version[ ]*[.0-9]*/\@version $numversion/' {} \\;");

    #Precision Generation
    print "Generate the different precision\n"; 
    myCmd("touch make.inc");
    myCmd("make generation");

    #Compile the documentation
    #print "Compile the documentation\n"; 
    #system("make -C ./docs");
    myCmd("rm -f make.inc");
    
    #Remove non required files (Makefile.gen)
    foreach my $file (@file2delete){
	print "Remove $file\n";
 	myCmd("rm -rf $RELEASE_PATH/$file");
    }
 
    # Remove 'include Makefile.gen from Makefile'
    myCmd("find $RELEASE_PATH -name Makefile -exec sed -i '/Makefile.gen/ d' {} \\;");

    # Remove the lines relative to include directory in root Makefile
    myCmd("sed -i '/cd include/ d' $RELEASE_PATH/Makefile");

    # Remove '.Makefile.gen files'
    myCmd("find $RELEASE_PATH -name .Makefile.gen -exec rm -f {} \\;");

    chdir $dir;

    # Save the InstallationGuide if we want to do a plasma-installer release
    #myCmd("cp $RELEASE_PATH/InstallationGuide README-installer");

    #Create tarball
    print "Create the tarball\n";
    $DIRNAME=`dirname $RELEASE_PATH`;
    $BASENAME=`basename $RELEASE_PATH`;
    chop $DIRNAME;
    chop $BASENAME;
    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");

}

#sub MakeInstallerRelease {
#
#    my $numversion = $major.'.'.$minor.'.'.$micro;
#    my $cmd;
#
#    $RELEASE_PATH = $ENV{ PWD}."/plasma-installer_".$numversion;
#
#    # Sauvegarde du rep courant
#    my $dir = `pwd`; chop $dir;
#
#    $cmd = 'svn export --force '.$NUMREL.' '.$user.' '.$svninst.' '.$RELEASE_PATH;
#    myCmd($cmd);
#    
#    # Save the InstallationGuide if we want to do a plasma-installer release
#    myCmd("cp README-installer $RELEASE_PATH/README");
#
#    #Create tarball
#    print "Create the tarball\n";
#    $DIRNAME=`dirname $RELEASE_PATH`;
#    $BASENAME=`basename $RELEASE_PATH`;
#    chop $DIRNAME;
#    chop $BASENAME;
#    myCmd("(cd $DIRNAME && tar cvzf ${BASENAME}.tar.gz $BASENAME)");
#}

sub Usage {
    
    print "MakeRelease.pl [ -h ][ -d Directory ] [ -u username ] [ -r numrelease ] Major Minor Micro\n";
    print "   -h   Print this help\n";
    print "   -d   Choose directory for release\n";
    print "   -r   Choose svn release number\n";
    print "   -s   Choose magma directory for export\n";
    print "   -u   username\n";

}

getopts("hd:u:r:s:c:",\%opts);

if ( defined $opts{h} ){
    Usage();
    exit;
}

if (defined $opts{d}){
    $RELEASE_PATH = $opts{d};
}
if (defined $opts{u}){
    $user = "--username $opts{u}";
}

if (defined $opts{r}){
    $NUMREL = "-r $opts{r}";
}
if (defined $opts{s}){
    $svn = $opts{s};
}
if (defined $opts{c}){
    $rc = $opts{c};
}

if ( ($#ARGV + 1) < 3 ) {
    Usage();
    exit;
}

$major = $ARGV[0];
$minor = $ARGV[1];
$micro = $ARGV[2];

MakeRelease();
#MakeInstallerRelease();
