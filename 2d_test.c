// Simulation of Schrodinger equation.
// Including inverse time evolution to find the ground state.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <omp.h>

// Constants etc.

#define Pi 3.141592653589793 // Pi
#define h 6.62606896E-34 // Plancks constant
#define hbar 1.054571628E-34 // Plancks reduced constant
#define kB 1.3806504E-23 // Boltzmann constant
#define m 3.8175405E-26 // Sodium mass
#define eps0 8.8541878176E-12 // vaccuum permittivity
#define me 9.10938215E-31 // Electron mass
#define qe 1.602176487E-19 // Electron charge
#define as 2.80358086E-9 // s-wave scatt length

// System size (MUST ALL BE EVEN)

#define Lx 2000
#define Ly 1200
#define STEPSPERSCALETIME 100000

// Temporary definition of MAX function
#define MAX(a,b) (((a)>(b))?(a):(b))

// System structure definition
struct dir {
	double w;
	double losc;		// Oscillator length
	double tosc;		// Oscillator period
	double s;			// System size
	double * x;		// System grid
	double dx;			// Lattice constant
	double * k;		// reciprocal system grid
	double dk;			// reciprocal lattice constant
	double rtf;		// Thomas Fermi radius
	double rtfs;		// Scaled Thomas Fermi radius
};

struct system {
	struct dir x;
	struct dir y;
	double lscale;		// Length scaling
	double tscale;			// Time scaling
	double dt;				// Size of time step
	double * k2;			//  k-squared array
	double * x2;			// x-squared array
	double consx;
	double consk;
	double consint;
	fftw_complex * psi;		// Wave function
	fftw_plan pf;			// Plan forward
	fftw_plan pb;			// Plan back
	int gridsize;			// Grid size
	double npart;			// Number of particles in the system
	double mu;			// Chemical potential of the system
	double mus;			// Scaled (Dimensionless) chemical potential of the system
	double gs;			// Scaled U0
	double * extrapot;		// Added potential
};

// Function definitions
struct system normalize(struct system sys);
double densintegrate(struct system sys);
double enerintegrate(struct system sys);
struct system initialize();
struct system timestep(struct system sys);
struct system imaginarytimestep(struct system sys);
double innerproductwithgroundstate(struct system sys);
void writetofile(struct system sys,int num);


//

// Main loop
void main(){
	struct system sys = initialize();

	int ii; // Initialize iterators

	printf("Sum: %E\n",densintegrate(sys));
	printf("WithGroundState: %d\t %E\n",-1,innerproductwithgroundstate(sys));

	for(ii=0;ii<50000;ii++){
		if(ii % 10 == 0)
			printf("WithGroundState: %d\t %E\n",ii,innerproductwithgroundstate(sys));
		sys = imaginarytimestep(sys);
	}
	printf("Done with imaginary time evolution!\n");

	for(ii=-1;ii<50000;ii++){
		if(ii % 100 == 0){
			printf("WithGroundState: %d\t %E\tSum: %E\n",ii,innerproductwithgroundstate(sys),densintegrate(sys));
			writetofile(sys,ii/100);
		}
		sys = timestep(sys);
	}

	//writetofile(sys,0);
	printf("Sum: %E\n",sys.dt);

	return;
}


// This will intialize the system construct for use
struct system initialize(){
	int ii,jj; // Local loop iterator

	struct system sys; // Define the system construct here.

	// Define the system size
	sys.gridsize = Lx*Ly;

	// Allocate memory
	sys.psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sys.gridsize);
	sys.k2 = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.x2 = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.extrapot = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.x.x = (double*) malloc(sizeof(double) * Lx);
	sys.x.k = (double*) malloc(sizeof(double) * Lx);
	sys.y.x = (double*) malloc(sizeof(double) * Ly);
	sys.y.k = (double*) malloc(sizeof(double) * Ly);

	if((sys.psi==NULL)||(sys.k2==NULL)||(sys.x2==NULL)||(sys.extrapot==NULL)||(sys.x.x==NULL)||(sys.x.k==NULL)||(sys.y.x==NULL)||(sys.y.k==NULL))
		printf("Memory wrongly allocated!\n");

	// Define chemical potential (calculate from mu later?)
	sys.npart = 2E7;
	sys.mu = 1.68115*pow(as,2./5.)*pow(m,1./5.)*pow(sys.npart,2./5.)*pow(2*Pi*104.,4./5.)*pow(2*Pi*16.,2./5.)*pow(hbar,4./5.); // mu ~ 2KHz

	// Initialize values for the x-direction
	sys.x.w = 2.*Pi* 16.;
	sys.x.losc = sqrt(hbar/(m*sys.x.w));
	sys.x.rtf = sqrt(2.*sys.mu/(m*sys.x.w*sys.x.w));
	sys.x.tosc = 2.*Pi/sys.x.w;

	// Initialize values for the y-direction
	sys.y.w = 2.*Pi* 104.;
	sys.y.losc = sqrt(hbar/(m*sys.y.w));
	sys.y.rtf = sqrt(2.*sys.mu/(m*sys.y.w*sys.y.w));
	sys.y.tosc = 2.*Pi/sys.y.w;

	// Initialize scaled parameters
	sys.tscale = sys.x.tosc;
	sys.dt = 1./(double)STEPSPERSCALETIME;
	sys.lscale = sys.x.losc;	// Does NOT work yet! Keep at x.losc!
	sys.mus = sys.mu*sys.tscale/hbar; // Calculate the scaled chemical potential
	sys.gs = 4*Pi*hbar*as*sys.tscale/(m * pow(sys.lscale,2.));

	sys.x.rtfs = sys.x.rtf/sys.lscale;
	sys.x.s = 2.* sys.x.rtfs;		// System size is from -param*lscale to param*lscale
	sys.x.dx = 2.*sys.x.s /(double)Lx;
	sys.x.dk = Pi/sys.x.s;

	sys.y.rtfs = sys.y.rtf/sys.lscale;
	sys.y.s = 2.* sys.y.rtfs;		// System size is from -param*lscale to param*lscale
	sys.y.dx = 2.*sys.y.s /(double)Ly;
	sys.y.dk = Pi/sys.y.s;

	// Initialize the grids
	for(ii=0;ii<Lx;ii++){
		sys.x.x[ii] = -sys.x.s + sys.x.dx*(double)ii;
		if(ii<Lx/2)
			sys.x.k[ii] = sys.x.dk*(double)ii;
		else
			sys.x.k[ii] = sys.x.dk*(double)(Lx-ii);
	}

	for(ii=0;ii<Ly;ii++){
		sys.y.x[ii] = -sys.y.s + sys.y.dx*(double)ii;
		if(ii<Ly/2)
			sys.y.k[ii] = sys.y.dk*(double)ii;
		else
			sys.y.k[ii] = sys.y.dk*(double)(Ly-ii);
	}

	// Initialize the k-sq grid
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			sys.k2[ii*Ly+jj] = sys.x.k[ii]*sys.x.k[ii]+sys.y.k[jj]*sys.y.k[jj];
			sys.x2[ii*Ly+jj] = sys.x.x[ii]*sys.x.x[ii]+(sys.y.w/sys.x.w)*(sys.y.w/sys.x.w)*sys.y.x[jj]*sys.y.x[jj];
		}
	}

	// Use multithreaded fftw (Not enabled on students server)
	omp_set_num_threads(3);
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());

	// Initialize fftw plans
	sys.pf = fftw_plan_dft_2d(Lx, Ly, sys.psi, sys.psi, FFTW_FORWARD, FFTW_MEASURE);
	sys.pb = fftw_plan_dft_2d(Lx, Ly, sys.psi, sys.psi, FFTW_BACKWARD, FFTW_MEASURE);

	// Initialize the wavefunction in the Thomas Fermi profile
	//double tempconst1 = sqrt(sys.lscale)*sqrt(2./9.* sys.mu* sys.mu/(hbar*hbar*2.*2.*Pi*Pi*104.*104.*as*sys.npart));
	double tempconst1 = sys.lscale*sqrt(pow(sys.mu,3./2.)/(3.*sqrt(3.*Pi)*pow(hbar*2.*Pi*104.,3./2.)*as*sqrt(hbar/(m*2.*Pi*104.))));//*sys.npart));
	//double tempconst2x = sqrt(2.*sys.mu/(m*2.*2.*Pi*Pi*16.*16.));
	//double tempconst2y = sqrt(2.*sys.mu/(m*2.*2.*Pi*Pi*104.*104.));
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			sys.psi[ii*Ly+jj][0] = tempconst1*pow(MAX(0.,1-(sys.x.x[ii]/sys.x.rtfs)*(sys.x.x[ii]/sys.x.rtfs)-(sys.y.x[jj]/sys.y.rtfs)*(sys.y.x[jj]/sys.y.rtfs)),3./4.);
			sys.psi[ii*Ly+jj][1] = 0;
		}
	}

	// Calculate the particle number
	// printf("npart: %E\n",densintegrate(sys));

	// Initialize the simulation constants
	sys.consx = -0.25*(m * sys.x.w * sys.lscale * sys.lscale)/hbar * sys.x.w * sys.tscale * sys.dt; // Divided by 2 because we do 2 seperate space steps
	sys.consk = -0.5*hbar/(m * sys.x.w * sys.lscale * sys.lscale) * sys.x.w * sys.tscale * sys.dt;
	sys.consint = -0.5*0.75*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*sys.gs/(pow(Pi,2./3.)*pow(hbar/(2. * m *Pi*104.),2./3.)*pow(as,1./3.))*sys.dt; // Divided by 2 because we do 2 seperate space steps

	// Initialize the extra potential.
	double gausstemp;
	gausstemp = sys.tscale*2*Pi*500./(sqrt(2*Pi)*10.*sys.x.dx)*sys.dt;
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			sys.extrapot[ii*Ly+jj] = -0.5*gausstemp*exp(-(sys.x.x[ii]/(10.*sys.x.dx))*(sys.x.x[ii]/(10.*sys.x.dx))/2.);
		}
	}
	printf("List of constants:\nmu: %E\nnpart: %E\nx: %E\nk: %E\nint: %E\nGauss: %E\n",sys.mu,sys.npart,sys.consx,sys.consk,sys.consint,gausstemp);

	return normalize(sys);
}

void writetofile(struct system sys,int num){
	int ii,jj; // Initialize iterators.

	char filename[50];
	sprintf(filename, "filedump/tempfile_gs_%d.dat", num);
	FILE * file = fopen(filename,"w");

	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			fprintf(file,"%E\t%E\t%E\t%E\n",sys.x.x[ii],sys.y.x[jj],sys.psi[ii*Ly+jj][0],sys.psi[ii*Ly+jj][1]);
		}
	}
	fclose(file);
	return;
}

// Performs a single time step on the system
struct system timestep(struct system sys){
	int ii,jj; // Initialize iterators
	double xi,psire,psiim,psi2,arg,cosarg,sinarg;

	// Half a space step
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			psire = sys.psi[ii*Ly+jj][0];
			psiim = sys.psi[ii*Ly+jj][1];

			psi2 = pow(psire*psire+psiim*psiim,2./3.);

			arg = sys.consx * sys.x2[ii*Ly+jj]+sys.consint*psi2;
			sinarg = sin(arg);
			cosarg = cos(arg);

			sys.psi[ii*Ly+jj][0] = psire * cosarg - psiim * sinarg;
			sys.psi[ii*Ly+jj][1] = psire * sinarg + psiim * cosarg;
		}
	}

	// Fourier transform of wave function
	fftw_execute(sys.pf);

	// Momentum space step
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			psire = sys.psi[ii*Ly+jj][0];
			psiim = sys.psi[ii*Ly+jj][1];

			arg = sys.consk * sys.k2[ii*Ly+jj];
			sinarg = sin(arg);
			cosarg = cos(arg);

			sys.psi[ii*Ly+jj][0] = (psire * cosarg - psiim * sinarg)/(double)sys.gridsize;
			sys.psi[ii*Ly+jj][1] = (psire * sinarg + psiim * cosarg)/(double)sys.gridsize;
		}
	}

	// Fourier transform back
	fftw_execute(sys.pb);

	// Half a space step
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			psire = sys.psi[ii*Ly+jj][0];
			psiim = sys.psi[ii*Ly+jj][1];

			psi2 = pow(psire*psire+psiim*psiim,2./3.);

			arg = sys.consx * sys.x2[ii*Ly+jj]+sys.consint*psi2;
			sinarg = sin(arg);
			cosarg = cos(arg);

			sys.psi[ii*Ly+jj][0] = psire * cosarg - psiim * sinarg;
			sys.psi[ii*Ly+jj][1] = psire * sinarg + psiim * cosarg;
		}
	}



	// Return updated system
	return sys;
}

// Performs a single time step on the system
struct system imaginarytimestep(struct system sys){

	#pragma omp parallel
	{

		int ii,jj; // Initialize iterators
		double psire,psiim,psi2,arg,exparg;

		// Half a space step
		#pragma omp for
		for(ii=0;ii<Lx;ii++){
			for(jj=0;jj<Ly;jj++){
				psire = sys.psi[ii*Ly+jj][0];
				psiim = sys.psi[ii*Ly+jj][1];

				psi2 = pow(psire*psire+psiim*psiim,2./3.);

				arg = sys.consx * sys.x2[ii*Ly+jj]+sys.consint*psi2;
				exparg = exp(arg);

				sys.psi[ii*Ly+jj][0] = psire * exparg;
				sys.psi[ii*Ly+jj][1] = psiim * exparg;
			}
		}
	}

	// Fourier transform of wave function
	fftw_execute(sys.pf);

	#pragma omp parallel
	{

		int ii,jj; // Initialize iterators
		double psire,psiim,psi2,arg,exparg;

		// Momentum space step
		#pragma omp for
		for(ii=0;ii<Lx;ii++){
			for(jj=0;jj<Ly;jj++){
				psire = sys.psi[ii*Ly+jj][0];
				psiim = sys.psi[ii*Ly+jj][1];
				arg = sys.consk * sys.k2[ii*Ly+jj];
				exparg = exp(arg);
				sys.psi[ii*Ly+jj][0] = psire * exparg/(double)sys.gridsize;
				sys.psi[ii*Ly+jj][1] = psiim * exparg/(double)sys.gridsize;
			}
		}
	}
	
	// Fourier transform back
	fftw_execute(sys.pb);


	#pragma omp parallel
	{
		int ii,jj; // Initialize iterators
		double psire,psiim,psi2,arg,exparg;

		#pragma omp for
		for(ii=0;ii<Lx;ii++){
			for(jj=0;jj<Ly;jj++){
				psire = sys.psi[ii*Ly+jj][0];
				psiim = sys.psi[ii*Ly+jj][1];

				psi2 = pow(psire*psire+psiim*psiim,2./3.);
				arg = sys.consx * sys.x2[ii*Ly+jj]+sys.consint*psi2;
				exparg = exp(arg);
				sys.psi[ii*Ly+jj][0] = psire * exparg;
				sys.psi[ii*Ly+jj][1] = psiim * exparg;
			}
		}
	}


	// Return updated system
	return normalize(sys);
}

// Normalize the wavefunction
struct system normalize(struct system sys){
	int ii,jj; // Initialize iterators

	double invtotal = 1/sqrt(densintegrate(sys));

	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			sys.psi[ii*Ly+jj][0] *= invtotal;
			sys.psi[ii*Ly+jj][1] *= invtotal;
		}
	}

	return sys;
}

// Calculate the density integral
double densintegrate(struct system sys){
	int ii,jj; // Initialize iterators

	double total = 0;
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			total += sys.psi[ii*Ly+jj][0]*sys.psi[ii*Ly+jj][0]+sys.psi[ii*Ly+jj][1]*sys.psi[ii*Ly+jj][1];
		}
	}

	return total*sys.x.dx*sys.y.dx;
}

double innerproductwithgroundstate(struct system sys){
	int ii,jj; // Initialize iterators

	double temp;
	double totalre = 0;
	double totalim = 0;

	double groundstate;

	double tempconst1 = sys.lscale*sqrt(pow(sys.mu,3./2.)/(3.*sqrt(3.*Pi)*pow(hbar*2.*Pi*104.,3./2.)*as*sqrt(hbar/(m*2.*Pi*104.))*sys.npart));
	double tempconst2x = sqrt(2.*sys.mu/(m*2.*2.*Pi*Pi*16.*16.));
	double tempconst2y = sqrt(2.*sys.mu/(m*2.*2.*Pi*Pi*104.*104.));
	for(ii=0;ii<Lx;ii++){
		for(jj=0;jj<Ly;jj++){
			groundstate = tempconst1*pow(MAX(0.,1-(sys.x.x[ii]*sys.lscale/tempconst2x)*(sys.x.x[ii]*sys.lscale/tempconst2x)-(sys.y.x[jj]*sys.lscale/tempconst2y)*(sys.y.x[jj]*sys.lscale/tempconst2y)),3./4.);
			totalre += groundstate * sys.psi[ii*Ly+jj][0];
			totalim += groundstate * sys.psi[ii*Ly+jj][1];
		}
	}

	return (totalre*totalre+totalim*totalim)*sys.x.dx*sys.x.dx*sys.y.dx*sys.y.dx;
}
