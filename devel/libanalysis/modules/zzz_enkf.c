/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'zzz_enkf.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/


#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/util/rng.h>
#include <ert/util/matrix.h>
#include <ert/util/matrix_blas.h>
#include <ert/util/bool_vector.h>

#include <ert/analysis/analysis_module.h>
#include <ert/analysis/analysis_table.h>
#include <ert/analysis/enkf_linalg.h>
#include <ert/analysis/std_enkf.h>

#include <zzz_enkf_common.h>

typedef struct zzz_enkf_data_struct zzz_enkf_data_type;


/*
  Observe that only one of the settings subspace_dimension and
  truncation can be valid at a time; otherwise the svd routine will
  fail. This implies that the set_truncation() and
  set_subspace_dimension() routines will set one variable, AND
  INVALIDATE THE OTHER. For most situations this will be OK, but if
  you have repeated calls to both of these functions the end result
  might be a surprise.  
*/
#define INVALID_SUBSPACE_DIMENSION     -1
#define INVALID_TRUNCATION             -1
#define DEFAULT_SUBSPACE_DIMENSION     INVALID_SUBSPACE_DIMENSION
#define DEFAULT_USE_PRIOR              true
#define DEFAULT_LAMBDA_INCREASE_FACTOR 4
#define DEFAULT_LAMBDA_REDUCE_FACTOR   0.1
#define DEFAULT_LAMBDA0                -1
#define DEFAULT_LAMBDA_MIN             0.01
#define DEFAULT_LAMBDA_RECALCULATE     false
#define DEFAULT_LOG_FILE               "zzz_enkf.out"
#define DEFAULT_CLEAR_LOG              true

 

#define  USE_PRIOR_KEY               "USE_PRIOR"
#define  LAMBDA_REDUCE_FACTOR_KEY    "LAMBDA_REDUCE"
#define  LAMBDA_INCREASE_FACTOR_KEY  "LAMBDA_INCREASE"
#define  LAMBDA0_KEY                 "LAMBDA0"
#define  LAMBDA_MIN_KEY              "LAMBDA_MIN"
#define  LAMBDA_RECALCULATE_KEY      "LAMBDA_RECALCULATE"
#define  ITER_KEY                    "ITER"
#define  LOG_FILE_KEY                "LOG_FILE"
#define  CLEAR_LOG_KEY               "CLEAR_LOG" 


/*
  The configuration data used by the zzz_enkf module is contained in a
  zzz_enkf_data_struct instance. The data type used for the zzz_enkf
  module is quite simple; with only a few scalar variables, but there
  are essentially no limits to what you can pack into such a datatype.

  All the functions in the module have a void pointer as the first
  argument, this will immediately be casted to a zzz_enkf_data_type
  instance, to get some type safety the UTIL_TYPE_ID system should be
  used (see documentation in util.h)

  The data structure holding the data for your analysis module should
  be created and initialized by a constructor, which should be
  registered with the '.alloc' element of the analysis table; in the
  same manner the desctruction of this data should be handled by a
  destructor or free() function registered with the .freef field of
  the analysis table.
*/


#define RML_ENKF_TYPE_ID 261123

struct zzz_enkf_data_struct {
  UTIL_TYPE_ID_DECLARATION;
  double    truncation;            // Controlled by config key: ENKF_TRUNCATION_KEY
  int       subspace_dimension;    // Controlled by config key: ENKF_NCOMP_KEY (-1: use Truncation instead)
  long      option_flags;
  int       iteration_nr;          // Keep track of the outer iteration loop
  double    Sk;                    // Objective function value
  double    Std;                   // Standard Deviation of the Objective function
  double  * Csc;                   // Vector with scalings for non-dimensionalizing states
  matrix_type *Am;                 // 
  matrix_type *active_prior;       // m_l
  matrix_type *prior0;             // m_0
  matrix_type *state;              // 
  bool_vector_type * ens_mask;     // 
  bool use_prior;                  // Use exact/approximate scheme? Approximate scheme drops the "prior" term in the LM step.

  double    lambda;                 // parameter to control the setp length in Marquardt levenberg optimization 
  double    lambda0;
  double    lambda_min;
  double    lambda_reduce_factor;
  double    lambda_increase_factor;
  bool      lambda_recalculate;
  
  bool      clear_log;
  char    * log_file;
  FILE    * log_stream;
};



static UTIL_SAFE_CAST_FUNCTION( zzz_enkf_data , RML_ENKF_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION_CONST( zzz_enkf_data , RML_ENKF_TYPE_ID )





/** GET/SET Settings **/
double zzz_enkf_get_truncation( zzz_enkf_data_type * data ) {
  return data->truncation;
}

int zzz_enkf_get_subspace_dimension( zzz_enkf_data_type * data ) {
  return data->subspace_dimension;
}

void zzz_enkf_set_truncation( zzz_enkf_data_type * data , double truncation ) {
  data->truncation = truncation;
  if (truncation > 0.0)
    data->subspace_dimension = INVALID_SUBSPACE_DIMENSION;
}

void zzz_enkf_set_lambda0( zzz_enkf_data_type * data , double lambda0) {
  data->lambda0 = lambda0;
}

double zzz_enkf_get_lambda0( const zzz_enkf_data_type * data ) {
  return data->lambda0;
}

void zzz_enkf_set_lambda_min( zzz_enkf_data_type * data , double lambda_min) {
  data->lambda_min = lambda_min;
}

double zzz_enkf_get_lambda_min( const zzz_enkf_data_type * data ) {
  return data->lambda_min;
}

void zzz_enkf_set_lambda_increase_factor( zzz_enkf_data_type * data , double increase_factor) {
  data->lambda_increase_factor = increase_factor;
}

void zzz_enkf_set_lambda_reduce_factor( zzz_enkf_data_type * data , double reduce_factor) {
  data->lambda_reduce_factor = reduce_factor;
}

double zzz_enkf_get_lambda_increase_factor( const zzz_enkf_data_type * data ) {
  return data->lambda_increase_factor;
}

double zzz_enkf_get_lambda_reduce_factor( const zzz_enkf_data_type * data ) {
  return data->lambda_reduce_factor;
}

bool zzz_enkf_get_clear_log( const zzz_enkf_data_type * data ) {
  return data->clear_log;
}

void zzz_enkf_set_clear_log( zzz_enkf_data_type * data , bool clear_log) {
  data->clear_log = clear_log;
}

bool zzz_enkf_get_use_prior( const zzz_enkf_data_type * data ) {
  return data->use_prior;
}

void zzz_enkf_set_use_prior( zzz_enkf_data_type * data , bool use_prior) {
  data->use_prior = use_prior;
}

void zzz_enkf_set_lambda_recalculate( zzz_enkf_data_type * data , bool lambda_recalculate) {
  data->lambda_recalculate = lambda_recalculate;
}

void zzz_enkf_set_subspace_dimension( zzz_enkf_data_type * data , int subspace_dimension) {
  data->subspace_dimension = subspace_dimension;
  if (subspace_dimension > 0)
    data->truncation = INVALID_TRUNCATION;
}

void zzz_enkf_set_iteration_nr( zzz_enkf_data_type * data , int iteration_nr) {
  data->iteration_nr = iteration_nr;
}

int zzz_enkf_get_iteration_nr( const zzz_enkf_data_type * data ) {
  return data->iteration_nr;
}



/** Log related stuff **/
void zzz_enkf_set_log_file( zzz_enkf_data_type * data , const char * log_file ) {
  data->log_file = util_realloc_string_copy( data->log_file , log_file );
}

const char * zzz_enkf_get_log_file( const zzz_enkf_data_type * data) {
  return data->log_file;
}

void zzz_enkf_log_line( zzz_enkf_data_type * data , const char * fmt , ...) {
  if (data->log_stream) {
    va_list ap;
    va_start(ap , fmt);
    vfprintf( data->log_stream , fmt , ap );
    va_end( ap );
  }
}

static void zzz_enkf_write_log_header( zzz_enkf_data_type * data ) {
  if (data->log_stream) {
    const char * column1 = "\"Iteration Number\"";
    const char * column2 = "\"Lambda Value\"";
    const char * column3 = "\"Current Object Function Value\"";
    const char * column4 = "\"Previous Object Function Value\"";
    const char * column5 = "\"Current Standard Deviation\"";

    if (data->log_stream) {
      zzz_enkf_log_line(data, "%-23s %-19s %-36s %-37s %-33s\n", column1, column2, column3, column4, column5);
    }
  }
}

static void zzz_enkf_open_log_file( zzz_enkf_data_type * data ) {
  data->log_stream = NULL;
  if (data->log_file) {
    if ( data->iteration_nr == 0) {
      if (data->clear_log){
        data->log_stream = util_mkdir_fopen( data->log_file , "w");
        zzz_enkf_write_log_header(data);
      }
      else
        data->log_stream = util_mkdir_fopen( data->log_file , "a");
    } else
      data->log_stream = util_fopen( data->log_file , "a");
  }
}




/** Alloc / Free **/
void * zzz_enkf_data_alloc( rng_type * rng) {
  zzz_enkf_data_type * data = util_malloc( sizeof * data);
  UTIL_TYPE_ID_INIT( data , RML_ENKF_TYPE_ID );
    
  data->log_file     = NULL;

  zzz_enkf_set_truncation( data , DEFAULT_ENKF_TRUNCATION_ );
  zzz_enkf_set_subspace_dimension( data , DEFAULT_SUBSPACE_DIMENSION );
  zzz_enkf_set_use_prior( data , DEFAULT_USE_PRIOR );
  zzz_enkf_set_lambda0( data , DEFAULT_LAMBDA0 );
  zzz_enkf_set_lambda_increase_factor(data , DEFAULT_LAMBDA_INCREASE_FACTOR);
  zzz_enkf_set_lambda_reduce_factor(data , DEFAULT_LAMBDA_REDUCE_FACTOR);
  zzz_enkf_set_lambda_min( data , DEFAULT_LAMBDA_MIN );
  zzz_enkf_set_log_file( data , DEFAULT_LOG_FILE );
  zzz_enkf_set_clear_log( data , DEFAULT_CLEAR_LOG );
  zzz_enkf_set_lambda_recalculate( data , DEFAULT_LAMBDA_RECALCULATE );

  data->option_flags = ANALYSIS_NEED_ED + ANALYSIS_UPDATE_A + ANALYSIS_ITERABLE + ANALYSIS_SCALE_DATA;
  data->iteration_nr = 0;
  data->Std          = 0; 
  data->ens_mask     = bool_vector_alloc(0,false);
  data->state        = matrix_alloc(1,1);
  data->active_prior = matrix_alloc(1,1);
  data->prior0       = matrix_alloc(1,1);
  return data;
}

void zzz_enkf_data_free( void * arg ) { 
  zzz_enkf_data_type * data = zzz_enkf_data_safe_cast( arg );

  matrix_free( data->state );
  matrix_free( data->prior0 );
  matrix_free( data->active_prior );

  util_safe_free( data->log_file );
  bool_vector_free( data->ens_mask );
  free( data );
}





// Notation
/*
 * NOTATION
 *
 * A, Acopy, data->state, data->prior0, data->active_prior
 * These are all ensemble matrices holding the state at some stage of the iteration.
 * Possible multiple use of notations => Case-by-case dependent.
 *
 * Variable name in code <-> D.Oliver notation     <-> Description
 * -----------------------------------------------------------------------------------------------------------
 * Am                    <-> A_m                   <-> Am = Um*Wm^(-1)
 * Csc                   <-> C_sc^(1/2)            <-> State scalings. Note the square root.
 * Dm (in init1__)       <-> Delta m               <-> Anomalies of active_prior wrt. its mean (row i scaled by 1/(Csc[i]*sqrt(N-1)))
 * Dm (in initA__)       <-> Csc * Delta m         <-> Anomalies of A wrt. its mean (only scaled by 1/sqrt(N-1))
 * Dk1 (in init2__)      <-> Delta m               <-> Anomailes of Acopy (row i scaled by 1/(Csc[i]*sqrt(N-1)))
 * Dk (in init2__)       <-> Csc^(-1) * (m - m_pr) <-> Anomalies wrt. prior (as opposed to the mean; only scaled by Csc)
 * dA1 (in initA__)      <-> delta m               <-> Ensemble updates coming from data mismatch
 * dA2 (in init2__)      <-> delta m               <-> Ensemble updates coming from prior mismatch
 *
 * X1-X7, intermediate calculations in iterations. See D.Oliver algorithm
*/

// Just (pre)calculates data->Am = Um*Wm^(-1).
static void zzz_enkf_init1__( zzz_enkf_data_type * data) {
	// Differentiate this routine from init2__, which actually calculates the prior mismatch update.
	// This routine does not change any ensemble matrix.
	// Um*Wm^(-1) are the scaled, truncated, right singular vectors of data->active_prior.


  int state_size    = matrix_get_rows( data->active_prior );
  int ens_size      = matrix_get_columns( data->active_prior );
  int nrmin         = util_int_min( ens_size , state_size); 
  matrix_type * Dm  = matrix_alloc_copy( data->active_prior );
  matrix_type * Um  = matrix_alloc( state_size , nrmin  );     /* Left singular vectors.  */
  matrix_type * VmT = matrix_alloc( nrmin , ens_size );        /* Right singular vectors. */
  double * Wm       = util_calloc( nrmin , sizeof * Wm ); 
  double nsc        = 1/sqrt(ens_size - 1); 

  matrix_subtract_row_mean(Dm);

  for (int i=0; i < state_size; i++){
    double sc = nsc / (data->Csc[i]);
    matrix_scale_row( Dm , i , sc);
  }

	// Um Wm VmT = Dm; nsign1 = num of non-zero singular values.
  int nsign1 = enkf_linalg_svd_truncation(Dm , data->truncation , -1 , DGESVD_MIN_RETURN  , Wm , Um , VmT);
  
	// Am = Um*Wm^(-1). I.e. scale *columns* of Um
  enkf_linalg_rml_enkfAm(Um, Wm, nsign1);

  data->Am = matrix_alloc_copy( Um );
  matrix_free(Um);
  matrix_free(VmT);
  matrix_free(Dm);
  free(Wm);
}

// Creates state scaling matrix
void zzz_enkf_init_Csc(zzz_enkf_data_type * data){
  int state_size = matrix_get_rows( data->active_prior );
  int ens_size   = matrix_get_columns( data->active_prior );

  for (int row=0; row < state_size; row++) {
    double sumrow = matrix_get_row_sum(data->active_prior , row);
    double tmp    = sumrow / ens_size;

    if (abs(tmp)< 1)
      data->Csc[row] = 0.05;
    else
      data->Csc[row] = 1.00;

  }
}

// Calculates update from data mismatch (delta m_1). Also provides SVD for later use.
static void zzz_enkf_initA__(zzz_enkf_data_type * data,  matrix_type * A , matrix_type * S ,  matrix_type * Cd ,  matrix_type * E ,  matrix_type * D , matrix_type * Udr, double * Wdr, matrix_type * VdTr) {
// A : ensemble matrix
// S : measured ensemble
// Cd : inv(SampleCov of E)
// E : perturbations for obs
// D = dObs + E - S : Innovations (wrt pert. obs)

  int ens_size      = matrix_get_columns( S );
  double nsc        = 1/sqrt(ens_size-1);
  int nsign;

	// SVD:
	// tmp = diag_sqrt(Cd^(-1)) * centered(S) / sqrt(N-1) = Ud * Wd * Vd(T)
  {
    int nrobs         = matrix_get_rows( S );
    matrix_type *tmp  = matrix_alloc (nrobs, ens_size);
    matrix_subtract_row_mean( S );                        // Center S
    matrix_inplace_diag_sqrt(Cd);                         // Assumes that Cd is diag!
    matrix_matmul(tmp , Cd , S );                         //
    matrix_scale(tmp , nsc);                              // 
  
    nsign = enkf_linalg_svd_truncation(tmp , data->truncation , -1 , DGESVD_MIN_RETURN  , Wdr , Udr , VdTr);
    matrix_free( tmp );
  }
  
  {
    matrix_type * X3  = matrix_alloc( ens_size, ens_size );
		// X3
    {
      matrix_type * X1  = matrix_alloc( nsign, ens_size);
      matrix_type * X2  = matrix_alloc( nsign, ens_size );
      
			// See LM-EnRML algorithm in Oliver'2013 (Comp. Geo.)
      enkf_linalg_rml_enkfX1(X1, Udr ,D ,Cd );                         // X1 = Ud(T)*Cd(-1/2)*D   -- D= -(dk-d0)
      enkf_linalg_rml_enkfX2(X2, Wdr ,X1 ,data->lambda + 1 , nsign);   // X2 = ((a*Ipd)+Wd^2)^-1  * X1
      enkf_linalg_rml_enkfX3(X3, VdTr ,Wdr,X2, nsign);                 // X3 = Vd *Wd*X2
      
      matrix_free(X2);
      matrix_free(X1);
    }
    
		// Update A. Why is there no scaling here?
    {
      matrix_type * dA1 = matrix_alloc( matrix_get_rows(A) , ens_size);
      matrix_type * Dm = matrix_alloc_copy( A );

      matrix_subtract_row_mean( Dm );           /* Remove the mean from the ensemble of model parameters*/
      matrix_scale(Dm, nsc);                    // /sqrt(N-1)

      matrix_matmul(dA1, Dm , X3);              // 
      matrix_inplace_add(A,dA1);                // Add the update into A 

      matrix_free(Dm);
      matrix_free(dA1);
    }
    matrix_free(X3);

  }
}

// Calculate prior mismatch update (delta m_2).
void zzz_enkf_init2__( zzz_enkf_data_type * data, matrix_type *A, matrix_type *Acopy, double * Wdr, matrix_type * VdTr) {
	// Distinguish from init1__ which only makes preparations, and is only called at iter=0
	// Acopy : Copy of A. Possibly of an A that was rejected? See updateA()

  int state_size   = matrix_get_rows( Acopy );
  int ens_size     = matrix_get_columns( Acopy );
  double nsc       = 1/sqrt(ens_size-1); 

  matrix_type *Am  = matrix_alloc_copy(data->Am);
  matrix_type *Apr = matrix_alloc_copy(data->active_prior);


  int nsign1 = matrix_get_columns(data->Am);
  

  matrix_type * X4  = matrix_alloc(nsign1,ens_size);
  matrix_type * X5  = matrix_alloc(state_size,ens_size);
  matrix_type * X6  = matrix_alloc(ens_size,ens_size);
  matrix_type * X7  = matrix_alloc(ens_size,ens_size);
  matrix_type * dA2 = matrix_alloc(state_size , ens_size);
  matrix_type * Dk1 = matrix_alloc_copy( Acopy );
  
	// Dk = Csc^(-1) * (Acopy - Aprior)
	// X4 = Am' * Dk
  {
    matrix_type * Dk = matrix_alloc_copy( Acopy );
    matrix_inplace_sub(Dk, Apr);
    zzz_enkf_common_scaleA(Dk , data->Csc , true);
    matrix_dgemm(X4 , Am , Dk , true, false, 1.0, 0.0); // X4 = Am' * Dk
    matrix_free(Dk);
  }
	// X5 = Am * X4
  matrix_matmul(X5 , Am , X4);

  // Dk1 = Csc^(-1)/sqrt(N-1) * Acopy*(I - 1/N*ones(m,N))
  matrix_subtract_row_mean(Dk1);                  // Dk1 = Dk1 * (I - 1/N*ones(m,N))
  zzz_enkf_common_scaleA(Dk1 , data->Csc , true); // Dk1 = Csc^(-1) * Dk1
  matrix_scale(Dk1,nsc);                          // Dk1 = Dk1 / sqrt(N-1)
	
	// X6 = Dk1' * X5
  matrix_dgemm(X6, Dk1, X5, true, false, 1.0, 0.0); 
	// X7
  enkf_linalg_rml_enkfX7(X7, VdTr , Wdr , data->lambda + 1, X6);
  
	// 
  zzz_enkf_common_scaleA(Dk1 , data->Csc , false);
  matrix_matmul(dA2 , Dk1 , X7);
  matrix_inplace_sub(A, dA2);

  matrix_free(Am);
  matrix_free(Apr);
  matrix_free(X4); 
  matrix_free(X5);
  matrix_free(X6);
  matrix_free(X7);
  matrix_free(dA2);
  matrix_free(Dk1);
}

// Initialize state, prior0 and active_prior from A. Initialize lambda0, lambda. Call initA__, init1__
static void zzz_enkf_updateA_iter0(zzz_enkf_data_type * data, matrix_type * A ,  matrix_type * S ,  matrix_type * R ,  matrix_type * dObs ,  matrix_type * E ,  matrix_type * D, matrix_type * Cd) {
        
  int ens_size      = matrix_get_columns( S );
  int nrobs         = matrix_get_rows( S );
  int nrmin         = util_int_min( ens_size , nrobs); 
  int state_size    = matrix_get_rows( A );
	matrix_type * Skm = matrix_alloc(ens_size,ens_size);      // Mismatch
  matrix_type * Ud  = matrix_alloc( nrobs , nrmin    );    /* Left singular vectors.  */
  matrix_type * VdT = matrix_alloc( nrmin , ens_size );    /* Right singular vectors. */
  double * Wd       = util_calloc( nrmin , sizeof * Wd ); 

  data->Csc = util_calloc(state_size , sizeof * data->Csc);
  data->Sk  = enkf_linalg_data_mismatch(D,Cd,Skm);  
  data->Std = matrix_diag_std(Skm,data->Sk);
  
  if (data->lambda0 < 0)
    data->lambda = pow(10 , floor(log10(data->Sk/(2*nrobs))) );
  else
    data->lambda = data->lambda0;
  
	// state = A, prior0 = A, active_prior = prior0
  zzz_enkf_common_store_state( data->state  , A , data->ens_mask );
  zzz_enkf_common_store_state( data->prior0 , A , data->ens_mask );
  zzz_enkf_common_recover_state( data->prior0 , data->active_prior , data->ens_mask );

	// Update from data mismatch
  zzz_enkf_initA__(data , A, S , Cd , E , D , Ud , Wd , VdT);
	// Update from prior mismatch. This should be zero. So init1__ just prepares some matrices.
  if (data->use_prior) {
    zzz_enkf_init_Csc( data );
    zzz_enkf_init1__(data );
  }

  {
    const char * prev_obj_func_value_dummy = "-";
    zzz_enkf_log_line( data , "%-d-%-21d %-19.5f %-36.5f %-37s %-33.5f \n", data->iteration_nr, data->iteration_nr+1, data->lambda, data->Sk , prev_obj_func_value_dummy, data->Std);
  }
  
  matrix_free( Skm );
  matrix_free( Ud );
  matrix_free( VdT );
  free( Wd );
}

// Main routine. Controls the iterations. Called from analysis_module.c: analysis_module_updateA()
void zzz_enkf_updateA(void * module_data ,  matrix_type * A ,  matrix_type * S ,  matrix_type * R ,  matrix_type * dObs ,  matrix_type * E ,  matrix_type * D) {
// A : ensemble matrix
// R : Obs error cov inv?
// S : measured ensemble
// dObs: observed data
// E : perturbations for obs
// D = dObs + E - S : Innovations (wrt pert. obs)

  double Sk_new;   // Mismatch
  double  Std_new; // Std dev(Mismatch)
  zzz_enkf_data_type * data = zzz_enkf_data_safe_cast( module_data );
  int nrobs                 = matrix_get_rows( S );           // Num obs
  int ens_size              = matrix_get_columns( S );        // N
  double nsc                = 1/sqrt(ens_size-1);             // Scale factor
  matrix_type * Cd          = matrix_alloc( nrobs, nrobs );   // Cov(E), where E = measurement perturbations?
 
  
  enkf_linalg_Covariance(Cd ,E ,nsc, nrobs); // Cd = SampCov(E) (including (N-1) normalization)
  matrix_inv(Cd); // In-place inversion

  zzz_enkf_open_log_file(data);

	fprintf(stdout, "**********************\n");
	fprintf(stdout, "Iteration number %d\n", data->iteration_nr);

	matrix_type * SmeanMat = matrix_alloc(nrobs,1);
	//vector_type * SmeanVec = vector_alloc_NULL_initialized(nrobs);

	int i; 
	for ( i=0; i < nrobs; i++) {
		double row_mean = matrix_get_row_sum(S , i) / ens_size;
    matrix_iset(SmeanMat , i , 0, row_mean );
		//SmeanVec[i] = row_mean;
	}

	zzz_enkf_log_line( data , "Here comes the cavalry\n");
	matrix_pretty_fprint(S , "S mat" , "%4.2f " , data->log_stream);
	matrix_pretty_fprint(SmeanMat , "Mean mat" , "%4.2f " , data->log_stream);

  if (data->iteration_nr == 0) {
		// ITERATION 0
    zzz_enkf_updateA_iter0(data , A , S , R , dObs , E , D , Cd);
    data->iteration_nr++;
  } else {
		// ITERATION 1,2....
    int nrmin           = util_int_min( ens_size , nrobs);      // Min(p,N)
    matrix_type * Ud    = matrix_alloc( nrobs , nrmin    );     // Left singular vectors.  */
    matrix_type * VdT   = matrix_alloc( nrmin , ens_size );     // Right singular vectors. */
    double * Wd         = util_calloc( nrmin , sizeof * Wd );   // Singular values, vector
    matrix_type * Skm   = matrix_alloc(ens_size,ens_size);      // Mismatch
    matrix_type * Acopy = matrix_alloc_copy (A);                // Copy of A
    Sk_new              = enkf_linalg_data_mismatch(D,Cd,Skm);  // Skm = D'*inv(Cd)*D; Sk_new = trace(Skm)/N
    Std_new             = matrix_diag_std(Skm,Sk_new);          // Standard deviation of mismatches.
    
		// Lambda = Normalized data mismatch (rounded)
    if (data->lambda_recalculate)
      data->lambda = pow(10 , floor(log10(Sk_new / (2*nrobs))) );
    
		zzz_enkf_log_line( data , "Prior0 size: %d x %d\n", matrix_get_rows(data->prior0), matrix_get_columns(data->prior0));

		// active_prior = prior0
    zzz_enkf_common_recover_state( data->prior0 , data->active_prior , data->ens_mask );

		// Accept/Reject update? Lambda calculation.
    {
      bool mismatch_reduced = false;
      bool std_reduced = false;

      if (Sk_new < data->Sk)
        mismatch_reduced = true;
      
      if (Std_new <= data->Std)
        std_reduced = true;

      if (data->log_stream){
        zzz_enkf_log_line( data , "\n%-d-%-4d %-7.3f %-7.3f %-7.3f %-7.3f \n", data->iteration_nr, data->iteration_nr+1,  data->lambda, Sk_new, data->Sk, Std_new);
      }

      if (mismatch_reduced) {
        /*
          Stop check: if ( (1- (Sk_new/data->Sk)) < .0001)  // check convergence ** model change norm has to be added in this!!
        */

				// Reduce lambda
				if (std_reduced) 
					data->lambda = data->lambda * data->lambda_reduce_factor;

				// data->state = A
				zzz_enkf_common_store_state(data->state , A , data->ens_mask );

				data->Sk = Sk_new;
				data->Std=Std_new;
				data->iteration_nr++;
			} else {
				// Increase lambda
				data->lambda = data->lambda * data->lambda_increase_factor;
				// A = data->state
				zzz_enkf_common_recover_state( data->state , A , data->ens_mask );
			}
		}

		// Update from data mismatch
    zzz_enkf_initA__(data , A , S , Cd , E , D , Ud , Wd , VdT);

		// Update from prior mismatch
    if (data->use_prior) {
      zzz_enkf_init_Csc( data );
      zzz_enkf_init2__(data , A , Acopy , Wd , VdT);
    }

		//
    matrix_free(Acopy);
    matrix_free(Skm);
    matrix_free( Ud );
    matrix_free( VdT );
    free( Wd );
  }

  if (data->lambda < data->lambda_min)
    data->lambda = data->lambda_min;


  if (data->log_stream)
    fclose( data->log_stream );
                 
  matrix_free(Cd);
}

// Called from analysis_module.c: analysis_module_init_update()
void zzz_enkf_init_update(void * arg ,  const bool_vector_type * ens_mask ,  const matrix_type * S ,  const matrix_type * R ,  const matrix_type * dObs ,  const matrix_type * E ,  const matrix_type * D ) {
  
  zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast( arg );
  bool_vector_memcpy( module_data->ens_mask , ens_mask );
}







/** GET/SET access methods **/
bool zzz_enkf_set_double( void * arg , const char * var_name , double value) {
  zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_TRUNCATION_KEY_) == 0)
      zzz_enkf_set_truncation( module_data , value );
    else if (strcmp( var_name , LAMBDA_INCREASE_FACTOR_KEY) == 0)
      zzz_enkf_set_lambda_increase_factor( module_data , value );
    else if (strcmp( var_name , LAMBDA_REDUCE_FACTOR_KEY) == 0)
      zzz_enkf_set_lambda_reduce_factor( module_data , value );
    else if (strcmp( var_name , LAMBDA0_KEY) == 0)
      zzz_enkf_set_lambda0( module_data , value );
    else if (strcmp( var_name , LAMBDA_MIN_KEY) == 0)
      zzz_enkf_set_lambda_min( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

bool zzz_enkf_set_int( void * arg , const char * var_name , int value) {
  zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , ENKF_NCOMP_KEY_) == 0)
      zzz_enkf_set_subspace_dimension( module_data , value );
    else if (strcmp( var_name , ITER_KEY) == 0)
      zzz_enkf_set_iteration_nr( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

bool zzz_enkf_set_bool( void * arg , const char * var_name , bool value) {
  zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , USE_PRIOR_KEY) == 0)
      zzz_enkf_set_use_prior( module_data , value );
    else if (strcmp( var_name , CLEAR_LOG_KEY) == 0)
      zzz_enkf_set_clear_log( module_data , value );
    else if (strcmp( var_name , LAMBDA_RECALCULATE_KEY) == 0)
      zzz_enkf_set_lambda_recalculate( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

bool zzz_enkf_set_string( void * arg , const char * var_name , const char * value) {
  zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , LOG_FILE_KEY) == 0)
      zzz_enkf_set_log_file( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

long zzz_enkf_get_options( void * arg , long flag ) {
  zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast( arg );
  {
    return module_data->option_flags;
  }
}

bool zzz_enkf_has_var( const void * arg, const char * var_name) {
		{
				if (strcmp(var_name , ITER_KEY) == 0)
						return true;
				else if (strcmp(var_name , USE_PRIOR_KEY) == 0)
						return true;
				else if (strcmp(var_name , LAMBDA_INCREASE_FACTOR_KEY) == 0)
						return true;
				else if (strcmp(var_name , LAMBDA_REDUCE_FACTOR_KEY) == 0)
						return true;
				else if (strcmp(var_name , LAMBDA0_KEY) == 0)
						return true;
				else if (strcmp(var_name , LAMBDA_MIN_KEY) == 0)
						return true;
				else if (strcmp(var_name , LAMBDA_RECALCULATE_KEY) == 0)
						return true;
				else if (strcmp(var_name , ENKF_TRUNCATION_KEY_) == 0)
						return true;
				else if (strcmp(var_name , LOG_FILE_KEY) == 0)
						return true;
				else if (strcmp(var_name , CLEAR_LOG_KEY) == 0)
						return true;
				else
						return false;
		}
}

int zzz_enkf_get_int( const void * arg, const char * var_name) {
  const zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast_const( arg );
  {
    if (strcmp(var_name , ITER_KEY) == 0)
      return module_data->iteration_nr;
    else
      return -1;
  }
}

void * zzz_enkf_get_ptr( const void * arg , const char * var_name ) {
  const zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast_const( arg );
  {
    if (strcmp(var_name , LOG_FILE_KEY) == 0)
      return module_data->log_file;
    else
      return NULL;
  }
}

bool zzz_enkf_get_bool( const void * arg, const char * var_name) {
		const zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast_const( arg );
		{
				if (strcmp(var_name , USE_PRIOR_KEY) == 0)
						return module_data->use_prior;
				else if (strcmp(var_name , CLEAR_LOG_KEY) == 0) 
						return module_data->clear_log;
				else if (strcmp(var_name , LAMBDA_RECALCULATE_KEY) == 0) 
						return module_data->lambda_recalculate;
				else
						return false;
		}
}

double zzz_enkf_get_double( const void * arg, const char * var_name) {
  const zzz_enkf_data_type * module_data = zzz_enkf_data_safe_cast_const( arg );
  {
    if (strcmp(var_name , LAMBDA_REDUCE_FACTOR_KEY) == 0)
      return module_data->lambda_reduce_factor;
    if (strcmp(var_name , LAMBDA_INCREASE_FACTOR_KEY) == 0)
      return module_data->lambda_increase_factor;
    if (strcmp(var_name , LAMBDA0_KEY) == 0)
      return module_data->lambda0;
    if (strcmp(var_name , LAMBDA_MIN_KEY) == 0)
      return module_data->lambda_min;
    if (strcmp(var_name , ENKF_TRUNCATION_KEY_) == 0)
      return module_data->truncation;
    else
      return -1;
  }
}






#ifdef INTERNAL_LINK
#define SYMBOL_TABLE zzz_enkf_symbol_table
#else
#define SYMBOL_TABLE EXTERNAL_MODULE_SYMBOL
#endif


analysis_table_type SYMBOL_TABLE = {
  .alloc           = zzz_enkf_data_alloc,
  .freef           = zzz_enkf_data_free,
  .set_int         = zzz_enkf_set_int , 
  .set_double      = zzz_enkf_set_double , 
  .set_bool        = zzz_enkf_set_bool, 
  .set_string      = zzz_enkf_set_string,
  .get_options     = zzz_enkf_get_options , 
  .initX           = NULL,
  .updateA         = zzz_enkf_updateA ,  
  .init_update     = zzz_enkf_init_update ,
  .complete_update = NULL,
  .has_var         = zzz_enkf_has_var,
  .get_int         = zzz_enkf_get_int,
  .get_double      = zzz_enkf_get_double,
  .get_bool        = zzz_enkf_get_bool,
  .get_ptr         = zzz_enkf_get_ptr,
};

