/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 */

#include "comm.h"


//#ifdef MLIP_MPI
MPI_data mpi;

void MPI_data::InitComm(MPI_Comm _comm)
{
    comm = _comm;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
#ifdef MLIP_MPI
    fnm_ending = "_" + std::to_string(rank);
#else
    fnm_ending = "";
#endif
}

//#endif
