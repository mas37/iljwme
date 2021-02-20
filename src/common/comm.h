/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 */

// A set of structures and functions with lightweight functional interface to a native mpi implementaions.

#ifndef MLIP_COMM_H
#define MLIP_COMM_H

//#ifdef MLIP_MPI
#include "mpi_stubs.h"
#include <string>


// A structure for any mpi communicator
struct MPI_data
{
    int rank = 0;       // Number of ranks in the mpi group.
    int size = 1;       // Number of this CPU in the mpi group.
    std::string fnm_ending = "";
    MPI_Comm comm;  // mpi communication group
    void InitComm(MPI_Comm _comm);   // Initialization, updating the object
};

// Common communicator
extern MPI_data mpi; 

#endif // MLIP_COMM_H
