// The documentation is written for "doxygen"
// download from http://ftp.stack.nl/pub/users/dimitri/doxygen-1.8.10-setup.exe
// or http://www.stack.nl/~dimitri/doxygen/download.html
//
// Load the Doxygen file, which is the configuration file
//  
// The output is written into /html (should we put it into .gitignore?)

/**
\mainpage MLIP Developer Documentation

\section coding Coding Conventions

\subsection coding_naming Naming Conventions

Files: `my_useful_class.cpp` or `my_useful_class.h`

Classes: `LinearRegression` or  `ExitStatus` (enum)

Variables:
```
local_variable
struct_data_member
class_data_member
pointers must have "p_" prefix
p_linear_system   
- avoid other prefixes (like psz_)
```

Constants:
```
const int DAYS_IN_A_WEEK = 7; (same as Enumerator names)
WeekDay::MONDAY
```

Functions: Regular functions have mixed case; "cheap" functions may use lower case with underscores.
```
AddTableEntry()
DeleteUrl()
bool is_empty()
```

Abbreviations: use only approved abbreviations:

- cfg = configuration
- cfgs = configurations
- pos = position, positions
- nbh = neighborhood
- nbhs = neighborhoods
- vel = velocity
- wgt = weight
- project/potential names: mlip, mtp, mtpr
- common abbreviations in the field: slae, rhs
    - Matrix regression_slae
    - SolveSlaeRegularized
=============
I do NOT approve below  -Alexander Shapeev:
	- cnt: to be changed to count
	- cntr: to be changed to count
	- fnm: change to filename
	- ene: change to energy
	- frc: change to forces
	- str: change to stress or stresses
	- eqtn: change to equation

Variable names with special meaning:
- a, b: indexing vector/matrix coordinates (from 0 to 2)
- ind, i, j: loop variables

\subsection coding_other Other:

\using variables
prefer Array1D, Array2D, Array3D, Array4D, ... instead of double* (with 'new' and 'delete')
prefer std::string instead of char*, use const std::string where vatiable is not changing

```


\subsection formatting Formatting:

- line length = 100  (except comments, but comment should start within 100 symbols)
- tabulation: 4
- indentation: 4 either tab or four spaces
- public/protected...: 2

```
class MyObject {
  public:
    MyObject...
  protected:
    ...
};
```
- #directives: left edge
#ifndef _MLIP_CONFIGURATION
#   define _MLIP_CONFIGURATION
...
#endif // _MLIP_CONFIGURATION
```

\subsection indexing Indexing

- spatial dimensions: a, b (but vectorial operations are preferred)
- other indexes: i, j, ...

\subsection Comments either according to doxygen or not
- Comments may exceed limitations for symbols in lines
- be in the process of converting comments describing functions, members, etc., to the goxigen format
- use `\\!` and `\\!<` (try to avoid `\*!`)

\subsection compiletime Compile Time Options

MLIP_DEBUG switches on some checks that can help to debug a newly written code

\section errors Errors and Warnings

For MLIP to wait for the Enter key to press in warnings and errors,
set an environment variable

    MLIP_WAIT_FOR_KEYPRESS=true

\section site_energy Site Energy

Site energy can be calculated in VASP as \f$E _\ell(x) = \sum _{i=1}^N ...\f$

*/
