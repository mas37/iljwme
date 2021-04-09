/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev
 */

#include "radial_basis.h"


#include <cmath>

using namespace std;


void AnyRadialBasis::ReadRadialBasis(std::ifstream & ifs)
{
	if ((!ifs.is_open()) || (ifs.eof()))
		ERROR("RadialBasis::ReadRadialBasis: Can't load radial basis");

	string tmpstr;

	// reading min_dist / scaling
	ifs >> tmpstr;
	if (tmpstr == "scaling") {
		ifs.ignore(2);
		ifs >> scaling;
		if (ifs.fail())
			ERROR("Error reading .mtp file");
		ifs >> tmpstr;
	}

	if (tmpstr != "min_dist")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> min_dist;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	// reading max_dist 
	ifs >> tmpstr;
	if (tmpstr != "max_dist")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> max_dist;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	// reading rb_size 
	ifs >> tmpstr;
	if (tmpstr != "radial_basis_size")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> rb_size;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	rb_vals.resize(rb_size);
	rb_ders.resize(rb_size);
}

void AnyRadialBasis::WriteRadialBasis(std::ofstream & ofs)
{
	if (!ofs.is_open())
		ERROR("RadialBasis::WriteRadialBasis: Output stream isn't open");

	ofs << "radial_basis_type = " << GetRBTypeString() << '\n';
	if (scaling != 1.0)
		ofs << "\tscaling = " << scaling << '\n';
	ofs << "\tmin_dist = " << min_dist << '\n'
		<< "\tmax_dist = " << max_dist << '\n'
		<< "\tradial_basis_size = " << rb_size << '\n';
}

AnyRadialBasis::AnyRadialBasis(double _min_dist, double _max_dist, int _size)
	: rb_size(_size), min_dist(_min_dist), max_dist(_max_dist)
{
	rb_vals.resize(rb_size);
	rb_ders.resize(rb_size);
}

AnyRadialBasis::AnyRadialBasis(std::ifstream & ifs)
{
	ReadRadialBasis(ifs);
}

void RadialBasis_Shapeev::InitShapeevRB()
{
#ifdef MLIP_DEBUG
	if (rb_size > ALLOCATED_DEGREE) {
		ERROR("error: RadialBasis::MAX_DEGREE > RadialBasis::ALLOCATED_DEGREE");
	}
#endif

	rb_coeffs[0][0] = (sqrt(21)*pow(max_dist, 2)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 2));
	rb_coeffs[0][1] = -((sqrt(21)*max_dist*pow(min_dist, 2)) / pow(max_dist - min_dist, 2));
	rb_coeffs[0][2] = (sqrt(21)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 2));
	rb_coeffs[0][3] = 0;
	rb_coeffs[0][4] = 0;
	rb_coeffs[0][5] = 0;
	rb_coeffs[0][6] = 0;
	rb_coeffs[0][7] = 0;
	rb_coeffs[0][8] = 0;
	rb_coeffs[0][9] = 0;
	rb_coeffs[0][10] = 0;
	rb_coeffs[0][11] = 0;
	rb_coeffs[0][12] = 0;
	rb_coeffs[0][13] = 0;
	rb_coeffs[1][0] = (-3 * sqrt(7)*pow(max_dist, 2)*pow(min_dist, 2)*(max_dist + 3 * min_dist)) / (4.*pow(max_dist - min_dist, 3));
	rb_coeffs[1][1] = (9 * sqrt(7)*max_dist*pow(min_dist, 2)*(max_dist + min_dist)) / (2.*pow(max_dist - min_dist, 3));
	rb_coeffs[1][2] = (9 * sqrt(7)*pow(min_dist, 2)*(3 * max_dist + min_dist)) / (4.*pow(-max_dist + min_dist, 3));
	rb_coeffs[1][3] = (3 * sqrt(7)*pow(min_dist, 2)) / pow(max_dist - min_dist, 3);
	rb_coeffs[1][4] = 0;
	rb_coeffs[1][5] = 0;
	rb_coeffs[1][6] = 0;
	rb_coeffs[1][7] = 0;
	rb_coeffs[1][8] = 0;
	rb_coeffs[1][9] = 0;
	rb_coeffs[1][10] = 0;
	rb_coeffs[1][11] = 0;
	rb_coeffs[1][12] = 0;
	rb_coeffs[1][13] = 0;
	rb_coeffs[2][0] = (sqrt(3.6666666666666665)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 2) + 7 * max_dist*min_dist + 7 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][1] = -((sqrt(3.6666666666666665)*max_dist*pow(min_dist, 2)*(11 * pow(max_dist, 2) + 35 * max_dist*min_dist + 14 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4));
	rb_coeffs[2][2] = (sqrt(3.6666666666666665)*pow(min_dist, 2)*(34 * pow(max_dist, 2) + 49 * max_dist*min_dist + 7 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][3] = -((sqrt(33)*pow(min_dist, 2)*(13 * max_dist + 7 * min_dist)) / pow(max_dist - min_dist, 4));
	rb_coeffs[2][4] = (5 * sqrt(33)*pow(min_dist, 2)) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][5] = 0;
	rb_coeffs[2][6] = 0;
	rb_coeffs[2][7] = 0;
	rb_coeffs[2][8] = 0;
	rb_coeffs[2][9] = 0;
	rb_coeffs[2][10] = 0;
	rb_coeffs[2][11] = 0;
	rb_coeffs[2][12] = 0;
	rb_coeffs[2][13] = 0;
	rb_coeffs[3][0] = (-3 * sqrt(6.5)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 3) + 12 * pow(max_dist, 2)*min_dist + 28 * max_dist*pow(min_dist, 2) + 14 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][1] = (3 * sqrt(6.5)*max_dist*pow(min_dist, 2)*(17 * pow(max_dist, 3) + 104 * pow(max_dist, 2)*min_dist + 126 * max_dist*pow(min_dist, 2) + 28 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][2] = (3 * sqrt(6.5)*pow(min_dist, 2)*(43 * pow(max_dist, 3) + 141 * pow(max_dist, 2)*min_dist + 84 * max_dist*pow(min_dist, 2) + 7 * pow(min_dist, 3))) / (2.*pow(-max_dist + min_dist, 5));
	rb_coeffs[3][3] = (15 * sqrt(6.5)*pow(min_dist, 2)*(18 * pow(max_dist, 2) + 30 * max_dist*min_dist + 7 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][4] = (165 * sqrt(6.5)*pow(min_dist, 2)*(3 * max_dist + 2 * min_dist)) / (4.*pow(-max_dist + min_dist, 5));
	rb_coeffs[3][5] = (165 * sqrt(6.5)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][6] = 0;
	rb_coeffs[3][7] = 0;
	rb_coeffs[3][8] = 0;
	rb_coeffs[3][9] = 0;
	rb_coeffs[3][10] = 0;
	rb_coeffs[3][11] = 0;
	rb_coeffs[3][12] = 0;
	rb_coeffs[3][13] = 0;
	rb_coeffs[4][0] = (sqrt(0.6)*pow(max_dist, 2)*pow(min_dist, 2)*(5 * pow(max_dist, 4) + 90 * pow(max_dist, 3)*min_dist + 360 * pow(max_dist, 2)*pow(min_dist, 2) + 420 * max_dist*pow(min_dist, 3) + 126 * pow(min_dist, 4))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][1] = (-3 * sqrt(0.6)*max_dist*pow(min_dist, 2)*(20 * pow(max_dist, 4) + 195 * pow(max_dist, 3)*min_dist + 450 * pow(max_dist, 2)*pow(min_dist, 2) + 294 * max_dist*pow(min_dist, 3) + 42 * pow(min_dist, 4))) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][2] = (3 * sqrt(0.6)*pow(min_dist, 2)*(295 * pow(max_dist, 4) + 1680 * pow(max_dist, 3)*min_dist + 2232 * pow(max_dist, 2)*pow(min_dist, 2) + 756 * max_dist*pow(min_dist, 3) + 42 * pow(min_dist, 4))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][3] = (-22 * sqrt(0.6)*pow(min_dist, 2)*(65 * pow(max_dist, 3) + 216 * pow(max_dist, 2)*min_dist + 153 * max_dist*pow(min_dist, 2) + 21 * pow(min_dist, 3))) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][4] = (33 * sqrt(0.6)*pow(min_dist, 2)*(137 * pow(max_dist, 2) + 246 * max_dist*min_dist + 72 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][5] = (-429 * sqrt(0.6)*pow(min_dist, 2)*(4 * max_dist + 3 * min_dist)) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][6] = (1001 * sqrt(0.6)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][7] = 0;
	rb_coeffs[4][8] = 0;
	rb_coeffs[4][9] = 0;
	rb_coeffs[4][10] = 0;
	rb_coeffs[4][11] = 0;
	rb_coeffs[4][12] = 0;
	rb_coeffs[4][13] = 0;
	rb_coeffs[5][0] = -(sqrt(62.333333333333336)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 5) + 25 * pow(max_dist, 4)*min_dist + 150 * pow(max_dist, 3)*pow(min_dist, 2) + 300 * pow(max_dist, 2)*pow(min_dist, 3) + 210 * max_dist*pow(min_dist, 4) + 42 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][1] = (sqrt(62.333333333333336)*max_dist*pow(min_dist, 2)*(16 * pow(max_dist, 5) + 225 * pow(max_dist, 4)*min_dist + 825 * pow(max_dist, 3)*pow(min_dist, 2) + 1020 * pow(max_dist, 2)*pow(min_dist, 3) + 420 * max_dist*pow(min_dist, 4) + 42 * pow(min_dist, 5))) / (2.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][2] = -(sqrt(561)*pow(min_dist, 2)*(107 * pow(max_dist, 5) + 925 * pow(max_dist, 4)*min_dist + 2120 * pow(max_dist, 3)*pow(min_dist, 2) + 1580 * pow(max_dist, 2)*pow(min_dist, 3) + 350 * max_dist*pow(min_dist, 4) + 14 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][3] = (5 * sqrt(62.333333333333336)*pow(min_dist, 2)*(73 * pow(max_dist, 4) + 397 * pow(max_dist, 3)*min_dist + 555 * pow(max_dist, 2)*pow(min_dist, 2) + 228 * max_dist*pow(min_dist, 3) + 21 * pow(min_dist, 4))) / pow(max_dist - min_dist, 7);
	rb_coeffs[5][4] = (-65 * sqrt(62.333333333333336)*pow(min_dist, 2)*(53 * pow(max_dist, 3) + 177 * pow(max_dist, 2)*min_dist + 138 * max_dist*pow(min_dist, 2) + 24 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][5] = (91 * sqrt(561)*pow(min_dist, 2)*(8 * pow(max_dist, 2) + 15 * max_dist*min_dist + 5 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][6] = (-91 * sqrt(62.333333333333336)*pow(min_dist, 2)*(31 * max_dist + 25 * min_dist)) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][7] = (182 * sqrt(62.333333333333336)*pow(min_dist, 2)) / pow(max_dist - min_dist, 7);
	rb_coeffs[5][8] = 0;
	rb_coeffs[5][9] = 0;
	rb_coeffs[5][10] = 0;
	rb_coeffs[5][11] = 0;
	rb_coeffs[5][12] = 0;
	rb_coeffs[5][13] = 0;
	rb_coeffs[6][0] = (sqrt(4.071428571428571)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 6) + 33 * pow(max_dist, 5)*min_dist + 275 * pow(max_dist, 4)*pow(min_dist, 2) + 825 * pow(max_dist, 3)*pow(min_dist, 3) + 990 * pow(max_dist, 2)*pow(min_dist, 4) + 462 * max_dist*pow(min_dist, 5) + 66 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][1] = -((sqrt(4.071428571428571)*max_dist*pow(min_dist, 2)*(41 * pow(max_dist, 6) + 781 * pow(max_dist, 5)*min_dist + 4125 * pow(max_dist, 4)*pow(min_dist, 2) + 8085 * pow(max_dist, 3)*pow(min_dist, 3) + 6270 * pow(max_dist, 2)*pow(min_dist, 4) + 1782 * max_dist*pow(min_dist, 5) + 132 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8));
	rb_coeffs[6][2] = (3 * sqrt(16.285714285714285)*pow(min_dist, 2)*(89 * pow(max_dist, 6) + 1078 * pow(max_dist, 5)*min_dist + 3740 * pow(max_dist, 4)*pow(min_dist, 2) + 4785 * pow(max_dist, 3)*pow(min_dist, 3) + 2310 * pow(max_dist, 2)*pow(min_dist, 4) + 363 * max_dist*pow(min_dist, 5) + 11 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][3] = (-13 * sqrt(16.285714285714285)*pow(min_dist, 2)*(124 * pow(max_dist, 5) + 990 * pow(max_dist, 4)*min_dist + 2255 * pow(max_dist, 3)*pow(min_dist, 2) + 1815 * pow(max_dist, 2)*pow(min_dist, 3) + 495 * max_dist*pow(min_dist, 4) + 33 * pow(min_dist, 5))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][4] = (65 * sqrt(4.071428571428571)*pow(min_dist, 2)*(161 * pow(max_dist, 4) + 847 * pow(max_dist, 3)*min_dist + 1221 * pow(max_dist, 2)*pow(min_dist, 2) + 561 * max_dist*pow(min_dist, 3) + 66 * pow(min_dist, 4))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][5] = (-39 * sqrt(4.071428571428571)*pow(min_dist, 2)*(497 * pow(max_dist, 3) + 1661 * pow(max_dist, 2)*min_dist + 1375 * max_dist*pow(min_dist, 2) + 275 * pow(min_dist, 3))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][6] = (26 * sqrt(16.285714285714285)*pow(min_dist, 2)*(394 * pow(max_dist, 2) + 759 * max_dist*min_dist + 275 * pow(min_dist, 2))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][7] = (-442 * sqrt(16.285714285714285)*pow(min_dist, 2)*(13 * max_dist + 11 * min_dist)) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][8] = (1326 * sqrt(16.285714285714285)*pow(min_dist, 2)) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][9] = 0;
	rb_coeffs[6][10] = 0;
	rb_coeffs[6][11] = 0;
	rb_coeffs[6][12] = 0;
	rb_coeffs[6][13] = 0;
	rb_coeffs[7][0] = -(sqrt(273)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 7) + 42 * pow(max_dist, 6)*min_dist + 462 * pow(max_dist, 5)*pow(min_dist, 2) + 1925 * pow(max_dist, 4)*pow(min_dist, 3) + 3465 * pow(max_dist, 3)*pow(min_dist, 4) + 2772 * pow(max_dist, 2)*pow(min_dist, 5) + 924 * max_dist*pow(min_dist, 6) + 99 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][1] = (3 * sqrt(273)*max_dist*pow(min_dist, 2)*(17 * pow(max_dist, 7) + 420 * pow(max_dist, 6)*min_dist + 3003 * pow(max_dist, 5)*pow(min_dist, 2) + 8470 * pow(max_dist, 4)*pow(min_dist, 3) + 10395 * pow(max_dist, 3)*pow(min_dist, 4) + 5544 * pow(max_dist, 2)*pow(min_dist, 5) + 1155 * max_dist*pow(min_dist, 6) + 66 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][2] = (-3 * sqrt(273)*pow(min_dist, 2)*(278 * pow(max_dist, 7) + 4473 * pow(max_dist, 6)*min_dist + 21714 * pow(max_dist, 5)*pow(min_dist, 2) + 41965 * pow(max_dist, 4)*pow(min_dist, 3) + 34650 * pow(max_dist, 3)*pow(min_dist, 4) + 11781 * pow(max_dist, 2)*pow(min_dist, 5) + 1386 * max_dist*pow(min_dist, 6) + 33 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][3] = (7 * sqrt(273)*pow(min_dist, 2)*(917 * pow(max_dist, 6) + 10038 * pow(max_dist, 5)*min_dist + 33495 * pow(max_dist, 4)*pow(min_dist, 2) + 43780 * pow(max_dist, 3)*pow(min_dist, 3) + 23265 * pow(max_dist, 2)*pow(min_dist, 4) + 4554 * max_dist*pow(min_dist, 5) + 231 * pow(min_dist, 6))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][4] = (-105 * sqrt(273)*pow(min_dist, 2)*(259 * pow(max_dist, 5) + 1953 * pow(max_dist, 4)*min_dist + 4422 * pow(max_dist, 3)*pow(min_dist, 2) + 3740 * pow(max_dist, 2)*pow(min_dist, 3) + 1155 * max_dist*pow(min_dist, 4) + 99 * pow(min_dist, 5))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][5] = (21 * sqrt(273)*pow(min_dist, 2)*(1624 * pow(max_dist, 4) + 8328 * pow(max_dist, 3)*min_dist + 12243 * pow(max_dist, 2)*pow(min_dist, 2) + 6050 * max_dist*pow(min_dist, 3) + 825 * pow(min_dist, 4))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][6] = (-119 * sqrt(273)*pow(min_dist, 2)*(436 * pow(max_dist, 3) + 1455 * pow(max_dist, 2)*min_dist + 1254 * max_dist*pow(min_dist, 2) + 275 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][7] = (153 * sqrt(273)*pow(min_dist, 2)*(307 * pow(max_dist, 2) + 602 * max_dist*min_dist + 231 * pow(min_dist, 2))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][8] = (-2907 * sqrt(273)*pow(min_dist, 2)*(8 * max_dist + 7 * min_dist)) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][9] = (4845 * sqrt(273)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][10] = 0;
	rb_coeffs[7][11] = 0;
	rb_coeffs[7][12] = 0;
	rb_coeffs[7][13] = 0;
	rb_coeffs[8][0] = (sqrt(161)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 8) + 52 * pow(max_dist, 7)*min_dist + 728 * pow(max_dist, 6)*pow(min_dist, 2) + 4004 * pow(max_dist, 5)*pow(min_dist, 3) + 10010 * pow(max_dist, 4)*pow(min_dist, 4) + 12012 * pow(max_dist, 3)*pow(min_dist, 5) + 6864 * pow(max_dist, 2)*pow(min_dist, 6) + 1716 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (6.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][1] = -(sqrt(161)*max_dist*pow(min_dist, 2)*(31 * pow(max_dist, 8) + 962 * pow(max_dist, 7)*min_dist + 8918 * pow(max_dist, 6)*pow(min_dist, 2) + 34034 * pow(max_dist, 5)*pow(min_dist, 3) + 60060 * pow(max_dist, 4)*pow(min_dist, 4) + 50622 * pow(max_dist, 3)*pow(min_dist, 5) + 19734 * pow(max_dist, 2)*pow(min_dist, 6) + 3146 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][2] = (sqrt(161)*pow(min_dist, 2)*(1241 * pow(max_dist, 8) + 25532 * pow(max_dist, 7)*min_dist + 164528 * pow(max_dist, 6)*pow(min_dist, 2) + 444444 * pow(max_dist, 5)*pow(min_dist, 3) + 553410 * pow(max_dist, 4)*pow(min_dist, 4) + 320892 * pow(max_dist, 3)*pow(min_dist, 5) + 81224 * pow(max_dist, 2)*pow(min_dist, 6) + 7436 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (6.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][3] = (-10 * sqrt(161)*pow(min_dist, 2)*(591 * pow(max_dist, 7) + 8463 * pow(max_dist, 6)*min_dist + 38675 * pow(max_dist, 5)*pow(min_dist, 2) + 73931 * pow(max_dist, 4)*pow(min_dist, 3) + 63635 * pow(max_dist, 3)*pow(min_dist, 4) + 24167 * pow(max_dist, 2)*pow(min_dist, 5) + 3575 * max_dist*pow(min_dist, 6) + 143 * pow(min_dist, 7))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][4] = (10 * sqrt(161)*pow(min_dist, 2)*(3150 * pow(max_dist, 6) + 32032 * pow(max_dist, 5)*min_dist + 103792 * pow(max_dist, 4)*pow(min_dist, 2) + 137566 * pow(max_dist, 3)*pow(min_dist, 3) + 77935 * pow(max_dist, 2)*pow(min_dist, 4) + 17446 * max_dist*pow(min_dist, 5) + 1144 * pow(min_dist, 6))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][5] = (-68 * sqrt(161)*pow(min_dist, 2)*(1498 * pow(max_dist, 5) + 10816 * pow(max_dist, 4)*min_dist + 24349 * pow(max_dist, 3)*pow(min_dist, 2) + 21307 * pow(max_dist, 2)*pow(min_dist, 3) + 7150 * max_dist*pow(min_dist, 4) + 715 * pow(min_dist, 5))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][6] = (34 * sqrt(161)*pow(min_dist, 2)*(6102 * pow(max_dist, 4) + 30654 * pow(max_dist, 3)*min_dist + 45656 * pow(max_dist, 2)*pow(min_dist, 2) + 23738 * max_dist*pow(min_dist, 3) + 3575 * pow(min_dist, 4))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][7] = (-1292 * sqrt(161)*pow(min_dist, 2)*(207 * pow(max_dist, 3) + 689 * pow(max_dist, 2)*min_dist + 611 * max_dist*pow(min_dist, 2) + 143 * pow(min_dist, 3))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][8] = (1615 * sqrt(161)*pow(min_dist, 2)*(131 * pow(max_dist, 2) + 260 * max_dist*min_dist + 104 * pow(min_dist, 2))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][9] = (-3230 * sqrt(161)*pow(min_dist, 2)*(29 * max_dist + 26 * min_dist)) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][10] = (17765 * sqrt(161)*pow(min_dist, 2)) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][11] = 0;
	rb_coeffs[8][12] = 0;
	rb_coeffs[8][13] = 0;
	rb_coeffs[9][0] = -(sqrt(3)*pow(max_dist, 2)*pow(min_dist, 2)*(5 * pow(max_dist, 9) + 315 * pow(max_dist, 8)*min_dist + 5460 * pow(max_dist, 7)*pow(min_dist, 2) + 38220 * pow(max_dist, 6)*pow(min_dist, 3) + 126126 * pow(max_dist, 5)*pow(min_dist, 4) + 210210 * pow(max_dist, 4)*pow(min_dist, 5) + 180180 * pow(max_dist, 3)*pow(min_dist, 6) + 77220 * pow(max_dist, 2)*pow(min_dist, 7) + 15015 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][1] = (sqrt(3)*max_dist*pow(min_dist, 2)*(185 * pow(max_dist, 9) + 7035 * pow(max_dist, 8)*min_dist + 81900 * pow(max_dist, 7)*pow(min_dist, 2) + 405132 * pow(max_dist, 6)*pow(min_dist, 3) + 966966 * pow(max_dist, 5)*pow(min_dist, 4) + 1171170 * pow(max_dist, 4)*pow(min_dist, 5) + 720720 * pow(max_dist, 3)*pow(min_dist, 6) + 214500 * pow(max_dist, 2)*pow(min_dist, 7) + 27027 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][2] = -(sqrt(3)*pow(min_dist, 2)*(8885 * pow(max_dist, 9) + 227115 * pow(max_dist, 8)*min_dist + 1870596 * pow(max_dist, 7)*pow(min_dist, 2) + 6703788 * pow(max_dist, 6)*pow(min_dist, 3) + 11657646 * pow(max_dist, 5)*pow(min_dist, 4) + 10180170 * pow(max_dist, 4)*pow(min_dist, 5) + 4384380 * pow(max_dist, 3)*pow(min_dist, 6) + 859716 * pow(max_dist, 2)*pow(min_dist, 7) + 63063 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][3] = (6 * sqrt(3)*pow(min_dist, 2)*(4265 * pow(max_dist, 8) + 77196 * pow(max_dist, 7)*min_dist + 461188 * pow(max_dist, 6)*pow(min_dist, 2) + 1206296 * pow(max_dist, 5)*pow(min_dist, 3) + 1516515 * pow(max_dist, 4)*pow(min_dist, 4) + 930930 * pow(max_dist, 3)*pow(min_dist, 5) + 266266 * pow(max_dist, 2)*pow(min_dist, 6) + 30888 * max_dist*pow(min_dist, 7) + 1001 * pow(min_dist, 8))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][4] = (-51 * sqrt(3)*pow(min_dist, 2)*(3274 * pow(max_dist, 7) + 43022 * pow(max_dist, 6)*min_dist + 187824 * pow(max_dist, 5)*pow(min_dist, 2) + 355810 * pow(max_dist, 4)*pow(min_dist, 3) + 315315 * pow(max_dist, 3)*pow(min_dist, 4) + 129129 * pow(max_dist, 2)*pow(min_dist, 5) + 22022 * max_dist*pow(min_dist, 6) + 1144 * pow(min_dist, 7))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][5] = (714 * sqrt(3)*pow(min_dist, 2)*(942 * pow(max_dist, 6) + 9054 * pow(max_dist, 5)*min_dist + 28665 * pow(max_dist, 4)*pow(min_dist, 2) + 38350 * pow(max_dist, 3)*pow(min_dist, 3) + 22737 * pow(max_dist, 2)*pow(min_dist, 4) + 5577 * max_dist*pow(min_dist, 5) + 429 * pow(min_dist, 6))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][6] = (-6783 * sqrt(3)*pow(min_dist, 2)*(258 * pow(max_dist, 5) + 1800 * pow(max_dist, 4)*min_dist + 4030 * pow(max_dist, 3)*pow(min_dist, 2) + 3614 * pow(max_dist, 2)*pow(min_dist, 3) + 1287 * max_dist*pow(min_dist, 4) + 143 * pow(min_dist, 5))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][7] = (1938 * sqrt(3)*pow(min_dist, 2)*(1545 * pow(max_dist, 4) + 7630 * pow(max_dist, 3)*min_dist + 11466 * pow(max_dist, 2)*pow(min_dist, 2) + 6188 * max_dist*pow(min_dist, 3) + 1001 * pow(min_dist, 4))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][8] = (-969 * sqrt(3)*pow(min_dist, 2)*(6905 * pow(max_dist, 3) + 22911 * pow(max_dist, 2)*min_dist + 20748 * max_dist*pow(min_dist, 2) + 5096 * pow(min_dist, 3))) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][9] = (3553 * sqrt(3)*pow(min_dist, 2)*(661 * pow(max_dist, 2) + 1323 * max_dist*min_dist + 546 * pow(min_dist, 2))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][10] = (-81719 * sqrt(3)*pow(min_dist, 2)*(23 * max_dist + 21 * min_dist)) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][11] = (163438 * sqrt(3)*pow(min_dist, 2)) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][12] = 0;
	rb_coeffs[9][13] = 0;
	rb_coeffs[10][0] = (3 * sqrt(0.5454545454545454)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 10) + 75 * pow(max_dist, 9)*min_dist + 1575 * pow(max_dist, 8)*pow(min_dist, 2) + 13650 * pow(max_dist, 7)*pow(min_dist, 3) + 57330 * pow(max_dist, 6)*pow(min_dist, 4) + 126126 * pow(max_dist, 5)*pow(min_dist, 5) + 150150 * pow(max_dist, 4)*pow(min_dist, 6) + 96525 * pow(max_dist, 3)*pow(min_dist, 7) + 32175 * pow(max_dist, 2)*pow(min_dist, 8) + 5005 * max_dist*pow(min_dist, 9) + 273 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][1] = (-9 * sqrt(0.5454545454545454)*max_dist*pow(min_dist, 2)*(29 * pow(max_dist, 10) + 1325 * pow(max_dist, 9)*min_dist + 18900 * pow(max_dist, 8)*pow(min_dist, 2) + 117390 * pow(max_dist, 7)*pow(min_dist, 3) + 363090 * pow(max_dist, 6)*pow(min_dist, 4) + 594594 * pow(max_dist, 5)*pow(min_dist, 5) + 525525 * pow(max_dist, 4)*pow(min_dist, 6) + 246675 * pow(max_dist, 3)*pow(min_dist, 7) + 57915 * pow(max_dist, 2)*pow(min_dist, 8) + 5915 * max_dist*pow(min_dist, 9) + 182 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][2] = (9 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(822 * pow(max_dist, 10) + 25525 * pow(max_dist, 9)*min_dist + 261135 * pow(max_dist, 8)*pow(min_dist, 2) + 1195740 * pow(max_dist, 7)*pow(min_dist, 3) + 2757300 * pow(max_dist, 6)*pow(min_dist, 4) + 3360357 * pow(max_dist, 5)*pow(min_dist, 5) + 2177175 * pow(max_dist, 4)*pow(min_dist, 6) + 725010 * pow(max_dist, 3)*pow(min_dist, 7) + 113490 * pow(max_dist, 2)*pow(min_dist, 8) + 6825 * max_dist*pow(min_dist, 9) + 91 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][3] = (-255 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(397 * pow(max_dist, 9) + 8847 * pow(max_dist, 8)*min_dist + 66780 * pow(max_dist, 7)*pow(min_dist, 2) + 228228 * pow(max_dist, 6)*pow(min_dist, 3) + 392301 * pow(max_dist, 5)*pow(min_dist, 4) + 351351 * pow(max_dist, 4)*pow(min_dist, 5) + 162162 * pow(max_dist, 3)*pow(min_dist, 6) + 36270 * pow(max_dist, 2)*pow(min_dist, 7) + 3393 * max_dist*pow(min_dist, 8) + 91 * pow(min_dist, 9))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][4] = (2295 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(345 * pow(max_dist, 8) + 5676 * pow(max_dist, 7)*min_dist + 32004 * pow(max_dist, 6)*pow(min_dist, 2) + 81627 * pow(max_dist, 5)*pow(min_dist, 3) + 103285 * pow(max_dist, 4)*pow(min_dist, 4) + 66066 * pow(max_dist, 3)*pow(min_dist, 5) + 20566 * pow(max_dist, 2)*pow(min_dist, 6) + 2769 * max_dist*pow(min_dist, 7) + 117 * pow(min_dist, 8))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][5] = (-8721 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(444 * pow(max_dist, 7) + 5460 * pow(max_dist, 6)*min_dist + 22995 * pow(max_dist, 5)*pow(min_dist, 2) + 43225 * pow(max_dist, 4)*pow(min_dist, 3) + 39130 * pow(max_dist, 3)*pow(min_dist, 4) + 16926 * pow(max_dist, 2)*pow(min_dist, 5) + 3185 * max_dist*pow(min_dist, 6) + 195 * pow(min_dist, 7))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][6] = (20349 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(612 * pow(max_dist, 6) + 5625 * pow(max_dist, 5)*min_dist + 17475 * pow(max_dist, 4)*pow(min_dist, 2) + 23530 * pow(max_dist, 3)*pow(min_dist, 3) + 14430 * pow(max_dist, 2)*pow(min_dist, 4) + 3783 * max_dist*pow(min_dist, 5) + 325 * pow(min_dist, 6))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][7] = (-8721 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(3099 * pow(max_dist, 5) + 21025 * pow(max_dist, 4)*min_dist + 46830 * pow(max_dist, 3)*pow(min_dist, 2) + 42770 * pow(max_dist, 2)*pow(min_dist, 3) + 15925 * max_dist*pow(min_dist, 4) + 1911 * pow(min_dist, 5))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][8] = (43605 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(913 * pow(max_dist, 4) + 4444 * pow(max_dist, 3)*min_dist + 6720 * pow(max_dist, 2)*pow(min_dist, 2) + 3731 * max_dist*pow(min_dist, 3) + 637 * pow(min_dist, 4))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][9] = (-111435 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(352 * pow(max_dist, 3) + 1164 * pow(max_dist, 2)*min_dist + 1071 * max_dist*pow(min_dist, 2) + 273 * pow(min_dist, 3))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][10] = (334305 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(74 * pow(max_dist, 2) + 149 * max_dist*min_dist + 63 * pow(min_dist, 2))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][11] = (-334305 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(27 * max_dist + 25 * min_dist)) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][12] = (1448655 * sqrt(0.5454545454545454)*pow(min_dist, 2)) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][13] = 0;
	rb_coeffs[11][0] = -(sqrt(82.16666666666667)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 11) + 88 * pow(max_dist, 10)*min_dist + 2200 * pow(max_dist, 9)*pow(min_dist, 2) + 23100 * pow(max_dist, 8)*pow(min_dist, 3) + 120120 * pow(max_dist, 7)*pow(min_dist, 4) + 336336 * pow(max_dist, 6)*pow(min_dist, 5) + 528528 * pow(max_dist, 5)*pow(min_dist, 6) + 471900 * pow(max_dist, 4)*pow(min_dist, 7) + 235950 * pow(max_dist, 3)*pow(min_dist, 8) + 62920 * pow(max_dist, 2)*pow(min_dist, 9) + 8008 * max_dist*pow(min_dist, 10) + 364 * pow(min_dist, 11))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][1] = (sqrt(82.16666666666667)*max_dist*pow(min_dist, 2)*(101 * pow(max_dist, 11) + 5456 * pow(max_dist, 10)*min_dist + 93500 * pow(max_dist, 9)*pow(min_dist, 2) + 711480 * pow(max_dist, 8)*pow(min_dist, 3) + 2762760 * pow(max_dist, 7)*pow(min_dist, 4) + 5861856 * pow(max_dist, 6)*pow(min_dist, 5) + 7002996 * pow(max_dist, 5)*pow(min_dist, 6) + 4719000 * pow(max_dist, 4)*pow(min_dist, 7) + 1746030 * pow(max_dist, 3)*pow(min_dist, 8) + 331760 * pow(max_dist, 2)*pow(min_dist, 9) + 28028 * max_dist*pow(min_dist, 10) + 728 * pow(min_dist, 11))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][2] = -(sqrt(82.16666666666667)*pow(min_dist, 2)*(1667 * pow(max_dist, 11) + 61754 * pow(max_dist, 10)*min_dist + 767360 * pow(max_dist, 9)*pow(min_dist, 2) + 4363590 * pow(max_dist, 8)*pow(min_dist, 3) + 12852840 * pow(max_dist, 7)*pow(min_dist, 4) + 20762742 * pow(max_dist, 6)*pow(min_dist, 5) + 18762744 * pow(max_dist, 5)*pow(min_dist, 6) + 9390810 * pow(max_dist, 4)*pow(min_dist, 7) + 2492490 * pow(max_dist, 3)*pow(min_dist, 8) + 318890 * pow(max_dist, 2)*pow(min_dist, 9) + 16016 * max_dist*pow(min_dist, 10) + 182 * pow(min_dist, 11))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][3] = (11 * sqrt(739.5)*pow(min_dist, 2)*(809 * pow(max_dist, 10) + 21740 * pow(max_dist, 9)*min_dist + 201990 * pow(max_dist, 8)*pow(min_dist, 2) + 871920 * pow(max_dist, 7)*pow(min_dist, 3) + 1957410 * pow(max_dist, 6)*pow(min_dist, 4) + 2395484 * pow(max_dist, 5)*pow(min_dist, 5) + 1611610 * pow(max_dist, 4)*pow(min_dist, 6) + 580840 * pow(max_dist, 3)*pow(min_dist, 7) + 104520 * pow(max_dist, 2)*pow(min_dist, 8) + 8060 * max_dist*pow(min_dist, 9) + 182 * pow(min_dist, 10))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][4] = (-1045 * sqrt(739.5)*pow(min_dist, 2)*(157 * pow(max_dist, 9) + 3156 * pow(max_dist, 8)*min_dist + 22272 * pow(max_dist, 7)*pow(min_dist, 2) + 73332 * pow(max_dist, 6)*pow(min_dist, 3) + 124852 * pow(max_dist, 5)*pow(min_dist, 4) + 113932 * pow(max_dist, 4)*pow(min_dist, 5) + 55328 * pow(max_dist, 3)*pow(min_dist, 6) + 13572 * pow(max_dist, 2)*pow(min_dist, 7) + 1482 * max_dist*pow(min_dist, 8) + 52 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][5] = (209 * sqrt(739.5)*pow(min_dist, 2)*(4569 * pow(max_dist, 8) + 69792 * pow(max_dist, 7)*min_dist + 375900 * pow(max_dist, 6)*pow(min_dist, 2) + 939400 * pow(max_dist, 5)*pow(min_dist, 3) + 1193920 * pow(max_dist, 4)*pow(min_dist, 4) + 787696 * pow(max_dist, 3)*pow(min_dist, 5) + 260988 * pow(max_dist, 2)*pow(min_dist, 6) + 39000 * max_dist*pow(min_dist, 7) + 1950 * pow(min_dist, 8))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][6] = (-1463 * sqrt(739.5)*pow(min_dist, 2)*(633 * pow(max_dist, 7) + 7383 * pow(max_dist, 6)*min_dist + 30200 * pow(max_dist, 5)*pow(min_dist, 2) + 56385 * pow(max_dist, 4)*pow(min_dist, 3) + 51870 * pow(max_dist, 3)*pow(min_dist, 4) + 23387 * pow(max_dist, 2)*pow(min_dist, 5) + 4732 * max_dist*pow(min_dist, 6) + 325 * pow(min_dist, 7))) / pow(max_dist - min_dist, 13);
	rb_coeffs[11][7] = (209 * sqrt(739.5)*pow(min_dist, 2)*(11814 * pow(max_dist, 6) + 104698 * pow(max_dist, 5)*min_dist + 320155 * pow(max_dist, 4)*pow(min_dist, 2) + 433020 * pow(max_dist, 3)*pow(min_dist, 3) + 272545 * pow(max_dist, 2)*pow(min_dist, 4) + 75166 * max_dist*pow(min_dist, 5) + 7007 * pow(min_dist, 6))) / pow(max_dist - min_dist, 13);
	rb_coeffs[11][8] = (-4807 * sqrt(739.5)*pow(min_dist, 2)*(3817 * pow(max_dist, 5) + 25300 * pow(max_dist, 4)*min_dist + 56080 * pow(max_dist, 3)*pow(min_dist, 2) + 51940 * pow(max_dist, 2)*pow(min_dist, 3) + 20020 * max_dist*pow(min_dist, 4) + 2548 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][9] = (24035 * sqrt(82.16666666666667)*pow(min_dist, 2)*(2959 * pow(max_dist, 4) + 14224 * pow(max_dist, 3)*min_dist + 21604 * pow(max_dist, 2)*pow(min_dist, 2) + 12264 * max_dist*pow(min_dist, 3) + 2184 * pow(min_dist, 4))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][10] = (-24035 * sqrt(82.16666666666667)*pow(min_dist, 2)*(1303 * pow(max_dist, 3) + 4294 * pow(max_dist, 2)*min_dist + 4000 * max_dist*pow(min_dist, 2) + 1050 * pow(min_dist, 3))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][11] = (28405 * sqrt(82.16666666666667)*pow(min_dist, 2)*(631 * pow(max_dist, 2) + 1276 * max_dist*min_dist + 550 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][12] = (-85215 * sqrt(739.5)*pow(min_dist, 2)*(47 * max_dist + 44 * min_dist)) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][13] = (596505 * sqrt(739.5)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 13));
}

RadialBasis_Shapeev::RadialBasis_Shapeev(double _min_dist, double _max_dist, int _size)
	: AnyRadialBasis(_min_dist, _max_dist, _size)
{
	if (rb_size > ALLOCATED_DEGREE + 1)
		ERROR("RadialBasis error: allocated degree ecceded.");
	InitShapeevRB();
}

RadialBasis_Shapeev::RadialBasis_Shapeev(std::ifstream & ifs)
	: AnyRadialBasis(ifs)
{
	if (rb_size > ALLOCATED_DEGREE + 1)
		ERROR("RadialBasis error: allocated degree ecceded.");
	InitShapeevRB();
}

void RadialBasis_Shapeev::RB_Calc(double r)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	for (int xi = 0; xi < rb_size; xi++) {
		rb_vals[xi] = 0;
		rb_ders[xi] = 0;
		for (int i = -2; i <= xi; i++) {
			rb_vals[xi] += scaling * rb_coeffs[xi][i + 2] * pow(r, i);
			rb_ders[xi] += scaling * rb_coeffs[xi][i + 2] * i * pow(r, i - 1);
		}
	}
}

static const int RADIAL_BASIS_MAX_SIZE = 21;
static const int RADIAL_BASIS_CHEB_SIZE = 26;
const double CHEB_RECURSIVE_COEFFS[RADIAL_BASIS_MAX_SIZE][3][RADIAL_BASIS_CHEB_SIZE] = { {{8.9018563985748373992641359898,-2.0776316424217116897913204927,-0.0333465680440708490770190183,-0.0048748091897258669925637665,-0.000799829997812288671733317,-0.0001422908745729173737609261,-0.000026802249562631672816656,-5.2622963001952907079656e-6,-1.0658134158235160038838e-6,-2.211182159979926745539e-7,-4.67579064795797415365e-8,-1.00419081955720209533e-8,-2.1844854536039879463e-9,-4.803627880198481844e-10,-1.066073869627883311e-10,-2.38481837469261183e-11,-5.3719420060227089e-12,-1.2174563529509179e-12,-2.774096436595483e-13,-6.35161747045299e-14,-1.46059254144378e-14,-3.3719204610329e-15,-7.813269958263e-16,-1.821203449159e-16,-4.46483908373e-17,-9.8949678506e-18},{-0.101276554781729938309987509368,-0.028558692944353252374014829401,-0.004400541588848111882003623624,-0.000730162947248263194948894339,-0.000128643863573766814516103557,-0.000023778846109127720727551711,-4.566771295000720362372234e-6,-9.04383037334599953468988e-7,-1.83605355117404096276666e-7,-3.8041722599505651350377e-8,-8.016264756742480156177e-9,-1.713355193050449421546e-9,-3.70647017402942394568e-10,-8.1016273022141985676e-11,-1.7868389326728508776e-11,-3.972022668122172459e-12,-8.89094099179896304e-13,-2.00242671086209855e-13,-4.5347634144108142e-14,-1.0320496013798011e-14,-2.359326365219513e-15,-5.41554173626215e-16,-1.24786017425398e-16,-2.8926981545065e-17,-7.050966818937e-18,-1.55542375303e-18},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},{{6.4693970953740129973574217161,-0.9216109919799212414066699587,-0.0192599622856911023092689753,-0.002555790026601034780976102,-0.0003962978861051813082740161,-0.0000680725859605390796787258,-0.0000125312766436908848939761,-2.4227714807242234452324e-6,-4.857013585407402691782e-7,-1.00110714050644942916e-7,-2.10906826647417753208e-8,-4.5223084605015386529e-9,-9.838386471080272505e-10,-2.166417168627708164e-10,-4.81954507222012562e-11,-1.08162087424281289e-11,-2.4458713335783525e-12,-5.567512674268389e-13,-1.274708347145548e-13,-2.9335464744126e-14,-6.7820981979923e-15,-1.5744199178785e-15,-3.668999035895e-16,-8.60230594515e-17,-2.12247481438e-17,-4.7287334463e-18},{-0.282134273590262084326147699063,-0.048779211715224874661970301263,-0.005132287910929449348270678875,-0.000648131893075145541454928007,-0.000095113499333915953778496845,-0.0000156355050722206804285429,-2.79090181838832149328872e-6,-5.28651928601176094007394e-7,-1.04583189016661771749662e-7,-2.1373375963420194505435e-8,-4.478331228506973629197e-9,-9.56903972995788055237e-10,-2.07706115183082199298e-10,-4.5668282622536320054e-11,-1.0148986684315401024e-11,-2.275862494581857389e-12,-5.14295823335435989e-13,-1.16994918744964574e-13,-2.6769366242202639e-14,-6.156341757528909e-15,-1.422227706999661e-15,-3.29890241297587e-16,-7.680805150999e-17,-1.7990615955989e-17,-4.43376762155e-18,-9.86830888528e-19},{-0.115547052374891864069853429439,-0.027454346827060181186715444938,-0.00370984918824216707601431029,-0.000559690417927877554962827431,-0.000092296424492658702401863441,-0.00001631362053424278181813564,-3.041354212539605261652888e-6,-5.90734095999118920468859e-7,-1.18457077693170831292001e-7,-2.4358902042470531086738e-8,-5.111246807453669774066e-9,-1.090326372613067996653e-9,-2.35788530018967816178e-10,-5.1579916864195895459e-11,-1.1394349358455740629e-11,-2.53839636591241765e-12,-5.69657721056740687e-13,-1.28665919069677818e-13,-2.9227089640800992e-14,-6.672835151462993e-15,-1.5304190845249e-15,-3.52447688598056e-16,-8.1480970441139e-17,-1.895142471682e-17,-4.635954234444e-18,-1.025661205149e-18}},{{5.75621777204673624205790067,-0.6320828726546473974452488277,-0.015703319017637724954195016,-0.0019909962782441368053805502,-0.0003023244447751850593862801,-0.0000513173473740909439636042,-9.3642658202949450996869e-6,-1.7963903710337667862701e-6,-3.574105110677497473599e-7,-7.31114166710291171678e-8,-1.52855107213373516986e-8,-3.2525565978079273209e-9,-7.022265926441136809e-10,-1.534692115140081505e-10,-3.38898209451867861e-11,-7.5508898484823378e-12,-1.6955321203221939e-12,-3.833387034943062e-13,-8.71943245830701e-14,-1.99405834161625e-14,-4.5823751786133e-15,-1.0576546287318e-15,-2.451211345155e-16,-5.71677408902e-17,-1.40291193459e-17,-3.1123439564e-18},{-0.333805741242794091017186945124,-0.04139263387419594757057834265,-0.003539594559868777089498902465,-0.000405555629949418658345737822,-0.000057367529865736566002985033,-9.29573242340335625875915e-6,-1.644019715957064652129225e-6,-3.08479158578329923631004e-7,-6.0380974655742634271971e-8,-1.2198742306655333479291e-8,-2.525825251953113160247e-9,-5.33366507085304801256e-10,-1.14453870140718421175e-10,-2.4891707903364173024e-11,-5.475245422809082651e-12,-1.216111001227515708e-12,-2.72395134787459082e-13,-6.1464239411269384e-14,-1.3959293132254522e-14,-3.188638488888059e-15,-7.32117652547652e-16,-1.68875172355672e-16,-3.9122374602236e-17,-9.12208209954e-18,-2.238399905458e-18,-4.96597430329e-19},{-0.156174780070977787055094639044,-0.022430872854951252269170654629,-0.002084792224872644313509285043,-0.000246798939543624606207874767,-0.000035208484029475318695338562,-5.711733992290461102331233e-6,-1.010222705569455249891687e-6,-1.89670692839033464406216e-7,-3.7178770136903304505291e-8,-7.527539278798733546165e-9,-1.562936530153834724641e-9,-3.31096358094923847894e-10,-7.129867844036627133e-11,-1.5563615961375195641e-11,-3.436425051507501936e-12,-7.66187824303848297e-13,-1.72269010264286836e-13,-3.9016188701467383e-14,-8.893147776488941e-15,-2.038509203367965e-15,-4.69615584413716e-16,-1.086716752457e-16,-2.5252118660457e-17,-5.90514829243e-18,-1.453308423526e-18,-3.2316702836e-19}},{{5.3982696671036842206113302334,-0.493499001267578732612176009,-0.0134608739440629935657660007,-0.0016631714146898796575885927,-0.0002501150788052230024001475,-0.000042264381934601706269823,-7.6918293884360166813046e-6,-1.472798982195325172144e-6,-2.925892104807881833522e-7,-5.97723134744009031036e-8,-1.24809479512510172886e-8,-2.6524465420123284967e-9,-5.719236599890472578e-10,-1.248234257995543867e-10,-2.75250260579769431e-11,-6.1236069203862619e-12,-1.372879639888954e-12,-3.098801772651451e-13,-7.03644317881292e-14,-1.60630553340511e-14,-3.6845158564185e-15,-8.488092988668e-16,-1.963372443894e-16,-4.56985635921e-17,-1.11898705268e-17,-2.4779260424e-18},{-0.362045216597553720348494237706,-0.036196364789045315827066818287,-0.002766546098163311254771192663,-0.00030590059237509183169215369,-0.000043042205245835066277860264,-6.987256425784452036465382e-6,-1.238840467060049923570686e-6,-2.32896931439768435404961e-7,-4.5641096611311987825029e-8,-9.225819512801956225775e-9,-1.910196680908954067689e-9,-4.03151109291560876032e-10,-8.642713501542436771e-11,-1.8770976102230285531e-11,-4.121979167263077611e-12,-9.13741725770855508e-13,-2.04218104370686821e-13,-4.597010838016492e-14,-1.0413655435206887e-14,-2.372324217613849e-15,-5.43166438535924e-16,-1.24929813491593e-16,-2.8856728992846e-17,-6.708216939537e-18,-1.640688695204e-18,-3.6301165681e-19},{-0.174787264013567179722871376716,-0.01930509775768932094940577047,-0.001549717774295113450740132618,-0.000173739011262607162500826504,-0.000024477874560505599072458848,-3.968874928324649006692706e-6,-7.02784352600440635745015e-7,-1.31989371941652582495331e-7,-2.5847659632042437027894e-8,-5.22231623716560578983e-9,-1.080979262770224310116e-9,-2.2812096185860549178e-10,-4.8907352837336923163e-11,-1.0624278651782113019e-11,-2.333796007060433028e-12,-5.17577806761838566e-13,-1.15740966245120089e-13,-2.6070534708302334e-14,-5.910110237719233e-15,-1.347462238194698e-15,-3.08783396926206e-16,-7.1086995769425e-17,-1.6435986081298e-17,-3.824750166742e-18,-9.36532556323e-19,-2.07413781922e-19}},{{5.1782261928881376636705109284,-0.4106189097372558226772067316,-0.0118879541126622335460026155,-0.0014445156525936593756319519,-0.0002159829242588304302844148,-0.0000364006476221444547132772,-6.6145487040002570216852e-6,-1.2652298157884161487413e-6,-2.511646607870315638189e-7,-5.12797141072076999191e-8,-1.07024429081724638157e-8,-2.2735266659973833292e-9,-4.900349126391412728e-10,-1.069130100582673957e-10,-2.35674272411933639e-11,-5.2413372417520174e-12,-1.1746673788017924e-12,-2.650444996845307e-13,-6.01608338178111e-14,-1.37282829864115e-14,-3.1476514960882e-15,-7.248105342173e-16,-1.675770588945e-16,-3.89851123495e-17,-9.5405940779e-18,-2.111683812e-18},{-0.380416262453494039619902376102,-0.032387768014671518779705292576,-0.002298234305505710712357564272,-0.000249728469203307062085618965,-0.000035131076625308698177983299,-5.716385125667998892621413e-6,-1.015659343507507989050623e-6,-1.91252869185686692206933e-7,-3.7527191899092006912339e-8,-7.593045382991990072181e-9,-1.573306690051914815162e-9,-3.32237688379070067897e-10,-7.1254193002800090436e-11,-1.5479902004638083351e-11,-3.399819136765480383e-12,-7.53692461157462028e-13,-1.68438415161795721e-13,-3.7910283927582249e-14,-8.585790498777928e-15,-1.955284720503048e-15,-4.47500238535324e-16,-1.02877023133224e-16,-2.3749840344864e-17,-5.517600861663e-18,-1.348484096003e-18,-2.98164745815e-19},{-0.18602707577830289053603690977,-0.017085203386592280822837981287,-0.001253773834333663953052468115,-0.000137210791310082259610168149,-0.000019297824724937602885953965,-3.136475236145595940269814e-6,-5.56733787244615431803659e-7,-1.04757647837385144966343e-7,-2.0543590080519261204631e-8,-4.154822653158222560405e-9,-8.60590179056295227742e-10,-1.81681212134193796343e-10,-3.8956115684984566169e-11,-8.461740894813211754e-12,-1.858203310882557073e-12,-4.11903287629136908e-13,-9.2049689968858565e-14,-2.0717326336343913e-14,-4.692097162537709e-15,-1.068613355711609e-15,-2.44591269854048e-16,-5.6236176388806e-17,-1.2984342007622e-17,-3.017059642261e-18,-7.37518350713e-19,-1.63105156153e-19}},{{5.0273207811257257325846071655,-0.3548313771334291203558746449,-0.0107120427463817019214540494,-0.0012864272597926630722585041,-0.0001916021436345691657790471,-0.0000322344212165264464630843,-5.8514131495151867920915e-6,-1.1184807744094057191035e-6,-2.219212599685561321333e-7,-4.52916799308513805819e-8,-9.4497899356393155272e-9,-2.0069162696702561081e-9,-4.324770340511658365e-10,-9.43377855748726775e-11,-2.07919565300019899e-11,-4.6233828289015065e-12,-1.0360288008500676e-12,-2.33731891865937e-13,-5.30465501575594e-14,-1.21033192755812e-14,-2.7747245609505e-15,-6.388545396977e-16,-1.476845041129e-16,-3.43525942892e-17,-8.4056055647e-18,-1.860224773e-18},{-0.393542474346508246601578621143,-0.02946558057968765648123677801,-0.0019809520790536922717120455,-0.00021318889949881244571446219,-0.00003002393901272308858592075,-4.89515172269129077370891e-6,-8.7112136206480016901219e-7,-1.6423030697133605291655e-7,-3.22542319326366835806e-8,-6.5308372781365934966e-9,-1.35399981793962232711e-9,-2.8606288109321769824e-10,-6.137554451685968026e-11,-1.333819598259572036e-11,-2.93025185297654079e-12,-6.4974529307581683e-13,-1.4523520752786461e-13,-3.269288742606419e-14,-7.40503265986852e-15,-1.68652618492386e-15,-3.8601078229918e-16,-8.874316682886e-17,-2.048685719435e-17,-4.75938476824e-18,-1.16310105504e-18,-2.57155608105e-19},{-0.193729032079952115164167668969,-0.015422305761029889690654635563,-0.001062974582895990979755044286,-0.000114860440731688824230005611,-0.000016165577847294395672794592,-2.633164419363018595734952e-6,-4.68246506757190072322355e-7,-8.8229585935963744242324e-8,-1.7320793478122563561602e-8,-3.505967674986711896666e-9,-7.26678998402693765256e-10,-1.53493684677398052225e-10,-3.2926441364213687806e-11,-7.154499444696702422e-12,-1.57155622434359709e-12,-3.4843314119039703e-13,-7.7876595922997166e-14,-1.7528858424039257e-14,-3.970073920926747e-15,-9.04152032592668e-16,-2.06932985543468e-16,-4.7572066012945e-17,-1.0982062751882e-17,-2.551264423739e-18,-6.23480839903e-19,-1.37849635988e-19}},{{4.9164421759057972140094859038,-0.3144008684446485038448392112,-0.0097932687887245764504904889,-0.0011658230243036926591993724,-0.0001731536495063294925090208,-0.0000290931903112484711285356,-5.2771878896129798424245e-6,-1.0082049673116121653499e-6,-1.999670225759896172825e-7,-4.07995036653292354197e-8,-8.5106143801746705359e-9,-1.8071249235550714637e-9,-3.893634770248848016e-10,-8.49221408590443491e-11,-1.87146322993019626e-11,-4.1610450176369226e-12,-9.323429565170007e-13,-2.103230490508874e-13,-4.7730284535997e-14,-1.08895938311731e-14,-2.4963139041566e-15,-5.747180039272e-16,-1.328502044015e-16,-3.09002328627e-17,-7.5603963763e-18,-1.6730896854e-18},{-0.403494795276385097437043007146,-0.02714066648174076906057406446,-0.00175032828763088787095403931,-0.00018730251427623681889899212,-0.0000264162421078032690023737,-4.31403754960614829825721e-6,-7.6864181184624556322506e-7,-1.4503947026300076624338e-7,-2.850459285726705706525e-8,-5.77470447459115223917e-9,-1.19775550564290782105e-9,-2.5314385906219741177e-10,-5.432916371927677218e-11,-1.180993091826434581e-11,-2.59509011133272632e-12,-5.755396525035222e-13,-1.2867022146595113e-13,-2.896839179763439e-14,-6.56228581761086e-15,-1.494759349994e-15,-3.4215374674028e-16,-7.866737047276e-17,-1.816215975381e-17,-4.21959202121e-18,-1.03124295207e-18,-2.28010975075e-19},{-0.199412504632823511160669658779,-0.014122703817602940340786728142,-0.000928651526672766087639281186,-0.000099615691484891158107968972,-0.000014038912301311631928209319,-2.290912712486691941925453e-6,-4.07946493131333479966648e-7,-7.6946011097033301711795e-8,-1.5117448445573235019459e-8,-3.06186562231844484669e-9,-6.34947202818387686084e-10,-1.34172899793190841331e-10,-2.879185606956597533e-11,-6.257948608316421567e-12,-1.374966301393343498e-12,-3.04912712353006483e-13,-6.8162221278440106e-14,-1.5344738520155022e-14,-3.475866868004387e-15,-7.91690069143526e-16,-1.81210708374919e-16,-4.1661824670375e-17,-9.618215135586e-18,-2.234512562336e-18,-5.46084515065e-19,-1.20738267797e-19}},{{4.8310047589317972498519152462,-0.2835775070168709448348882064,-0.0090517123149981442570439297,-0.0010702152250806124034043463,-0.0001586151311227120918578788,-0.0000266241544865502711493253,-4.8265107876673450571699e-6,-9.217403861212295527965e-7,-1.827653486641891247425e-7,-3.72816582421999305336e-8,-7.7754530180048136941e-9,-1.6507878462872031999e-9,-3.556368762248008822e-10,-7.7558380133901821e-11,-1.70903646514414641e-11,-3.7996137617905449e-12,-8.51301869413797e-13,-1.920298578675534e-13,-4.35765161714139e-14,-9.941431217763e-15,-2.2788564787047e-15,-5.246318296546e-16,-1.212677630108e-16,-2.82051954618e-17,-6.9007440014e-18,-1.5270665569e-18},{-0.411357915649903363881222145925,-0.02523818941865610271783014303,-0.00157433038727145913000857363,-0.00016788673956147850669364412,-0.00002371255314828061286765182,-3.87776736362681885930642e-6,-6.9157745236146735404278e-7,-1.3058902541566611670845e-7,-2.567823842217482974647e-8,-5.20428197568090873152e-9,-1.07980526294334010671e-9,-2.2827889563629883428e-10,-4.90042316230973709e-11,-1.065455918075284057e-11,-2.34162067250673493e-12,-5.1940479041455252e-13,-1.1613620635926011e-13,-2.614968176743304e-14,-5.92439658863509e-15,-1.34959202981018e-15,-3.0895175007154e-16,-7.103927106739e-17,-1.640221986589e-17,-3.8109579636e-18,-9.3143452725e-19,-2.05952924911e-19},{-0.203817975061891608803814230153,-0.013073625710872181764308728799,-0.000828421866875569568249999825,-0.000088474159340438346772660842,-0.000012487136327723163184634354,-2.040733533120669284788926e-6,-3.63788341906367778651803e-7,-6.8670887861920658669183e-8,-1.3499693151949818395912e-8,-2.735490682916487598935e-9,-5.67481517342213919802e-10,-1.19954246055991012263e-10,-2.5747545993448061337e-11,-5.59753677147062939e-12,-1.2301062195251314e-12,-2.72835560080493598e-13,-6.1000751054089909e-14,-1.3734392582525487e-14,-3.111466958479594e-15,-7.08767433454789e-16,-1.62245985625641e-16,-3.7304866457362e-17,-8.613006275447e-18,-2.001117709416e-18,-4.89077268435e-19,-1.08139154801e-19}},{{4.7628376632011248518837855453,-0.2591946183149977892390324757,-0.0084381004788033190321977182,-0.0009922080078167180990630321,-0.000146806372317843162509041,-0.0000246227030544550685148721,-4.4616011749532184443121e-6,-8.517839104689904134662e-7,-1.688554942230069627551e-7,-3.44381939775030758326e-8,-7.1814198279712467824e-9,-1.5244964989187449771e-9,-3.28398129225620444e-10,-7.16122867405285235e-11,-1.57790155071282914e-11,-3.5078550372982237e-12,-7.858913406740933e-13,-1.772666228506778e-13,-4.02246379921768e-14,-9.1763880567129e-15,-2.1034131099354e-15,-4.842261966267e-16,-1.119247580492e-16,-2.60314242316e-17,-6.3687292113e-18,-1.4093068533e-18},{-0.417761742173515342167290441503,-0.02364653115868649054984900439,-0.00143512523861111200195462914,-0.00015271623631158394952583395,-0.00002159967470746536714800414,-3.53626785732077967828645e-6,-6.3116939711978649162652e-7,-1.1924966904170099860889e-7,-2.345852592808773897109e-8,-4.75599415854908412943e-9,-9.8705893240464624891e-10,-2.087182069911660903e-10,-4.481360723935208414e-11,-9.74499982155799605e-12,-2.14202073108329242e-12,-4.7518902685275479e-13,-1.0626134636836478e-13,-2.392854243196749e-14,-5.4216546845997e-15,-1.23516363026805e-15,-2.8277680570531e-16,-6.502495176152e-17,-1.501447739035e-17,-3.48871813053e-18,-8.5272280657e-19,-1.88556634606e-19},{-0.20735521066961259215520583613,-0.012205265459509182717032545848,-0.000750458038013529318427308675,-0.000079931418651794055368472751,-0.000011297569648909258697686219,-1.848619923721776703918099e-6,-3.29827559373734884318634e-7,-6.2299207352432748578883e-8,-1.2252902108804234366873e-8,-2.483769778927633819888e-9,-5.15416145513471010753e-10,-1.08975709285949153489e-10,-2.3395959016447220599e-11,-5.087212455780128578e-12,-1.118132192822498413e-12,-2.48033778046000234e-13,-5.5462258080847984e-14,-1.2488741147688103e-14,-2.829544037624626e-15,-6.4460398387094e-16,-1.47569844955195e-16,-3.393286185378e-17,-7.83498891166e-18,-1.820466335439e-18,-4.44952230115e-19,-9.8387216058e-20}},{{4.7069827319138713699381417205,-0.2393565666437376165239651656,-0.0079202557580301355164590645,-0.0009271178874001477051297246,-0.0001369880064684799286227958,-0.0000229612363918233462034845,-4.15895621859220713272e-6,-7.937995527821783600885e-7,-1.573312066314510028733e-7,-3.20831797940885598074e-8,-6.6895602076821170004e-9,-1.41994983242182729e-9,-3.058533726093089383e-10,-6.66916210940403271e-11,-1.46939564418760586e-11,-3.2664706963587936e-12,-7.317799171513491e-13,-1.650547110147958e-13,-3.7452243048645e-14,-8.5436546129851e-15,-1.9583214428475e-15,-4.508128250017e-16,-1.041990260509e-16,-2.42340321332e-17,-5.9288554358e-18,-1.3119466374e-18},{-0.423100112685723753343437538158,-0.02229099281104365757956706891,-0.00132195711915199945997597132,-0.00014049234653473878393970351,-0.0000198960706993700930034723,-3.26050661310390461022251e-6,-5.8233194263238582018313e-7,-1.1007406818468961986175e-7,-2.166113521467570842513e-8,-4.39279802571583318794e-9,-9.1188352320392056102e-10,-1.9285738587276226155e-10,-4.141455278871375163e-11,-9.0070461653103718e-12,-1.98004058497035764e-12,-4.3929936391323763e-13,-9.824449641186559e-14,-2.212502438959703e-14,-5.01337832224741e-15,-1.14222398637784e-15,-2.6151474703497e-16,-6.013895324846e-17,-1.388697241182e-17,-3.22688318005e-18,-7.8876036858e-19,-1.74419217395e-19},{-0.210271639784283023658033196928,-0.01147204496212764898769506468,-0.000687889537144507659847347573,-0.000073146021958790398120333657,-0.000010352270089378000002872144,-1.695714788212181075532716e-6,-3.02763143796847406846113e-7,-5.7216466597375950872561e-8,-1.1257575625938814468787e-8,-2.282697267959245663381e-9,-4.73806286761302561315e-10,-1.0019822189903314642e-10,-2.1515177754525346098e-11,-4.678936455855208012e-12,-1.028525945202400108e-12,-2.28181816363584189e-13,-5.1028212663586719e-14,-1.1491309322213682e-14,-2.603763382089252e-15,-5.93210719368028e-16,-1.35813135247186e-16,-3.1231322029405e-17,-7.21160425878e-18,-1.675707007415e-18,-4.09591130435e-19,-9.0571711551e-20}},{{4.6602444931977423502760060222,-0.222855222516470795955069694,-0.0074761930301386000573726466,-0.00087182195278802465890661,-0.0001286710863760619571481565,-0.0000215556662842809910960548,-3.903117147488306077516e-6,-7.448074950798279103878e-7,-1.475976653686777139793e-7,-3.00946602920933909274e-8,-6.2743364995824715744e-9,-1.331708208325049015e-9,-2.86827518550094981e-10,-6.25395230151423413e-11,-1.37784742364111542e-11,-3.0628300486352603e-12,-6.861333533782209e-13,-1.54753920190663e-13,-3.51138736247796e-14,-8.0100092561672e-15,-1.8359582771849e-15,-4.226350408687e-16,-9.76841495381e-17,-2.27184108027e-17,-5.5579550564e-18,-1.2298555143e-18},{-0.427633369794037109122300875704,-0.02111957855204548537566091161,-0.00122793518848197900713185834,-0.00013040355258446523620067151,-0.00001848876963446964398540282,-3.03240215416429044381858e-6,-5.4189336171566328039164e-7,-1.0247069847000434954652e-7,-2.017086019291229521518e-8,-4.09152151717609072649e-9,-8.4950100400956703588e-10,-1.7969154639557196544e-10,-3.859229870538058337e-11,-8.39417937061047762e-12,-1.84549017622540216e-12,-4.0948205123733083e-13,-9.158302377104885e-14,-2.062621206436404e-14,-4.67403815470624e-15,-1.0649681723949e-15,-2.4383890234819e-16,-5.607668112252e-17,-1.294947178824e-17,-3.00915464475e-18,-7.355683002e-19,-1.62661632001e-19},{-0.21272655219664084073949181669,-0.010842864554043626984126250307,-0.00063643844746263841210583191,-0.000067608619057714008342288794,-9.580213180363005210618495e-6,-1.570655347632938463588169e-6,-2.80603192625935154247008e-7,-5.3051383129742566317017e-8,-1.0441434935678337697243e-8,-2.117740702451636754006e-9,-4.39656304024920706153e-10,-9.2991923380230653991e-11,-1.9970614616025452543e-11,-4.343562689640667092e-12,-9.54903968281261994e-13,-2.11868010947963945e-13,-4.7383818419524866e-14,-1.0671384529555623e-14,-2.418138365082818e-15,-5.50952685650847e-16,-1.26145121971262e-16,-2.9009512486997e-17,-6.698870739283e-18,-1.556632418177e-18,-3.80501757716e-19,-8.4141976857e-20}},{{4.620464137598567439532998868,-0.2088820438600549584548271851,-0.0070903330054413654813922123,-0.0008241506698342061378296714,-0.0001215181505039895439318095,-0.0000203481177099625283606446,-3.6834605784513432728055e-6,-7.02762031389590216777e-7,-1.392468047310398646528e-7,-2.8389015168650119105e-8,-5.9182458777773356051e-9,-1.256044722715154554e-9,-2.705156992474876828e-10,-5.89801006801731956e-11,-1.2993741175487581e-11,-2.8882873191334987e-12,-6.470119178638785e-13,-1.459261693175041e-13,-3.31100095009942e-14,-7.5527253500186e-15,-1.7311093592123e-15,-3.984914482661e-16,-9.21022174951e-17,-2.1419876156e-17,-5.2401910608e-18,-1.1595269619e-18},{-0.431541281959921935806536881897,-0.02009489047617394158088003862,-0.00114843090864307104480578553,-0.00012191524704570817575734996,-0.00001730353549461124719884912,-2.84006084049211684299212e-6,-5.0776472782684225646555e-7,-9.604956432692714198706e-8,-1.891167966914735944402e-8,-3.83686333980030279165e-9,-7.9675444312262235035e-10,-1.6855641643594668607e-10,-3.620481372234487035e-11,-7.87562451794478154e-12,-1.73162607769561577e-12,-3.8424522899837229e-13,-8.594413444091058e-14,-1.935732865653643e-14,-4.38672480409196e-15,-9.9955074371847e-16,-2.2887032508278e-16,-5.263631873574e-17,-1.215543837496e-17,-2.82473280129e-18,-6.9051014079e-19,-1.52701443966e-19},{-0.21482767152086932318203237816,-0.010295718754551405172962590881,-0.000593295152039042876781879608,-0.000062991874787376859756601613,-8.935899193748650900995406e-6,-1.466155593075382553456664e-6,-2.62068788304564959045506e-7,-4.9565303709214182762423e-8,-9.757978009395019635521e-9,-1.979543302219905482592e-9,-4.11036237553738650447e-10,-8.6950808178662188416e-11,-1.8675476155069357392e-11,-4.062288109328203244e-12,-8.93146713274697093e-13,-1.98181101408569778e-13,-4.4325819872585333e-14,-9.983300873753972e-15,-2.262343354230968e-15,-5.15481846521312e-16,-1.18029150297537e-16,-2.7144216680211e-17,-6.268376701741e-18,-1.456649317234e-18,-3.5607456199e-19,-7.8742437067e-20}},{{4.586126859301294711628647555,-0.1968742515704991228544502148,-0.0067512987509788632072625321,-0.0007825456127373331121748468,-0.0001152880814080875823530349,-0.0000192973325879465217902415,-3.4924228984460124863454e-6,-6.662078724744345932374e-7,-1.31988503897194049724e-7,-2.69068198525338141809e-8,-5.6088543822778542121e-9,-1.1903125495700628803e-9,-2.563464377499001959e-10,-5.58884914215232024e-11,-1.23121988736240611e-11,-2.7367070016964774e-12,-6.130392564434425e-13,-1.38260651075424e-13,-3.13700512633253e-14,-7.1556822737205e-15,-1.6400764194854e-15,-3.77530017895e-16,-8.72561521898e-17,-2.02925619521e-17,-4.9643347308e-18,-1.0984749793e-18},{-0.434952419478394254154639426865,-0.01918927972459344698002675168,-0.00108021454438732935658977892,-0.00011466011279481606650255127,-0.00001628943758983237759119779,-2.6753140918375003095138e-6,-4.785097173208564832592e-7,-9.054225900479678344144e-8,-1.783123457585768446951e-8,-3.61827845824526723372e-9,-7.5146712069992503131e-10,-1.5899378393061800687e-10,-3.415408996016074207e-11,-7.43013892199052217e-12,-1.63379230452397552e-12,-3.6255859833210756e-13,-8.1097947387114e-14,-1.826671150600388e-14,-4.13975375262823e-15,-9.4331406777911e-16,-2.1600148395029e-16,-4.967835331138e-17,-1.147269848507e-17,-2.66615055935e-18,-6.5176286471e-19,-1.44135898663e-19},{-0.216650754259694687675264505798,-0.009814558941118595581821547182,-0.000556534464940821041838804397,-0.000059075310038798526250702912,-8.388739669581773610092377e-6,-1.37731213957162203391765e-6,-2.46298205581864828702152e-7,-4.6597266668719848162239e-8,-9.175816265521093686011e-9,-1.861785215416585087632e-9,-3.86641835607238177229e-10,-8.1800376970992563927e-11,-1.7571058888247784428e-11,-3.822390869975583132e-12,-8.404661475487789e-13,-1.86504206528585724e-13,-4.1716591532918175e-14,-9.396131589559756e-15,-2.129384206370396e-15,-4.85207502292313e-16,-1.1110161120067e-16,-2.5551940523313e-17,-5.900868140047e-18,-1.371289296885e-18,-3.35218663155e-19,-7.4132094365e-20}},{{4.5561360788319686041928354347,-0.1864272975292021055301433446,-0.0064505699750264329161144928,-0.0007458558198405061377336708,-0.0001098035914020079585903276,-0.0000183730324065308118041884,-3.3244592494962080883927e-6,-6.340788993210751901583e-7,-1.256103322254939337975e-7,-2.56045793551071259743e-8,-5.337063873380338609e-9,-1.1325754317489811955e-9,-2.439017605516214508e-10,-5.31733873013258422e-11,-1.17136977492927618e-11,-2.6036036336482351e-12,-5.832092757620604e-13,-1.315301928617433e-13,-2.9842401621192e-14,-6.8070992937113e-15,-1.560157012477e-15,-3.591281867804e-16,-8.30019614976e-17,-1.93029586707e-17,-4.7221829555e-18,-1.0448835131e-18},{-0.437961435920679185644326529009,-0.01838182175119342652002232709,-0.00102096054246447736411780339,-0.00010837695975052072129143105,-0.00001541028918277743364225388,-2.53235177397303855183776e-6,-4.5310564153192356413334e-7,-8.575751409951402869822e-8,-1.68921890959393942294e-8,-3.42824352369580732604e-9,-7.1208538036002851959e-10,-1.5067647543762650516e-10,-3.237012575856020791e-11,-7.04254599047747041e-12,-1.54866165445057089e-12,-3.4368574591682485e-13,-7.688012296274309e-14,-1.731742133917114e-14,-3.92476920938856e-15,-8.943573698531e-16,-2.0479781089693e-16,-4.710298000003e-17,-1.08782333255e-17,-2.52806506458e-18,-6.1802194385e-19,-1.36676771987e-19},{-0.218250800964839036363828763265,-0.009387378862234164474845845645,-0.000524790856097565582852810136,-0.000055704683735042040202742712,-7.917358857612185053942385e-6,-1.300694696985900987456998e-6,-2.32687946077212668695292e-7,-4.4034446111760307880028e-8,-8.672932642967120895163e-9,-1.760030983384593809848e-9,-3.65557310797690843938e-10,-7.7347813294653667389e-11,-1.6616113989870457501e-11,-3.614929207912465165e-12,-7.94902160381126855e-13,-1.76403534770632295e-13,-3.9459335836185607e-14,-8.888120782553263e-15,-2.014340005322262e-15,-4.59010313766582e-16,-1.05106608373225e-16,-2.4173916783701e-17,-5.582791740085e-18,-1.297406656842e-18,-3.17166029712e-19,-7.01412644e-20}},{{4.5296769774579271640016601993,-0.1772424332197553847966298013,-0.0061816273694313580706080644,-0.0007132111060467379426501594,-0.0001049311435905389093463071,-0.0000175524475297984443220275,-3.1754036776472560235175e-6,-6.055745997517283778848e-7,-1.199528517674907615509e-7,-2.44496595878236713832e-8,-5.096049821551593182e-9,-1.0813812840530678574e-9,-2.328682653501639773e-10,-5.07663344564708739e-11,-1.11831335068619536e-11,-2.4856152447899243e-12,-5.567679558905642e-13,-1.255645540721325e-13,-2.84883971190036e-14,-6.4981493524539e-15,-1.4893263798367e-15,-3.428195377608e-16,-7.92317738609e-17,-1.84259639296e-17,-4.5075909904e-18,-9.973922559e-19},{-0.440639727373959924840628129104,-0.01765636012794319118962578283,-0.00096894972045333887960807347,-0.00010287468080162021687357923,-0.00001463962217841855558984346,-2.40691969112754129216221e-6,-4.3080297718444306566249e-7,-8.155504659851971378808e-8,-1.606714395388377058163e-8,-3.26123487538094087699e-9,-6.7746808637084697049e-10,-1.4336410999088567535e-10,-3.080147536967987502e-11,-6.7016894702667412e-12,-1.47378780723391837e-12,-3.2708512591184576e-13,-7.316978888715337e-14,-1.648228524249278e-14,-3.73562375658364e-15,-8.512820683059e-16,-1.9493951776088e-16,-4.483674747986e-17,-1.035510067436e-17,-2.40654355111e-18,-5.8832706113e-19,-1.30111876555e-19},{-0.219668814716911125937261700495,-0.009004999266662001852411118937,-0.000497067538421573470769457482,-0.000052768668871097568263943456,-7.506336517144107093218728e-6,-1.233826039896420324081513e-6,-2.20801755311982330278706e-7,-4.1795219526782589726653e-8,-8.233389411852170057034e-9,-1.67106827727973481369e-9,-3.47119145126810392529e-10,-7.3453367597210611294e-11,-1.5780736419495220632e-11,-3.433418789272130428e-12,-7.55033017460093664e-13,-1.67564381315395598e-13,-3.7483816426723181e-14,-8.443479914135643e-15,-1.913639077190602e-15,-4.3607774479171e-16,-9.9858363429964e-17,-2.2967476487029e-17,-5.304305932207e-18,-1.232716982985e-18,-3.01358867056e-19,-6.6646703081e-20}},{{4.5061305176690278511037227284,-0.1690939361633040283050793083,-0.0059393896503458627082757309,-0.0006839402450198875894345993,-0.0001005680637304733601922049,-0.000016818095140472170432251,-3.0420600494942337670489e-6,-5.800811402642074908769e-7,-1.14893851399669642709e-7,-2.34170532740575809557e-8,-4.8805838159701680239e-9,-1.0356178382337392523e-9,-2.230059287840022157e-10,-4.86149133314503778e-11,-1.07089410896412012e-11,-2.3801678898411192e-12,-5.331380633906499e-13,-1.202334182830304e-13,-2.72784434991023e-14,-6.2220763445458e-15,-1.4260348852377e-15,-3.282471192284e-16,-7.58630383812e-17,-1.76423684422e-17,-4.3158568455e-18,-9.549603272e-19},{-0.443042271189121427716218354213,-0.01700020125733050096261345693,-0.00092288235548843195406749454,-0.00009801002992543113270775138,-0.0000139575976959299947722492,-2.29582592806893322809722e-6,-4.1103887395974363337825e-7,-7.782944195120886828801e-8,-1.53354995282486874647e-8,-3.1130979317407474766e-9,-6.4675664941845207853e-10,-1.3687577361817319986e-10,-2.940941062206783473e-11,-6.39916921322262978e-12,-1.40732845812968514e-12,-3.1234883839995404e-13,-6.987588935683102e-14,-1.574083060652671e-14,-3.56768526828599e-15,-8.1303420968766e-16,-1.8618559293472e-16,-4.282429350861e-17,-9.89052968008e-18,-2.2986213201e-18,-5.6195423094e-19,-1.24281234464e-19},{-0.220936055192710125132901620749,-0.008660269709197917077750373808,-0.000472619021362180985478267297,-0.000050184778836102555353738149,-7.14424899142370152725067e-6,-1.174869087058767477672949e-6,-2.10315821034123235868168e-7,-3.9818959975063018769448e-8,-7.845342013689205037362e-9,-1.592508722248797632279e-9,-3.30833833762431738026e-10,-7.0013067003560941321e-11,-1.5042671280768256695e-11,-3.273032996163541189e-12,-7.19800244716309546e-13,-1.59752409885591453e-13,-3.573772576329724e-14,-8.050449083540588e-15,-1.824620775693413e-15,-4.15804445130196e-16,-9.5218454847411e-17,-2.1900824449188e-17,-5.058076546734e-18,-1.175517739575e-18,-2.87381427095e-19,-6.3556545762e-20}},{{4.4850173371511252862622316542,-0.1618078732592364510451622001,-0.0057198321540566984120385879,-0.0006575164175916710604108787,-0.0000966339917177479573320627,-0.0000161563085841222885591363,-2.9219316265198176276133e-6,-5.571192308378268443592e-7,-1.103379385791368956183e-7,-2.24872457690206471966e-8,-4.6865866325200881837e-9,-9.944174390630781299e-10,-2.141275392403542568e-10,-4.66782427695403435e-11,-1.02821019807233488e-11,-2.2852545580708303e-12,-5.118695393716579e-13,-1.154351867943903e-13,-2.61894695381039e-14,-5.9736135973548e-15,-1.3690745796589e-15,-3.151326979917e-16,-7.28314116929e-17,-1.69372003107e-17,-4.1433158823e-18,-9.167765089e-19},{-0.445212162213046739106342196396,-0.01640321938849914161157499065,-0.00088175668220947544838828553,-0.00009367339214880353729094287,-0.00001334903171610797198184003,-2.19662514811633567455815e-6,-3.9338179089574883998068e-7,-7.449982973108634355832e-8,-1.468144514027449575982e-8,-2.98064277030128769032e-9,-6.1929163645797021227e-10,-1.3107248127717229445e-10,-2.816417259754715177e-11,-6.12852929607297144e-12,-1.34786747478467798e-12,-2.9916329599651109e-13,-6.692841304718337e-14,-1.507731456496356e-14,-3.41739139892455e-15,-7.7880317261342e-16,-1.7835065254356e-16,-4.102303060658e-17,-9.47469563336e-18,-2.20201745249e-18,-5.383464106e-19,-1.19061750447e-19},{-0.222076815133075679855957602388,-0.008347528713297002236782329456,-0.000450876049126670233423031133,-0.000047890517289108574624410638,-6.822440255066454779118821e-6,-1.122430457109442395448669e-6,-2.00984337852137905802616e-7,-3.8059616744873216746256e-8,-7.499789218001058168534e-9,-1.52253657090875815585e-9,-3.1632607225672898973e-10,-6.6947820163355247464e-11,-1.4384985578735898467e-11,-3.13009869435804154e-12,-6.88398187466441776e-13,-1.52789228937502325e-13,-3.4181238660388738e-14,-7.700073615126679e-15,-1.745258907310697e-15,-3.97729369370574e-16,-9.1081448778814e-17,-2.094974068135e-17,-4.838516349463e-18,-1.124511827324e-18,-2.74916944251e-19,-6.0800795274e-20}},{{4.4659600186894017450823241373,-0.1552479091706683803650871964,-0.005519721776071584303702039,-0.0006335198257181360730446477,-0.0000930650491433126170165132,-0.0000155562358582036246715452,-2.8130370978008508950655e-6,-5.363086820054662601372e-7,-1.062094669468992980749e-7,-2.16447654130437491583e-8,-4.5108247409703153638e-9,-9.570924102933730848e-10,-2.060847364477567539e-10,-4.49239288163311029e-11,-9.8954705908095368e-12,-2.1992851669620279e-12,-4.926058421435732e-13,-1.110893767491315e-13,-2.52032000070074e-14,-5.748589461707e-15,-1.3174887069751e-15,-3.032559081271e-16,-7.00859344829e-17,-1.6298602337e-17,-3.9870659372e-18,-8.821983836e-19},{-0.44718370893784879925421324673,-0.01585722814970730673139646322,-0.00084478749411357428171578785,-0.00008977937144288586734908848,-0.00001280209105860998847851148,-2.10740992119694171243606e-6,-3.7749487104561835433984e-7,-7.150304884985082803012e-8,-1.409262719575565889242e-8,-2.86137612377872163064e-9,-5.9455748801929680662e-10,-1.2584554641084684212e-10,-2.70424852265377825e-11,-5.88471934032069482e-12,-1.294296874023301e-12,-2.8728311590448066e-13,-6.427256964022591e-14,-1.447941560167932e-14,-3.28195373720708e-15,-7.4795439289342e-16,-1.7128956465758e-16,-3.939961584944e-17,-9.09990599394e-18,-2.11494592815e-18,-5.1706739452e-19,-1.14357027308e-19},{-0.223110289216033939815498600593,-0.008062229197053828199674601481,-0.000431396042126377713785747962,-0.000045837614700304403561788657,-6.534222925759895984447142e-6,-1.075432659158446806310588e-6,-1.9261708342537225159501e-7,-3.6481532110139398976211e-8,-7.189758166583677917715e-9,-1.459744747812160641642e-9,-3.0330494238166859541e-10,-6.4196304426683148068e-11,-1.3794547801712703029e-11,-3.001766870257118511e-12,-6.60201854837199299e-13,-1.4653643145645281e-13,-3.2783452574762888e-14,-7.385404331296375e-15,-1.673980912792452e-15,-3.81494661410527e-16,-8.7365502508269e-17,-2.0095423087887e-17,-4.641287645589e-18,-1.078692168266e-18,-2.63719463496e-19,-5.8325099195e-20}},{{4.4486570470841160411943555257,-0.1493055660677436818625955388,-0.005336428609826324679180212,-0.0006116114490208480664149198,-0.0000898097610500147326021624,-0.0000150091404691418303988506,-2.713782272059752570459e-6,-5.173436999542535204746e-7,-1.024476087330258898978e-7,-2.08771738353709087807e-8,-4.3506989721216340932e-9,-9.230900569064860844e-10,-1.987582939952323517e-10,-4.33259411656614635e-11,-9.5433054993495376e-12,-2.1209821857459034e-12,-4.750605301799483e-13,-1.071313325807485e-13,-2.43049542765704e-14,-5.5436529642378e-15,-1.2705087606847e-15,-2.924397487955e-16,-6.75856768845e-17,-1.57170512589e-17,-3.8447761693e-18,-8.507100128e-19},{-0.44898459977321610476660168382,-0.01535552984023732143140176148,-0.00081135016701311570297143768,-0.00008626038889571005966450943,-0.00001230740729912645621296928,-2.02666882197086347776128e-6,-3.631110154692174125321e-7,-6.878899302215157992403e-8,-1.355924171343699139713e-8,-2.75331869724947413301e-9,-5.7214482213460117115e-10,-1.2110864808646845682e-10,-2.602585894508245962e-11,-5.66372691973273512e-12,-1.24573626815732624e-12,-2.7651329762531189e-13,-6.186481390358458e-14,-1.393734013772894e-14,-3.15915576831425e-15,-7.199834154238e-16,-1.6488694671327e-16,-3.792753930358e-17,-8.76004429965e-18,-2.03598657592e-18,-4.9777031066e-19,-1.10090406469e-19},{-0.224051865941091636860427410168,-0.007800672656451362725984070135,-0.000413829445690781097209536453,-0.000043988160605080131264828396,-6.274342707357440293791201e-6,-1.033028357434056537881239e-6,-1.85064364049200077394067e-7,-3.5056630992630841226307e-8,-6.90975658531223084214e-9,-1.403024595481233657426e-9,-2.91541165143984589064e-10,-6.1710179546568586896e-11,-1.3261004548153235854e-11,-2.885790890856639589e-12,-6.34718332178908854e-13,-1.4088484590552685e-13,-3.1519989795549793e-14,-7.100958783440554e-15,-1.609546035842048e-15,-3.66817948570686e-16,-8.4006031875377e-17,-1.9323033464558e-17,-4.46296700948e-18,-1.037263903526e-18,-2.5359484827e-19,-5.608655426e-20}},{{4.4328644178022398832435813954,-0.1438933805466138363365678305,-0.0051677894264381769153279841,-0.0005915142304555558454108545,-0.0000868261421512589517650013,-0.000014507902638312104777597,-2.6228686123865522637458e-6,-4.999752848153991650943e-7,-9.90028452850570984275e-8,-2.01743470472321665908e-8,-4.2040940821612345044e-9,-8.919606396876631573e-10,-1.920512043394816919e-10,-4.18631022255778677e-11,-9.220935963841242e-12,-2.0493063856258859e-12,-4.59000605459456e-13,-1.035084645202324e-13,-2.34827918802006e-14,-5.3560787018917e-15,-1.2275097235156e-15,-2.82540271922e-16,-6.52973531673e-17,-1.51848026075e-17,-3.7145511531e-18,-8.218918468e-19},{-0.450637451990832612303986392275,-0.01489258609809482357705104856,-0.00078094127266134171500334948,-0.00008306222090883252442946498,-0.00001185745960970399882438265,-1.95318751832673468644545e-6,-3.500154957741082631501e-7,-6.631736200305176124649e-8,-1.307340063126798489671e-8,-2.65487755080977259191e-9,-5.5172409257669748867e-10,-1.1679228609724316557e-10,-2.509940471938083091e-11,-5.46232051259264602e-12,-1.20147652624783385e-12,-2.6669675627187402e-13,-5.967006415835826e-14,-1.344319742862335e-14,-3.04721149666447e-15,-6.9448373849344e-16,-1.5904981486192e-16,-3.658543605423e-17,-8.4501809012e-18,-1.96399473175e-18,-4.8017556483e-19,-1.06200095348e-19},{-0.224914041621864810230732318125,-0.00755981673549348309771614832,-0.000397896314262238195659073958,-0.000042311939481979759959209141,-6.038609788860411768470327e-6,-9.94541317654629407421722e-7,-1.78206637339330932728281e-7,-3.3762482543465182517713e-8,-6.655394764278206412402e-9,-1.351489759687615367986e-9,-2.80851392062532623977e-10,-5.9450781420841826252e-11,-1.2776073438276426284e-11,-2.780373263552568124e-12,-6.11553197115506743e-13,-1.35747105256470991e-13,-3.0371339570329927e-14,-6.84234869335476e-15,-1.550961067131105e-15,-3.53473178782668e-16,-8.0951330923562e-17,-1.8620691652117e-17,-4.300813333419e-18,-9.99590563219e-19,-2.44387638901e-19,-5.4050803249e-20}},{{4.4183823846610449217551287696,-0.1389399958668829559597135795,-0.0050120069764055125666327165,-0.0005729993335391904041318424,-0.0000840795749185290196959142,-0.0000140466565313500302052223,-2.5392267622453756121116e-6,-4.83998444412473929203e-7,-9.58344185842578394248e-8,-1.95279535258572921094e-8,-4.0692695607873173146e-9,-8.633341337973450762e-10,-1.858836611576705916e-10,-4.0517990883555414e-11,-8.9245199848896558e-12,-1.9834029790211169e-12,-4.442344316513019e-13,-1.001775204019502e-13,-2.27268928319014e-14,-5.1836253489873e-15,-1.1879776082277e-15,-2.734391042765e-16,-6.31935921144e-17,-1.46954881798e-17,-3.5948323363e-18,-7.953989094e-19},{-0.452160940722661067570064033687,-0.01446377319715345631260993824,-0.00075315029866308473698579119,-0.00008014082203407963157881564,-0.00001144613546638755328067673,-1.88597831280845217628543e-6,-3.380335596682476149098e-7,-6.405534447229860126816e-8,-1.262867964094130183004e-8,-2.56475500774880311476e-9,-5.3302678098867802173e-10,-1.1283982097795033214e-10,-2.425098688612819356e-11,-5.2778658559210689e-12,-1.16093951703546462e-12,-2.577054132167175e-13,-5.765971420483126e-14,-1.299055272855274e-14,-2.94466437657262e-15,-6.7112382387332e-16,-1.5370232814001e-16,-3.535587918748e-17,-8.16629459065e-18,-1.8980366291e-18,-4.6405506606e-19,-1.02635682237e-19},{-0.225707080360386196045026888188,-0.007337133430293568270887169796,-0.000383369663214541850270824794,-0.000040784553276671412903022626,-5.823639213640435439851767e-6,-9.59424779607621128249471e-7,-1.71947191894121560500565e-7,-3.2580931970431656045799e-8,-6.423118624083719799211e-9,-1.304422422522344459854e-9,-2.71087105305075637486e-10,-5.7386785284703628023e-11,-1.233304324840876913e-11,-2.684057285082739776e-12,-5.9038676957697438e-13,-1.31052391184369716e-13,-2.9321685366877207e-14,-6.60601638427888e-15,-1.497420730646668e-15,-3.41277061035946e-16,-7.8159472106648e-17,-1.7978763683934e-17,-4.152603660242e-18,-9.65155964627e-19,-2.3597175026e-19,-5.2189979819e-20}} };
const double CHEB_ZEROTH_POLY[RADIAL_BASIS_CHEB_SIZE] = { 6.59847513031748382511625676719,-4.4869251131191746725290662587,0.7496714847137209670682941396,-0.0844151398026490633109318719,0.0068870633758359991462957016,-0.0004783856737519424471746534,0.0000220933258304189139717034,-1.8454904855839843868591e-6,-8.55443393683511954091e-8,-2.98268695174062491305e-8,-5.8923659491332722808e-9,-1.2802527652642414182e-9,-2.777453642358750697e-10,-6.09406379087692623e-11,-1.34808585874582187e-11,-3.004282842016014e-12,-6.739089630156531e-13,-1.520525930680874e-13,-3.4487286159652e-14,-7.8591405857736e-15,-1.7986686386677e-15,-4.13261515722e-16,-9.53039079889e-17,-2.21086369711e-17,-5.392575066e-18,-1.19016224866e-18 };

void RadialBasis_Shapeev2::InitShapeev2RB() {
	size_ = rb_size;
	mindist_ = min_dist;
	maxdist_ = max_dist;

	min_to_max_ratio = mindist_ / maxdist_;
	maxdist_sq = maxdist_ * maxdist_;
	maxdist_sq_minus_eps = maxdist_sq * (49.0 + min_to_max_ratio) / 50.0; // exp factor will be exp(-100)
	exp_ratio = -2 * (1 - min_to_max_ratio * min_to_max_ratio);

	// maps mindist = 4/5 * maxdist -> cheb_x=1
	const double cheb_acos_x = acos(1.25 * min_to_max_ratio);
	for (int i = 0; i < size_; i++)
		for (int j = 0; j < 3; j++) {
			double accumulator = 0;
			for (int k = RADIAL_BASIS_CHEB_SIZE - 1; k >= 0; k--)
				// chebyshev polynomial (k,x):        
				accumulator += CHEB_RECURSIVE_COEFFS[i][j][k] * cos(2 * k * cheb_acos_x);
			recursive_coeffs[i][j] = accumulator;
		}

	{
		double accumulator = 0;
		for (int k = RADIAL_BASIS_CHEB_SIZE - 1; k >= 0; k--)
			// chebyshev polynomial (k,x):        
			accumulator += CHEB_ZEROTH_POLY[k] * cos(2 * k * cheb_acos_x);
		zeroth_poly = accumulator;
	}
}

RadialBasis_Shapeev2::RadialBasis_Shapeev2(double _min_dist, double _max_dist, int _size)
	: AnyRadialBasis(_min_dist, _max_dist, _size)
{
	if (rb_size > RADIAL_BASIS_MAX_SIZE)
		ERROR("RadialBasis error: allocated degree exceeded.");
	InitShapeev2RB();
}

RadialBasis_Shapeev2::RadialBasis_Shapeev2(std::ifstream& ifs)
	: AnyRadialBasis(ifs)
{
	if (rb_size > RADIAL_BASIS_MAX_SIZE)
		ERROR("RadialBasis error: allocated degree exceeded.");
	InitShapeev2RB();
}

void RadialBasis_Shapeev2::RB_Calc(double r)
{
	double r_sq = r * r;
	if (r_sq >= maxdist_sq_minus_eps) {
		memset(&rb_vals[0], 0, sizeof(double) * rb_size);
		memset(&rb_ders[0], 0, sizeof(double) * rb_size);
		return;
	}
	const double x_sq = r_sq / maxdist_sq;
	const double mult = scaling * exp(exp_ratio / (1 - x_sq));
	rb_vals[0] = zeroth_poly * mult;
	rb_ders[0] = exp_ratio / ((1 - x_sq) * (1 - x_sq)) * zeroth_poly * mult;
	double prev_val = 0;
	double prev_der = 0;
	for (int i = 0; i < size_ - 1; i++) {
		rb_vals[i + 1] = recursive_coeffs[i][0] * (
			(x_sq + recursive_coeffs[i][1]) * rb_vals[i] + recursive_coeffs[i][2] * prev_val);
		rb_ders[i + 1] = recursive_coeffs[i][0] * (
			rb_vals[i]
			+
			(x_sq + recursive_coeffs[i][1]) * rb_ders[i] + recursive_coeffs[i][2] * prev_der);
		prev_val = rb_vals[i];
		prev_der = rb_ders[i];
	}
}


void RadialBasis_Chebyshev::RB_Calc(double r)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	double mult = 2.0 / (max_dist - min_dist);
	double ksi = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist);

	rb_vals[0] = scaling * (1 * (r - max_dist)*(r - max_dist));
	rb_ders[0] = scaling * (0 * (r - max_dist)*(r - max_dist) + 2 * (r - max_dist));
	rb_vals[1] = scaling * (ksi*(r - max_dist)*(r - max_dist));
	rb_ders[1] = scaling * (mult * (r - max_dist)*(r - max_dist) + 2 * ksi*(r - max_dist));
	for (int i = 2; i < rb_size; i++) {
		rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
		rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
	}
}

void RadialBasis_Chebyshev_repuls::RB_Calc(double r)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	if (r < min_dist)
		r = min_dist;

	double mult = 2.0 / (max_dist - min_dist);
	double ksi = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist);

	rb_vals[0] = scaling * (1 * (r - max_dist)*(r - max_dist));
	rb_ders[0] = scaling * (0 * (r - max_dist)*(r - max_dist) + 2 * (r - max_dist));
	rb_vals[1] = scaling * (ksi*(r - max_dist)*(r - max_dist));
	rb_ders[1] = scaling * (mult * (r - max_dist)*(r - max_dist) + 2 * ksi*(r - max_dist));
	for (int i = 2; i < rb_size; i++) {
		rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
		rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
	}
	if (r == min_dist)
		for (int i = 0; i < rb_size; i++)
			rb_ders[i] = 0.0;
}

void RadialBasis_Taylor::RB_Calc(double r)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	rb_vals[0] = scaling * 1;
	rb_ders[0] = scaling * 0;
	for (int i = 1; i < rb_size; i++)
	{
		rb_ders[i] = scaling * i * rb_vals[i - 1];
		rb_vals[i] = scaling * r * rb_vals[i - 1];
	}
}
