/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Ivan Novikov
 */

#ifndef MLIP_PAIR_POTENTIALS_H
#define MLIP_PAIR_POTENTIALS_H


#include "basic_potentials.h"
#include <iostream>


class PairPotential : public AnyPotential
{
public:
	double cutoff = 5.0;

	virtual double F(double r) = 0;
	virtual double dF_dr(double r) = 0;

	virtual double f_cut(double r)
	{
		return (r < cutoff) ? 1.0 - exp(-pow((r - cutoff) / 0.5, 2)) : 0.0;
	}
	virtual double dfcut_dr(double r)
	{
		return  (r < cutoff) ? 8.0 * (r - cutoff) * exp(-pow((r - cutoff) / 0.5, 2)) : 0.0;
	}

	void CalcEFS(Configuration& cfg)
	{
		ResetEFS(cfg);
		cfg.has_site_energies(true);

		Neighborhoods neighborhoods(cfg, cutoff);

		for (int i = 0; i < cfg.size(); i++) {
			Neighborhood& nbh = neighborhoods[i];
			for (int j = 0; j < nbh.count; j++)
			{
				double r = nbh.dists[j];

				cfg.site_energy(i) += 0.5*F(r)*f_cut(r);

				for (int a = 0; a < 3; a++)
				{
					double force_val = (dF_dr(r)*f_cut(r) + F(r)*dfcut_dr(r)) * nbh.vecs[j][a] / r;
					cfg.force(i, a) += force_val;
					for (int b = 0; b < 3; b++)
						cfg.stresses[a][b] -= 0.5 * force_val * nbh.vecs[j][b];
				}
			}
			cfg.energy += cfg.site_energy(i);
		}
	}
};


class TestPotential : public PairPotential
{
public:
	double r0;
	double alpha;

	TestPotential(	double _r0, double _alpha, double _cutoff) : 
		r0(_r0), alpha(_alpha) 
	{
		cutoff = _cutoff;
	};

	double F(double r) override
	{
		//	return exp(-2 * alpha * (r - r0)) - 2 * exp(-alpha * (r - r0));
		double tmp = exp(-alpha * (r - r0));
		return tmp * (tmp - 2.0);
		//		return -2.0 / r;
	}

	double dF_dr(double r) override
	{
		//	return -2 * exp(-2 * alpha * (r - r0)) + 2 * exp(-alpha * (r - r0));
		double tmp = exp(-alpha * (r - r0));
		return 2.0 * alpha * tmp * (1 - tmp);
		//		return 2.0 / (r*r);
	}
};

// TODO: rename to LennardJones
class LJ : public PairPotential, protected InitBySettings
{
private:
	double r_min; // minimizer of the pair function (in Angstroms)
	double scale; // Value of pair function at the minimum (in eV)

	void InitSettings() // Sets correspondence between variables and setting names in settings file
	{
		MakeSetting(r_min, "r_min");
		MakeSetting(scale, "scale");
		MakeSetting(cutoff, "cutoff");
	}

public:
	LJ(const Settings& settings)
	{
		InitSettings();
		ApplySettings(settings);
	}

	LJ(double _r_min, double _scale, double _cutoff) :
		r_min(_r_min), scale(_scale)
	{
		cutoff = _cutoff;
	};


	double F(double r) override
	{
		r = r_min/r;
		double r6 = pow(r*r, 3);
		double r12 = r6*r6;
		return scale * (r12 - 2.0*r6);
	}

	double dF_dr(double r) override
	{
		r = r_min/r;
		double r6 = pow(r*r, 3);
		double r12 = r6*r6;
		return 12 * scale * (r6 - r12) * r;
	}
};

class ZBL : public AnyPotential
{

private:
    const std::string filename;
    int species_count;
    double r_min, r_cut;
    std::vector<int> z;
    const double m1 = 0.18175;
    const double m2 = 0.50986;
    const double m3 = 0.28022;
    const double m4 = 0.02817;
    const double p1 = -3.1998;
    const double p2 = -0.94229;
    const double p3 = -0.4029;
    const double p4 = -0.20162;

public:
    void Load(const std::string& filename) {
    
        std::ifstream ifs(filename);
        if (!ifs.is_open())
        ERROR((std::string)"Cannot open " + filename);
        
        char tmpline[1000];
        std::string tmpstr;

        ifs.getline(tmpline, 1000);
        int len = (int)((std::string)tmpline).length();
        if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
            tmpline[len - 1] = '\0';

        if ((std::string)tmpline != "ZBL")
            ERROR("Can read only ZBL format potentials");

        ifs >> tmpstr;
        if (tmpstr != "species_count")
            ERROR("Error reading species_count in .zbl file");
        ifs.ignore(2);
        ifs >> species_count;
        
        z.resize(species_count);
        for (int i = 0; i < species_count; i++) 
            ifs >> z[i];
        
        ifs >> tmpstr;
        if (tmpstr != "min_dist")
            ERROR("Error reading min_dist in .zbl file");
        ifs.ignore(2);
        ifs >> r_min;
        
        ifs >> tmpstr;
        if (tmpstr != "max_dist")
            ERROR("Error reading max_dist in .zbl file");
        ifs.ignore(2);
        ifs >> r_cut;
        
    } 

    ZBL(const std::string& filename_) :
        filename(filename_)
    {
        Load(filename);
    }
    
    double F(double r, double a)
    {
        double phi = 0;
        phi += m1 * exp(p1 * r / a);
        phi += m2 * exp(p2 * r / a);
        phi += m3 * exp(p3 * r / a);
        phi += m4 * exp(p4 * r / a);
        return phi / r;
    }
    double dF_dr(double r, double a)
    {
        double phi = 0;
        phi += m1 * exp(p1 * r / a);
        phi += m2 * exp(p2 * r / a);
        phi += m3 * exp(p3 * r / a);
        phi += m4 * exp(p4 * r / a);
        double d_phi_dr = 0;
        d_phi_dr += p1 * m1 * exp(p1 * r / a) / a;
        d_phi_dr += p2 * m2 * exp(p2 * r / a) / a;
        d_phi_dr += p3 * m3 * exp(p3 * r / a) / a;
        d_phi_dr += p4 * m4 * exp(p4 * r / a) / a;
        return - phi / (r * r) + d_phi_dr / r;
    }
    
    double f_repulsion(double r)
    {
        double A = -1 * (r_cut - r_min) / (r_cut - r);
        double B = exp((r_cut - r) / (r_min - r));
        if (r < 1.05 * r_min) return 1;
        else if (1.05 * r_min <= r < 0.95 * r_cut) return exp(A * B);
        else return 0;
    }
    double df_repulsion_dr(double r)
    {
        double A = -1 * (r_cut - r_min) / (r_cut - r);
        double B = exp((r_cut - r) / (r_min - r));
        double dA_dr = -1 * (r_cut - r_min) / ((r_cut - r) * (r_cut - r));
        double dB_dr = B * (- 1.0 / (r_min - r) + (r_cut - r) / ((r_min - r)*(r_min - r)));
        if (1.05 * r_min <= r < 0.95 * r_cut) return exp(A * B) * (dA_dr * B + A * dB_dr);
        else return 0;
    }
    
    void CalcEFS(Configuration& cfg)
    {
        ResetEFS(cfg);
        cfg.has_site_energies(true);

        Neighborhoods neighborhoods(cfg, r_cut);

        for (int i = 0; i < cfg.size(); i++) {
            Neighborhood& nbh = neighborhoods[i]; 
            int type_central = nbh.my_type;
            for (int j = 0; j < nbh.count; j++)
            {
                double r = nbh.dists[j];
                int type_outer = nbh.types[j];
                double C = 10000 * z[type_central] * z[type_outer] / (4 * M_PI * 55.26349406);
                double a = 0.4685 / (pow(z[type_central], 0.23) + pow(z[type_outer], 0.23));

                cfg.site_energy(i) += C * F(r,a) * f_repulsion(r);

                for (int l = 0; l < 3; l++)
                {
                    double force_val = C * (dF_dr(r,a)*f_repulsion(r) + F(r,a)*df_repulsion_dr(r)) * nbh.vecs[j][l] / r;
                    cfg.force(i, l) += 2 * force_val;
                    for (int b = 0; b < 3; b++)
                        cfg.stresses[l][b] -= force_val * nbh.vecs[j][b];
                }
            }
            cfg.energy += cfg.site_energy(i);
        }
    }

    ~ZBL(){};

};

#endif //#ifndef MLIP_PAIR_POTENTIALS_H

