/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>

#include "mtpr_train.h"

using namespace std;

char* mtpfnm;
char* cfgfnm;

int prank = 0;
int psize = 1;

vector<Configuration> training_set;
vector<Configuration> validSet;

void AddConfigs(const string cfgfnm, NonLinearRegression &dtr, int proc_rank, int proc_size)
{

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	Configuration cfg;
	ifstream ifs(cfgfnm, ios::binary);

	for (int cntr = 0; cfg.Load(ifs); cntr++)
	{
		if (cntr % proc_size == proc_rank)
			training_set.push_back(cfg);

	}

	ifs.close();

}

void Rescale(MTPR_trainer& trainer,MLMTPR& mtpr)
{
		double min_scaling = mtpr.scaling;
		double max_scaling = mtpr.scaling;
		int ind;
		do {
			double condition_number[5];
			double scaling = mtpr.scaling;
			double scalings[5] = { scaling / 1.2,scaling / 1.1, scaling, scaling * 1.1, scaling*1.2 };
			vector<double> coeffs;
			coeffs.resize(mtpr.linear_coeffs.size());
			if (prank == 0) std::cout << "Rescaling...\n";
			for (int j = 0; j < 5; j++) {
				if (prank == 0) std::cout << "   scaling = " << scalings[j] << ", condition number = ";
				mtpr.scaling = scalings[j];
				trainer.TrainLinear(prank, training_set);
				mtpr.LinCoeff();
				double rms = 0;

				for (int i = 0; i < mtpr.linear_coeffs.size(); i++) {
					coeffs[i] = std::abs(mtpr.linear_coeffs[i]);
					rms += coeffs[i] * coeffs[i];
				}
				rms = sqrt(rms);
				std::sort(coeffs.begin(), coeffs.end());

				double median = coeffs[coeffs.size() / 2];

				condition_number[j] = rms / median;
				if (prank == 0) std::cout << rms / median << "\n";
			}

			// finds minimal condition number
			ind = 2;
			for (int j = 0; j < 5; j++)
				if (condition_number[j] < condition_number[ind]) ind = j;

			mtpr.scaling = scalings[ind];
			if (prank == 0) std::cout << "Rescaling to " << mtpr.scaling << "... ";
			trainer.TrainLinear(prank, training_set);
			if (prank == 0) std::cout << "done" << std::endl;
			if ((min_scaling < mtpr.scaling) && (mtpr.scaling < max_scaling))
				ind = 2;
			else {
				min_scaling = std::min(min_scaling, mtpr.scaling);
				max_scaling = std::max(max_scaling, mtpr.scaling);
			}
		} while (ind != 2);
}

void Train_MTPR(std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	//args[0] - potname
	//args[1] - ts_name

	double weight_energy = 1.0;
	if (opts["energy-weight"] != "")
		weight_energy = stod(opts["energy-weight"]);

	double weight_force = 0.01;
	if (opts["force-weight"] != "")
		weight_force = stod(opts["force-weight"]);

	double weight_stress = 0.001;
	if (opts["stress-weight"] != "")
		weight_stress = stod(opts["stress-weight"]);

	double scale_by_force = 0.0;
	if (opts["scale-by-force"] != "")
		scale_by_force = stod(opts["scale-by-force"]);

	string validfnm = "";
	if (opts["valid-cfgs"] != "")
		validfnm = opts["valid-cfgs"];

	int maxits = 1000;
	if (opts["max-iter"] != "")
		maxits = stoi(opts["max-iter"]);

	bool skip_preinit = false;
	if (opts["skip-preinit"] != "")
		skip_preinit = true;

	string curr_fnm = "";
	if (opts["curr-pot-name"] != "")
		curr_fnm = opts["curr-pot-name"];

	string trained_fnm = "Trained.mtp_";
	if (opts["trained-pot-name"] != "")
		trained_fnm = opts["trained-pot-name"];

	double bfgs_conv_tol = 1e-3;
	if (opts["bfgs-conv-tol"] != "")
		bfgs_conv_tol = stod(opts["bfgs-conv-tol"]);

	string weighting = "vibrations";
	if (opts["weighting"] != "")
		weighting = opts["weighting"];

	if (opts["init-params"] == "")
		opts["init-params"] = "random";
	if (opts["init-params"] != "random" && opts["init-params"] != "same")
		ERROR("--init-params should be 'random' or 'same'");
		
	bool mindist_update = false;
	if (opts["update-mindist"] != "")
	    mindist_update = true;

	SetTagLogStream("dev", &std::cout);
	int end = 1;
	MLMTPR mtpr = MLMTPR();
	for (int i = 0; i < end; i++) {
		try {
			mtpr.Load(args[0]);
			end = 1;
		}
		catch (MlipException& exp) {
			std::cout << exp.What() << std::endl;
			end = 10;
		}
	}
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
#endif

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	MTPR_trainer trainer(&mtpr, weight_energy, weight_force, weight_stress, scale_by_force, 1e-9, curr_fnm, 0);
	//LOSS FUNCTIONAL MODIFICATION!!!
	trainer.weighting = weighting;
	trainer.linstop = bfgs_conv_tol;	//if in 100 iterations loss decreases less than this, BFGS is finished
	trainer.curr_pot_name = curr_fnm;

	if (prank == 0)
		std::cout << "MTPR from " << args[0]<< ", Database: " << args[1] << std::endl;

	Configuration cfg;
	if (validfnm != "")
	{
		try
		{
			ifstream ifs(validfnm, ios::binary);
			for (int cntr = 0; cfg.Load(ifs); cntr++)
			{
				if (cntr % psize == prank)
					validSet.push_back(cfg);
			}
			ifs.close();
			if (prank == 0)
				std::cout << "validation set: " << validfnm << std::endl;
		}
		catch (MlipException& exp)
		{
			std::cout << exp.What() << std::endl;
		}
	}

	for (int i = 0; i < end; i++)
	{
		try
		{
#ifdef MLIP_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			training_set.clear();
			AddConfigs(args[1], trainer, prank, psize);
			end = 1;
		}
		catch (MlipException& exp)
		{
			std::cout << exp.What() << std::endl;
			end = 10;
		}
	}


	//random initialization of radial coefficients
	if (opts["init-params"] == "random" && !mtpr.inited) {
		if (prank == 0) {
			std::random_device rand_device;
			std::default_random_engine generator(rand_device());
			std::uniform_real_distribution<> uniform(-1.0, 1.0);

			std::cout << "Random initialization of radial coefficients" << std::endl;
			int rb_size = mtpr.Get_RB_size();
			for (int k = 0; k < mtpr.species_count*mtpr.species_count; k++)
				for (int i = 0; i < mtpr.radial_func_count; i++) {
					for (int j = 0; j < rb_size; j++)
						mtpr.regression_coeffs[k*mtpr.radial_func_count*rb_size +
						i*rb_size + j] = 5e-7*uniform(generator);

					mtpr.regression_coeffs[k*mtpr.radial_func_count*rb_size +
						i*rb_size + min(i, rb_size - 1)] = 5e-7 * uniform(generator);
				}
		}

#ifdef MLIP_MPI
		MPI_Bcast(&mtpr.Coeff()[0], mtpr.CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	}

	if (!mtpr.inited && maxits > 0 && !skip_preinit) {
		trainer.max_step_count = 75;
		trainer.TrainLinear(prank, training_set);
		Rescale(trainer, mtpr);

		if (prank == 0)
			std::cout << "Pre-training started" << std::endl;

		trainer.Train(training_set);

		Rescale(trainer, mtpr);

		if (prank == 0)
			std::cout << "Pre-training ended" << std::endl;
	}
	
        //getting the lowest min_dist for the training set
        double min_dist = 999;
        for (int i = 0; i < training_set.size(); i++) {
             if (training_set[i].MinDist() < min_dist)
                 min_dist = training_set[i].MinDist();
        }

        double total_min_dist = min_dist;
        
        //finding minimum distance in configuration among the processes
#ifdef MLIP_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&min_dist, &total_min_dist, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

        if (mindist_update) {
            if (prank == 0)
            {
                std::cout << "Found configuration with mindist=" << total_min_dist << ", MTP's mindist will be decreased\n";
            }
            mtpr.p_RadialBasis->min_dist = 0.99 * total_min_dist;
        }

	trainer.max_step_count = maxits;				//maximum step count (linesearch doesn't count)

	if (maxits > 0) {
		if (prank == 0) {
			std::cout << "BFGS iterations count set to " << trainer.max_step_count << std::endl;
			std::cout << "BFGS convergence tolerance set to " << bfgs_conv_tol << std::endl;
			if ((weight_energy != 0) || (weight_force != 0) || (weight_stress != 0)) {
				std::cout << "Energy weight: " << weight_energy << std::endl;
				std::cout << "Force weight: " << weight_force << std::endl;
				std::cout << "Stress weight: " << weight_stress << std::endl;
			}
		}

		trainer.Train(training_set);
		trainer.TrainLinear(prank, training_set);
		Rescale(trainer, mtpr);

		if (prank == 0)
			mtpr.Save(trained_fnm);

	}
	ErrorMonitor errmon, bufferrmon;
	std::cout.precision(15);
	errmon.reset();
	double errmax = 0;
	int ind = -1;

	for (Configuration& cfg_orig : training_set)
	{
		cfg = cfg_orig;
		mtpr.CalcEFS(cfg);
		errmon.collect(cfg_orig, cfg);

	}
	bufferrmon.reset();
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&errmon.ene_all.max, &bufferrmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.ene_all.sum, &bufferrmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.epa_all.max, &bufferrmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.epa_all.sum, &bufferrmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.frc_all.max, &bufferrmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.frc_all.sum, &bufferrmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.ene_all.count, &bufferrmon.ene_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.epa_all.count, &bufferrmon.epa_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.frc_all.count, &bufferrmon.frc_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.str_all.count, &bufferrmon.str_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.str_all.max, &bufferrmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.str_all.sum, &bufferrmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.vir_all.count, &bufferrmon.vir_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.vir_all.max, &bufferrmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&errmon.vir_all.sum, &bufferrmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (prank == 0)
	{
		std::cout << std::endl << "\t\t* * * TRAIN ERRORS * * *" << std::endl << std::endl;
		std::string report;
		bufferrmon.GetReport(report);
		std::cout << report.c_str();
	}

#else
	std::cout << std::endl << "\t\t* * * TRAIN ERRORS * * *" << std::endl << std::endl;
	std::string report;
	errmon.GetReport(report);
	std::cout << report.c_str();
#endif

	errmon.reset();
	bufferrmon.reset();
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (validfnm != "")
	{
		for (Configuration& cfg_orig : validSet)
		{
			cfg = cfg_orig;
			mtpr.CalcEFS(cfg);
			errmon.collect(cfg_orig, cfg);
		}
		bufferrmon.reset();
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.max, &bufferrmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.sum, &bufferrmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.max, &bufferrmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.sum, &bufferrmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.max, &bufferrmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.sum, &bufferrmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.count, &bufferrmon.ene_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.count, &bufferrmon.epa_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.count, &bufferrmon.frc_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.count, &bufferrmon.str_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.max, &bufferrmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.sum, &bufferrmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.count, &bufferrmon.vir_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.max, &bufferrmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.sum, &bufferrmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (prank == 0)
		{
			std::cout << std::endl << "\t\t* * * VALIDATION ERRORS * * *" << std::endl << std::endl;
			std::string report;
			bufferrmon.GetReport(report);
			std::cout << report.c_str();
		}
#else
		std::cout << std::endl << "\t\t* * * VALIDATION ERRORS * * *" << std::endl << std::endl;
		std::string report;
		errmon.GetReport(report);
		std::cout << report.c_str();
#endif
	}
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Finalize();
#endif
}
