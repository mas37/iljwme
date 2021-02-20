/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */
#include <algorithm>
#include <sstream>
#include "mlp.h"
#include "../mtpr_trainer.h"
#include "../wrapper.h"
#include "../drivers/basic_drivers.h"
#include "../drivers/relaxation.h"
#include "../magnetic_moments.h"


bool RunAllTests(bool is_parallel);
using namespace std;


// does a number of unit tests (not dev)
// returns true if all tests are successful
// otherwise returns false and stops further tests

bool self_test()
{
    ofstream logstream("temp/log");
    SetStreamForOutput(&logstream);

#ifndef MLIP_DEBUG
    if (mpi.rank == 0) {
        std::cout << "Note: self-test is running without #define MLIP_DEBUG;\n"
            << "      build with -DMLIP_DEBUG and run if troubles encountered" << std::endl;
    }
#endif

#ifndef MLIP_MPI
    cout << "Serial tests:" << endl;
    if (!RunAllTests(false)) return false;
#else
    if (mpi.rank == 0) 
        cout << "mpi tests (" << mpi.size << " cores):" << endl;
    if (!RunAllTests(true)) return false;
#endif

    logstream.close();
    return true;
}


bool Commands(const string& command, vector<string>& args, map<string, string>& opts)
{
    bool is_command_found = false;

    if (command == "list" || command == "help") 
    {
        if(command == "list" || args.size() == 0) is_command_found = true;
        if (mpi.rank == 0) 
        {
            cout << "mlp "
#ifdef MLIP_MPI
                << "mpi version (" << mpi.size << " cores), "
#else
                << "serial version, "
#endif
                << "(C) A. Shapeev, E. Podryabinkin, I. Novikov (Skoltech).\n";
            if (command == "help" && args.size() == 0)
                cout << USAGE;
            if (command == "list")
                cout << "List of available commands:\n";
        }
    }

#include "../sample_potentials.h"

    BEGIN_COMMAND("run",
        "reads configurations from a cfg_filename and processes them according to the settings specified in mlip.ini",
        "mlp run mlip.ini cfg_filename [settings]:\n"
        "Settings can be given in any order and are with the same names as in mlip.ini.\n"
        "    Settings given from the command line overrides the settings taken from the file.\n"
        ) 
    {

        if (args.size() != 2) {
            cout << "mlp run: 2 argument required\n";
            return 1;
        }
        string mlip_fnm = args[0];
        Message("Settings are read from \""+mlip_fnm+"\"");
        string read_cfg_fnm = args[1];
        Message("Configurations are read from \""+read_cfg_fnm+"\"");

        ios::sync_with_stdio(false);

        Settings settings;
        settings = LoadSettings(mlip_fnm);
        settings.Modify(opts);

        Wrapper mlip_wrapper(settings);

        map<string, string> driver_settings;
        driver_settings.emplace("cfg_filename", read_cfg_fnm);

        CfgReader cfg_reader(driver_settings, &mlip_wrapper);

        cfg_reader.Run();

        Message("Configuration reading complete");
    } END_COMMAND;

    BEGIN_COMMAND("calculate_efs",
        "calculate energy, forces and stresses for configurations from a file",
        "mlp calculate_efs pot.mtp cfg_filename [settings]:\n"
        "reads configurations from a cfg_filename and calculates energy, forces and stresses for them"
        "Settings can be given in any order"
        "    --output_filename=<filename> file for output configurations. The same as cfg_filename by default.\n"
    )
    {
        if (args.size() != 2) {
            cout << "mlp calculate_efs: 2 arguments required\n";
            return 1;
        }
        string mlip_filename = args[0];
        string read_cfg_fnm = args[1];

        MLMTPR mtp(mlip_filename);
        Message("MTP loaded from " + mlip_filename);

        if (opts["output_filename"].empty())
        {
            auto cfgs = MPI_LoadCfgs(read_cfg_fnm);
            Message("Configurations are read from \""+read_cfg_fnm+"\"");
            for (auto& cfg : cfgs)
            {
                cfg.has_site_energies(true);
                mtp.CalcEFS(cfg);
            }
            
            {
                ofstream ofs(read_cfg_fnm);
                if (!ofs.is_open())
                    ERROR("Can't open output configurations file");
            }

            int max_count = 0;
            int local_count = (int)cfgs.size();
            MPI_Allreduce(&local_count, &max_count, 1, MPI_INT, MPI_MAX, mpi.comm);

            for (Configuration& cfg : cfgs)
                for (int rnk=0; rnk<mpi.size; rnk++)
                {
                    if (rnk == mpi.rank)
                        cfg.AppendToFile(read_cfg_fnm);

                    MPI_Barrier(mpi.comm);
                }
            if (cfgs.size() < max_count)
                for (int rnk=0; rnk<mpi.size; rnk++)
                    MPI_Barrier(mpi.comm);
        }
        else
        {
            ifstream ifs(read_cfg_fnm, ios::binary);
            if (ifs.is_open())
                Message("Configurations are read from \""+read_cfg_fnm+"\"");
            else
                ERROR("Can't open input configurations file");

            string out_filename = opts["output_filename"] + mpi.fnm_ending;
            {
                ofstream ofs(out_filename);
                if (!ofs.is_open())
                    ERROR("Can't open output configurations file");
            }

            int count=0;
            for (Configuration cfg; cfg.Load(ifs); count++)
                 if (count % mpi.size == mpi.rank)
                 {
                     cfg.has_site_energies(true);
                     mtp.CalcEFS(cfg);
                     cfg.AppendToFile(out_filename);
                 }
        }

        Message("EFS calculation complete");
    } END_COMMAND;

    BEGIN_COMMAND("relax",
        "relaxes the configuration(s)",
        "mlp relax mlip.ini for_relax_filename relaxed_filename [settings]:\n"
        "for_relax_filename is a file with the configurations for relaxation.\n"
        "relaxed_filename is a file with the relaxed configurations.\n"
        "Settings can be given in any order and can be with the same names as in mlip.ini.\n"
        "    --save_unrelaxed=<filename>: cfg_relax_failed_filename is a file with the original configurations which relaxation failed.\n"
        "    Settings given from the command line overrides the settings taken from the file.\n"
    ) 
    {
        if (args.size() != 3)
        {
            cout << "mlp relax: 3 argument required\n";
            return 1;
        }
        string mlip_fnm = args[0];
        Message("Settings are read from \""+mlip_fnm+"\"");
        string read_cfg_fnm = args[1];
        Message("Configurations are read from \""+read_cfg_fnm+"\"");
        string relaxed_cfg_fnm = args[2];
        Message("Relaxed configurations will be saved to \""+relaxed_cfg_fnm+"\"");

        string unrelaxed_cfg_fnm;
        if (!opts["save_unrelaxed"].empty())
        {
            unrelaxed_cfg_fnm = opts["save_unrelaxed"];
            Message("Unrelaxed configurations will be saved to \""+relaxed_cfg_fnm+"\"");
        }

        Settings settings;
        settings = LoadSettings(mlip_fnm);
        settings.Modify(opts);

#ifdef MLIP_MPI
        relaxed_cfg_fnm += mpi.fnm_ending;
        if (!unrelaxed_cfg_fnm.empty())
            unrelaxed_cfg_fnm += mpi.fnm_ending;
#endif // MLIP_MPI

        ios::sync_with_stdio(false);

        // Fix settings, switch off settings enabling interprocessor communication
        settings["calculate_efs"] = "TRUE";
        settings["fit"] = "FALSE";
        if (settings["fit"] == "TRUE" ||
            settings["fit"] == "True" ||
            settings["fit"] == "true")
            ERROR("Learning is not compatible with relaxation. \
                   Change \"fit\" setting to \"FALSE\"");
        if (settings["lotf"] == "TRUE" ||
            settings["lotf"] == "True" ||
            settings["lotf"] == "true")
            ERROR("Learning is not compatible with relaxation. \
                   Change \"lotf\" setting to \"FALSE\"");
        if (settings["select:update_active_set"] == "TRUE" ||
            settings["select:update_active_set"] == "True" ||
            settings["select:update_active_set"] == "true")
            ERROR("Active set update is not allowed while relaxation. \
                   Change \"select:update_active_set\" setting to \"FALSE\"");

        Wrapper mlip_wrapper(settings);

        Settings driver_settings(settings.ExtractSubSettings("relax"));

        Relaxation rlx(driver_settings, &mlip_wrapper);

        ofstream ofs_relaxed, ofs_unrelaxed;
        // check and clean output files
        ifstream ifs(read_cfg_fnm, ios::binary);
        if (!ifs.is_open())
            ERROR("Cannot open \"" + read_cfg_fnm + "\" file for input");
        ofs_relaxed.open(relaxed_cfg_fnm, ios::binary);
        if (!ofs_relaxed.is_open())
            ERROR("Cannot open \"" + relaxed_cfg_fnm + "\" file for output");

        if (relaxed_cfg_fnm != unrelaxed_cfg_fnm && !unrelaxed_cfg_fnm.empty())
        {
            ofs_unrelaxed.open(unrelaxed_cfg_fnm, ios::binary);
            if (!ofs_unrelaxed.is_open())
                ERROR("Cannot open \"" + unrelaxed_cfg_fnm + "\" file for output");
        }

		int relaxed_proc=0;

        Configuration cfg_orig;                                //to save this cfg if failed to relax
        int count=0;
        bool error;                                            //was the relaxation successful or not
        for (Configuration cfg; cfg.Load(ifs); count++)
            if (count%mpi.size == mpi.rank)
            {
                if (cfg.size() != 0)
                {
                    error = false;
                    cfg_orig = cfg;
                    rlx.cfg = cfg;

                    try { rlx.Run(); }
                    catch (MlipException& e) {
                        Warning("Relaxation of cfg#"+to_string(count+1)+" failed: "+e.message);
                        error = true;
                    }

                    if (!error) 
                    {
                        rlx.cfg.Save(ofs_relaxed);
                        relaxed_proc++;
					}
                    else if (!unrelaxed_cfg_fnm.empty()) 
                    {
                        if (rlx.cfg.features["from"]=="" || 
                            rlx.cfg.features["from"]=="relaxation_OK")
                            rlx.cfg.features["from"]="relaxation_FAILED";

                        if (relaxed_cfg_fnm != unrelaxed_cfg_fnm)
                            cfg_orig.Save(ofs_unrelaxed);
                        else 
                            cfg_orig.Save(ofs_relaxed);
                    }
                }
            }

		int relaxed_total=relaxed_proc;
        MPI_Allreduce(&relaxed_proc, &relaxed_total, 1, MPI_INT, MPI_SUM, mpi.comm);

        Message("From " + to_string(count) + " configurations " + to_string(relaxed_total) + " were relaxed successfully\n");

    } END_COMMAND;

    BEGIN_COMMAND("select_add",
        "Actively selects configurations to be added to the current training set",
        "mlp select_add pot.mtp train_set.cfg candidates.cfg new_selected.cfg:\n"
        "actively selects configurations from candidates.cfg and saves to new_selected.cfg those that are required to update the current training set (in train_set.cfg)\n"
        "The specified selection weights overwrites the weights stored in MTP file (if MTP file contain the selection state). Default values of weights are set only if MTP file has no selection state data and the weights are not specified by options"
        "  Options:\n"
        "  --energy_weight=<double>: set the weight for energy equation, default=1\n"
        "  --force_weight=<double>: set the weight for force equations, default=0\n"
        "  --stress_weight=<double>: set the weight for stress equations, default=0\n"
        "  --site_en_weight=<double>: set the weight for site energy equations, default=0\n"
        "  --weight_scaling=<0 or 1 or 2>: how energy and stress weights are scaled with the number of atoms. final_energy_weight = energy_weight/N^(weight_scaling/2)\n"
        "  --swap_limit=<int>: swap limit for multiple selection, unlimited by default\n"
        "  --batch_size=<int>: size of batch, default=9999 \n"
        "  --save_to=<string>: save maxvol state to this mvs file\n"
        "  --save_selected_to=<string>: filename for saving the active set\n"
        "  --threshold=<double>: set the select threshold to num, default=1.001\n"
        "  --log=<string>: where to write the selection log, default="" (none) \n"
    ) {

        if (args.size() != 4) {
            std::cout << "\tError: 4 arguments required\n";
            return 1;
        }

        const string mtp_filename = args[0];
        const string train_filename = args[1];   
        const string candidates_filename = args[2];
        const string new_selected_filename = args[3];

        int selection_limit=0;                      //limits the number of swaps in MaxVol

        Settings settings;

        settings.Modify(opts);

        MLMTPR mtp(mtp_filename);
        Message("MTP loaded from " + mtp_filename);

        CfgSelection select(&mtp, settings);
        if (settings.GetMap().count("energy_weight") == 0 &&
            settings.GetMap().count("force_weight") == 0 &&
            settings.GetMap().count("stress_weight") == 0 &&
            settings.GetMap().count("site_en_weight") == 0 &&
            settings.GetMap().count("weight_scaling") == 0)
        {
            try { select.Load(args[0]); }
            catch (MlipException& excp) { Message(excp.What()); }
        }
//        select.Reset();

        auto train_cfgs = MPI_LoadCfgs(train_filename);
        Message("Training set loaded from " + train_filename);

        // selection from training set
        for (auto& cfg : train_cfgs)
            select.AddForSelection(cfg);
        double threshold_backup = select.threshold_select;
        select.threshold_select = 1.001; // set minimal threshold for selection from the training set
        select.Select();
        select.threshold_select = threshold_backup; // restore given threshold
        train_cfgs.clear();
        int local = (int)select.selected_cfgs.size();
        int global;
        MPI_Reduce(&local, &global, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
        Message(to_string(global) + " configurations selected from training set");

        // selection from candidates
        ifstream ifs(candidates_filename, ios::binary);
        if (!ifs.is_open())
            ERROR("Can't open file \"" + candidates_filename + "\" for reading configurations");
        int count = 0;
        for (Configuration cfg; cfg.Load(ifs); count++)
        {
            if (count % mpi.size == mpi.rank)
            {
                cfg.features["!@$%^&is_new"] = "true";
                select.Process(cfg);
            }
        }
        ifs.close();
        Configuration cfg;
        if ((count % mpi.size != 0) && mpi.rank >= (count % mpi.size))
            select.Process(cfg);
        select.Select((select.swap_limit>0) ? select.swap_limit : HUGE_INT);
        Message("Loading and selection of candidates completed");
        
        // selecting from training set again
        train_cfgs = MPI_LoadCfgs(train_filename);
        for (auto& cfg : train_cfgs)
            select.AddForSelection(cfg);
        threshold_backup = select.threshold_select;
        select.Select();
        select.threshold_select = 1.001; // set minimal threshold for selection from the training set
        select.threshold_select = threshold_backup; // restore given threshold
        train_cfgs.clear();

        // saving freshely selected configurations
        if (mpi.rank == 0)
            ofstream ofs(new_selected_filename); // clean output file

        count = 0;
        for (int rnk=0; rnk<mpi.size; rnk++)
        {
            if (rnk == mpi.rank)
            {
                ofstream ofs(new_selected_filename, ios::binary | ios::app);
                if (!ofs.is_open())
                    ERROR("Can not open file \"" + new_selected_filename+ "\" for output");

                for (Configuration& cfg : select.selected_cfgs)
                {
                    if (cfg.features["!@$%^&is_new"] == "true" && cfg.size() > 0)
                    {
                        cfg.features.erase("!@$%^&is_new");

                        cfg.features.erase("selected_eqn_inds");
                        int first = *cfg.fitting_items.begin();
                        for (int ind : cfg.fitting_items)
                            cfg.features["selected_eqn_inds"] += ((ind == first) ? "" : ",") + to_string(ind);

                        cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
                        count++;
                    }
                    else
                        cfg.features.erase("!@$%^&is_new");
                }
            }

            MPI_Barrier(mpi.comm);
        }

        int new_count=0;
        MPI_Allreduce(&count, &new_count, 1, MPI_INT, MPI_SUM, mpi.comm);
        if (mpi.rank==0)
            cout << "Training set was increased by " << new_count << " configurations" << endl;

    } END_COMMAND;

    BEGIN_COMMAND("sample",
        "separates and saves the configurations with the high extrapolation grade",
        "mlp sample mlip.mtp in.cfg sampled.cfg [options]:\n"
        "actively samples the configurations from in.cfg and saves them to sampled.cfg\n"
        "  Options:\n"
        "  --threshold=<double>: configurations with grade above this value will be saved to output file, default=2.1\n"
        "  --threshold_break=<double>: grade calculation will break if configuration grade is above this value, default=11\n"
        "  --add_grade_feature=<bool>: adds grade as a feature to configurations, default true"
        "  --log=<string>: log, default=stdout\n"
    ) {
        if (args.size() != 3) {
            std::cout << "\tError: 3 arguments required\n";
            return 1;
        }

        const string mtp_filename = args[0];
        const string input_filename = args[1];
        string output_filename = args[2];

        cout << "MTPR from " << mtp_filename
            << ", input: " << input_filename
            << endl;
        MLMTPR mtpr(mtp_filename);

        Settings settings;
        settings["add_grade_feature"] = "true";
        settings["save_sampled_to"] = output_filename;
        if (settings["log"] != "none")
            settings["log"] = "stdout";
        else
            settings["log"] = "";
        settings.Modify(opts);

        CfgSampling sampler(&mtpr, mtp_filename, settings);

        Message("Starting grades evaluation");

        ifstream ifss(input_filename, ios::binary);
        if (!ifss.is_open())
            ERROR("Can't open file \"" + input_filename + "\" for reading configurations");
        int count = 0;
        for (Configuration cfg; cfg.Load(ifss); count++)
            if (count % mpi.size == mpi.rank)
                sampler.Evaluate(cfg);

        if (mpi.rank == 0)
            Message("Sampling complete");
    } END_COMMAND;

    BEGIN_COMMAND("train",
        "fits an MTP",
        "mlp train potential.mtp train_set.cfg [options]:\n"
        "  trains potential.mtp on the training set from train_set.cfg\n"
        "  Options include:\n"
        "    --save_to=<string>: filename for trained potential. By default equal to the first argument (overwrites input MTP file)\n"
        "    --energy_weight=<double>: weight of energies in the fitting. Default=1\n"
        "    --force_weight=<double>: weight of forces in the fitting. Default=0.01\n"
        "    --stress_weight=<double>: weight of stresses in the fitting. Default=0.001\n"
        "    --weight_scaling=<0 or 1 or 2>: defines how energy and stress weights are scaled with the number of atoms (when fitting to configuration of different size). Default=1.0\n"
        "    --weight_scaling_forces=<0 or 1 or 2>: defines how forces weights are scaled with the number of atoms (when fitting to configuration of different size). Default=0.0\n"
        "        Typical combinations of weight_scaling and weight_scaling_forces: \n"
        "           1) weight_scaling=1, weight_scaling_forces=0 used for fiting the configurations sampled from MD trajectories \n"
        "           2) weight_scaling=0, weight_scaling_forces=0 used for fiting of molecules (non-periodic) \n"
        "           3) weight_scaling=2, weight_scaling_forces=1 used for fiting of structures (periodic) \n"
        "    --scale_by_force=<double>: if >0 wights for the small forces increse. Default=0\n"
        "    --iteration_limit=<int>: maximal number of iterations. Default=1000\n"
        "    --tolerance=<double>: stopping criterion for optimization. Default=1e-8\n"
        "    --init_random=<bool>: Initialize parameters if a not pre-fitted MTP. Default is false - this is when interaction of all species is the same (more accurate fit, but longer optimization)\n"
        "    --skip_preinit=<bool>: skip the 75 iterations done when params are not given\n"
        "    --no_mindist_update=<bool>: if true prevent updating of mindist parameter with actual minimal interatomic distance in the training set. Default=false\n"
        "    --log=<string>: where to write log. --log=none switches logging off. Default=stdout\n"
    ) {
        if (args.size() != 2) 
        {
            cout << "mlp train: 2 arguments are required\n";
            return 1;
        }

        Settings settings;
        settings.Modify(opts);

        if (settings["log"]=="")
            settings["log"]="stdout";
        if (settings["log"]=="none")
            settings["log"]="";
        if (settings["save_to"]=="")
            settings["save_to"]=args[0];

        SetTagLogStream("dev", &std::cout);
        MLMTPR mtp(args[0]);

        bool has_selection = false;
        double sel_ene_wgt = 0.0;
        double sel_frc_wgt = 0.0;
        double sel_str_wgt = 0.0;
        double sel_nbh_wgt = 0.0;
        int sel_wgt_scl = 0;
        
        {
            ifstream ifs(args[0], ios::binary);
            ifs.ignore(HUGE_INT, '#');
            if (ifs.fail() || ifs.eof())
                Message("Selection data was not provided within MTP file");
            else
            {
                Message("Selection data detected within MTP file");

                string tmpstr;
                ifs >> tmpstr;
                if (tmpstr != "MVS_v1.1")
                    ERROR("Invalid MVS-file format");

                ifs >> tmpstr;
                if (tmpstr != "energy_weight")
                    ERROR("Invalid MVS-file format");
                ifs >> sel_ene_wgt;

                ifs >> tmpstr;
                if (tmpstr != "force_weight")
                    ERROR("Invalid MVS-file format");
                ifs >> sel_frc_wgt;

                ifs >> tmpstr;
                if (tmpstr != "stress_weight")
                    ERROR("Invalid MVS-file format");
                ifs >> sel_str_wgt;

                ifs >> tmpstr;
                if (tmpstr != "site_en_weight")
                    ERROR("Invalid MVS-file format");
                ifs >> sel_nbh_wgt;

                ifs >> tmpstr;
                if (tmpstr != "weight_scaling")
                    ERROR("Invalid MVS-file format");
                ifs >> sel_wgt_scl;
            }

            has_selection = true;
        }

#ifdef MLIP_MPI
        vector<Configuration> training_set = MPI_LoadCfgs(args[1]);
#else
        vector<Configuration> training_set = LoadCfgs(args[1]);
#endif
        // training
        MTPR_trainer mtptr(&mtp, settings);
        mtptr.Train(training_set);

        // training errrors calculation
        Settings erm_settings;
        erm_settings["reprt_to"] = settings["log"];
        ErrorMonitor errmon(erm_settings);
        for (Configuration& cfg : training_set)
        {
            Configuration cfg_check(cfg);
            mtp.CalcEFS(cfg_check);
            errmon.AddToCompare(cfg, cfg_check);
        }
        errmon.GetReport();

        {
            Settings select_setup;
            CfgSelection select(&mtp, select_setup);
            if (sel_ene_wgt > 0)
		select.MV_ene_cmpnts_weight = sel_ene_wgt;
            if (sel_frc_wgt > 0)
                select.MV_frc_cmpnts_weight = sel_frc_wgt;
            if (sel_str_wgt > 0)
                select.MV_str_cmpnts_weight = sel_str_wgt;
            if (sel_nbh_wgt > 0)
                select.MV_nbh_cmpnts_weight = sel_nbh_wgt;
            if (sel_wgt_scl > 0)
                select.wgt_scale_power = sel_wgt_scl;

            for (auto& cfg : training_set)
                select.AddForSelection(cfg);

            select.Select();

            select.Save(settings["save_to"]);
        }

        Message("training complete");
    } END_COMMAND;

    BEGIN_COMMAND("select",
        "Acively selects configurations from a pull and adds the selection state data to an MTP",
        "mlp select mtp_filename cfg_filename [options]:\n"
        "Settings can be given in any order.\n"
        "  Options include:\n"
        "  --energy_weight=<double>: set the weight for energy equation, default=1\n"
        "  --force_weight=<double>: set the weight for force equations, default=0\n"
        "  --stress_weight=<double>: set the weight for stress equations, default=0\n"
        "  --site_en_weight=<double>: set the weight for site energy equations, default=0\n"
        "  --weight_scaling=<0 or 1 or 2>: how energy and stress weights are scaled with the number of atoms. final_energy_weight = energy_weight/N^(weight_scaling/2)\n"
        "  --save_to=<string>: were to save mtp with selection state data, equal to the first argument by default\n"
        "  --save_selected_to=<string>: were to save selected configurations, default="" (not saved)\n"
        "  --batch_size=<int>: Configurations are committed for selection by batches. Configuratoions are accumulated in batch until the number of committed configurations reaches this number. After that, selection among all collected configurations is performed. Unlimited if 0 is set"
        "  --log=<int>: Where to write selection log. \"stdout\" and \"stderr\" corresponds to standard output streams; Default="" (none)"
    )
    {
        if (args.size() != 2) 
        {
            cout << "mlp select: 2 argument required\n";
            return 1;
        }

        Settings settings;
        if (!opts["settings_file"].empty())
            settings = LoadSettings(opts["settings_file"]);
        settings.Modify(opts);

        if (settings["save_to"].empty())
            settings["save_to"] = args[0];

        MLMTPR mtp(args[0]);
        CfgSelection selector(&mtp, settings);

#ifdef MLIP_MPI
        auto cfgs = MPI_LoadCfgs(args[1]);
#else 
        auto cfgs = LoadCfgs(args[1]);
#endif

        for (auto& cfg : cfgs)
            cfg.InitNbhs(mtp.CutOff());

        for (auto& cfg : cfgs)
            selector.Process(cfg);

        selector.Select();

    } END_COMMAND;

    BEGIN_COMMAND("check_errors",
        "calculates EFS for configuraion in database and compares them with the provided",
        "mlp check_errors mlip.mtp db.cfg:\n"
        "calculates errors of \"mlip.mtp\" on the database \"db.cfg\"\n"
        "  Options include:\n"
        "  --report_to <string> where to write the integral report. Default=stdout\n"
        "  --log <string> where to write the report for each configuration\n"
    ) {

        if (args.size() < 2) {
            cout << "mlp check_errors: 2 arguments are required\n";
            return 1;
        }

        Settings settings;
        settings.Modify(opts);

        MLMTPR mtp(args[0]);

        ErrorMonitor errmon(settings);

        ifstream ifs(args[1], ios::binary);
        if (!ifs.is_open())
        {
            Message("Can not open a file with configurations \""+args[1]+"\" for reading");
        }
        Configuration cfg;
        int count=0;
        for (; cfg.Load(ifs); count++)
            if (count % mpi.size == mpi.rank)
            {
                Configuration cfg_copy(cfg);
                mtp.CalcEFS(cfg_copy);
                errmon.AddToCompare(cfg, cfg_copy);
            }
        if ((count % mpi.size != 0) && mpi.rank >= (count % mpi.size))
            errmon.AddToCompare(cfg, cfg);

        errmon.GetReport();

    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("calc_grade",
        "calculates and saves maxvol grades of input configurations",
        "mlp calc_grade mlip.mvs in.cfg out.cfg [options]:\n"
        "actively selects from train.cfg, generates state.mvs file from train.cfg, and\n"
        "calculates maxvol grades of configurations located in in.cfg\n"
        "and writes them to out.cfg.\n"
        "  Options:\n"
        "  --threshold=<num>: configurations with grade below this value will not be saved to output file, default=0\n"
        "  --threshold_break=<num>: grade calculation will break if configuration grade is above this value, default=9.9e99 [disabled]\n"
        "  --log=<num>: log, default=stdout\n"
    ) {

        if (args.size() != 3) {
            std::cout << "\tError: 3 arguments required\n";
            return 1;
        }

        const string mtp_filename = args[0];
        const string input_filename = args[1];
        string output_filename = args[2];

        cout << "MTPR from " << mtp_filename
            << ", input: " << input_filename
            << endl;
        MLMTPR mtpr(mtp_filename);

        Settings settings;
        settings["threshold"] = "0";
        settings["threshold_break"] = "0";
        settings["log"] = "stdout";
        settings["add_grade_feature"] = "true";
        settings["save_sampled_to"] = output_filename;
        settings.Modify(opts);

        CfgSampling sampler(&mtpr, mtp_filename, settings);

        Message("Starting grades evaluation");

        ifstream ifss(input_filename, ios::binary);
        if (!ifss.is_open())
            ERROR("Can't open file \"" + input_filename + "\" for reading configurations");

        int count = 0;
        for (Configuration cfg; cfg.Load(ifss); count++)
        {
            cfg.features["ID"] = to_string(count);
            if (count % mpi.size == mpi.rank)
                sampler.Evaluate(cfg);
        }

        ifss.close();
        if (mpi.rank == 0)
            Message("Grade evaluation complete");

    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("make_cell_square",
        "makes the unit cell as square as possible by replicating it",
        "mlp make_cell_square in.cfg out.cfg\n"
        "  Options:\n"
        "  --min_atoms=<int>: minimal number of atoms (default: 32)\n"
        "  --max_atoms=<int>: maximal number of atoms (default: 64)\n"
    ) {
        int min_atoms=32, max_atoms=64;
        if (opts["min_atoms"] != "") {
            try { min_atoms = stoi(opts["min_atoms"]); }
            catch (invalid_argument) {
                cerr << (string)"mlp error: " + opts["min_atoms"] + " is not an integer\n";
            }
        }
        if (opts["max_atoms"] != "") {
            try { max_atoms = stoi(opts["max_atoms"]); }
            catch (invalid_argument) {
                cerr << (string)"mlp error: " + opts["max_atoms"] + " is not an integer\n";
            }
        }

        Configuration cfg;
        ifstream ifs(args[0], ios::binary);
        ofstream ofs(args[1], ios::binary);
        while(cfg.Load(ifs))
        {
            int cfg_size = cfg.size();
            const int M = 4;
            Matrix3 A, B;
            Matrix3 L = cfg.lattice * cfg.lattice.transpose();
            Matrix3 X;
            double score = 1e99;
            for (A[0][0] = -M; A[0][0] <= M; A[0][0]++)
                for (A[0][1] = -M; A[0][1] <= M; A[0][1]++)
                    for (A[0][2] = -M; A[0][2] <= M; A[0][2]++)
                        for (A[1][0] = -M; A[1][0] <= M; A[1][0]++)
                            for (A[1][1] = -M; A[1][1] <= M; A[1][1]++)
                                for (A[1][2] = -M; A[1][2] <= M; A[1][2]++)
                                    for (A[2][0] = -M; A[2][0] <= M; A[2][0]++)
                                        for (A[2][1] = -M; A[2][1] <= M; A[2][1]++)
                                            for (A[2][2] = -M; A[2][2] <= M; A[2][2]++) {
                                                if (fabs(A.det()) * cfg_size < min_atoms) continue;
                                                if (fabs(A.det()) * cfg_size > max_atoms) continue;
                                                X = A*L*A.transpose();
                                                double shift = (X[0][0] + X[1][1] + X[2][2]) / 3;
                                                X[0][0] -= shift;
                                                X[1][1] -= shift;
                                                X[2][2] -= shift;
                                                double curr_score = X.NormFrobeniusSq();
                                                if (curr_score < score) {
                                                    score = curr_score;
                                                    B = A;
                                                }
                                            }
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++)
                    cout << (int)B[a][b] << " ";
                cout << "\n";
            }
            cout << B.det() << "\n";
            cfg.ReplicateUnitCell(templateMatrix3<int>((int)B[0][0], (int)B[0][1], (int)B[0][2], (int)B[1][0], (int)B[1][1], (int)B[1][2], (int)B[2][0], (int)B[2][1], (int)B[2][2]));
            cfg.CorrectSupercell();
            cfg.Save(ofs);
        }
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("replicate_cell",
        "replicates unit cell",
        "mlp replicate_cell in.cfg out.cfg nx ny nz:\n"
        "or"
        "mlp replicate_cell in.cfg out.cfg nxx nxy nxz nyx nyy nyz nzx nzy nzz:\n"
        "Replicate the supercells by nx*ny*nz.\n"
        "Alternatively, applies the 3x3 integer matrix to the lattice\n"
    ) {
        if (args.size() == 5) {
            int nx, ny, nz;
            try { nx = stoi(args[2]); }
            catch (invalid_argument) {
                cerr << (string)"mlp error: " + args[2] + " is not an integer\n";
            }
            try { ny = stoi(args[3]); }
            catch (invalid_argument) {
                cerr << (string)"mlp error: " + args[3] + " is not an integer\n";
            }
            try { nz = stoi(args[4]); }
            catch (invalid_argument) {
                cerr << (string)"mlp error: " + args[4] + " is not an integer\n";
            }

            Configuration cfg;
            ifstream ifs(args[0], ios::binary);
            ofstream ofs(args[1], ios::binary);
            while (cfg.Load(ifs)) {
                cfg.ReplicateUnitCell(nx, ny, nz);
                cfg.Save(ofs);
            }
        } else if (args.size() == 11) {
            int args_int[3][3];
            for (int i = 0; i < 9; i++) {
                try { args_int[i/3][i%3] = stoi(args[i+2]); }
                catch (invalid_argument) {
                    cerr << (string)"mlp error: " + args[i+2] + " is not an integer\n";
                }
            }
            templateMatrix3<int> A(args_int);

            Configuration cfg;
            ifstream ifs(args[0], ios::binary);
            ofstream ofs(args[1], ios::binary);
            while (cfg.Load(ifs)) {
                cfg.ReplicateUnitCell(A);
                cfg.Save(ofs);
            }
        }
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("mindist_filter",
        "removes the configuration with too short interatomic distance",
        "mlp mindist_filter input.cfg 2.1 output.cfg:\n"
    ) {
        if (args.size() != 3) {
            cout << "3 arguments are required\n";
            return 1;
        }

        {    ofstream ofs(args[2]);    }

        auto cfgs = MPI_LoadCfgs(args[0]);
        double md = stod(args[1]);
        for (int rnk=0; rnk<mpi.size; rnk++)
        {
            MPI_Barrier(mpi.comm);
            if (rnk == mpi.rank)
                for (auto cfg : cfgs)
                    if (cfg.MinDist() >= md)
                        cfg.AppendToFile(args[2]);
        }
    } END_COMMAND;

    BEGIN_COMMAND("convert_cfg",
        "convert a file with configurations from one format to another",
        "mlp convert_cfg [options] inputfilename outputfilename\n"
        "  input filename sould always be entered before output filename\n"
        "  Options can be given in any order. Options include:\n"
        "  --input_format=<format>: format of the input file\n"
        "  --output_format=<format>: format of the output file\n"
        "  --input_chgcar=<filename>: name of the chgcar file\n"
        "  --input_near_neigh=<filename>: name of the file with nearest neighbor distances\n"
        "  --append:      opens output file in append mode\n"
        "  --last:        ignores all configurations in the input file except the last one\n"
        "                 (useful with relaxation)\n"
        "  --fix_lattice: creates an equivalent configuration by moving the atoms into\n"
        "                 the supercell (if they are outside) and tries to make the\n"
        "                 lattice as much orthogonal as possible and lower triangular\n"
		"  --save_nonconverged: writes configurations with nonconverged VASP calculations, otherwise they are ignored\n"
        "  --absolute_elements: writes absolute atomic numbers into the cfg file instead of 0,1,2,.... Disabled by default, used only while reading from OUTCAR\n"
        "  --elements_order=18,22,46,... atomic numbers separated with commas in the order as they are in POTCAR. Used only while writing to POSCAR\n"
        "  --no_forces_stresses: no saving of forces and stresses into the cfg file\n"  
        "\n"
        "  <format> can be:\n"
        "  txt (default): mlip textual format\n"
        "  bin:           mlip binary format\n"
        "  vasp_outcar:   only as input; VASP versions 5.3.5 and 5.4.1 were tested\n"
        "  vasp_poscar:   only as output. When writing multiple configurations,\n"
        "                 POSCAR0, POSCAR1, etc. are created\n"
        "  lammps_dump: only as input. Only lattice, atomic positions and types are saved. Only cartesian coordinates are processed.\n"
        "  lammps_datafile: only as output. Can be read by read_data from lammps.\n"
        "                 Multiple configurations are saved to several files.\n"
        
        ) {

        if (opts["input_format"] == "") opts["input_format"] = "txt";
        if (opts["output_format"] == "") opts["output_format"] = "txt";

        if (opts["output_format"] == "vasp_outcar" || opts["output_format"] == "lammps_dump")
            ERROR("Format " + opts["output_format"] + " is not allowed for output");
        if (opts["input_format"] == "vasp_poscar" || opts["input_format"] == "lammps_datafile")
            ERROR("Format " + opts["input_format"] + " is not allowed for input");

        const bool possibly_many_output_files = (opts["output_format"] == "vasp_poscar")
            || (opts["output_format"] == "lammps_datafile");

        ifstream ifs(args[0], ios::binary);
        if (ifs.fail()) ERROR("cannot open " + args[0] + " for reading");

        ofstream ofs;
        if (!possibly_many_output_files) {
            if (opts["append"] == "") {
                ofs.open(args[1], ios::binary);
                if (ofs.fail()) ERROR("cannot open " + args[1] + " for writing");
            } else {
                ofs.open(args[1], ios::binary | ios::app);
                if (ofs.fail()) ERROR("cannot open " + args[1] + " for appending");
            }
        }

        vector<Configuration> db;
        vector<int> db_species;
        vector<int> types_mapping;
        int count = 0;
        bool read_success = true;

        // block that reads configurations from VASP OUTCAR into the database
        if (opts["input_format"] == "vasp_outcar") {
            ifs.close();
	    Configuration::LoadDynamicsFromOUTCAR(db, args[0],opts["save_nonconverged"] != "",opts["absolute_elements"] == "");
            //to ensure the unified way of POSCAR writing    
            for (int i=0;i<200;i++)
                types_mapping.push_back(i);

            //block that creates magnetic moments
            if (opts["input_chgcar"] != "" && opts["input_near_neigh"] != "") {
                ifstream ifs_nnd(opts["input_near_neigh"]);
                int n_species;
                ifs_nnd >> n_species;

                Array1D nearest_neigh_dist(n_species);
                for (int i = 0; i < n_species; i++) {
                    ifs_nnd >> nearest_neigh_dist[i];
                    nearest_neigh_dist[i] /= 4;
                }

                MagneticMoments MagMom(false, nearest_neigh_dist);
                MagMom.LoadDensityFromCHGCAR(opts["input_chgcar"], n_species);
                MagMom.CalculateWeights(db[0]);
                MagMom.Calculate(db[0]);
                if (opts["no_forces_stresses"] != "") {
                    db[0].has_forces(false);
                    db[0].has_stresses(false);
                }
            }
        }

        //block that reads configurations from LAMMPS dump file into the database
        bool is_eof_lammps_file = true;
                if ((opts["input_format"] == "lammps_dump") && (opts["output_format"] != "vasp_poscar")) {
            while (is_eof_lammps_file) {
                Configuration cfg;
                is_eof_lammps_file = cfg.LoadNextFromLAMMPSDump(ifs);
                if (is_eof_lammps_file) db.push_back(cfg);
            }
        }  

        Configuration cfg_last; // used to differentiate between single-configuration and multiple-configuration inputs
            
        //scanning the entire database for all atomic numbers present
        if ((opts["output_format"] == "vasp_poscar") && ((opts["input_format"] == "txt") || (opts["input_format"] == "bin") || (opts["input_format"] == "") || (opts["input_format"] == "lammps_dump")))
        {    
            while (read_success) {
            Configuration cfg;
            
            // Read cfg
            if (db_species.size() == 0) {
                if (opts["input_format"] == "txt" || opts["input_format"] == "bin") {
                    read_success = cfg.Load(ifs);
                }
                else if (opts["input_format"] == "lammps_dump") {
                    read_success = cfg.LoadNextFromLAMMPSDump(ifs);
                    if (read_success) db.push_back(cfg);
                }
                else {
                    ERROR("unknown file format");
                }
            } else {
                if (opts["input_format"] == "txt" || opts["input_format"] == "bin") {
                    read_success = cfg.Load(ifs);
                }
                else if (opts["input_format"] == "lammps_dump") {
                    read_success = cfg.LoadNextFromLAMMPSDump(ifs);
                    if (read_success) db.push_back(cfg);
                }
            }

            for (int i=0; i<cfg.size(); i++)
                if (std::find(db_species.begin(), db_species.end(), cfg.type(i)) == db_species.end())
                    db_species.push_back(cfg.type(i));
                }

            //if (std::distance(db_species.begin(), std::max_element(db_species.begin(), db_species.end()))==db_species[db_species.size() - 1])
            bool relative=true;
            for (int i=0; i<db_species.size(); i++)
                if (std::find(db_species.begin(), db_species.end(), i) == db_species.end())
                    relative=false;
            if (relative)
            {
                cout << "relative numeration of species detected" << endl;
                for (int i=0;i<db_species.size();i++)
                    types_mapping.push_back(i);

            }
            else
            {
            cout << "absolute numeration of species detected" << endl;
            if (opts["elements_order"] == "")
                ERROR("Absolute numeration of atomic species in cfg file detected. Use [--elements_order] option, see [mlp help convert_cfg] for details");

            string t = opts["elements_order"];
            int k = 0;
            while (k<t.size())
            {
                string type ="";
                while ((t[k]!=',')&&(k<t.size()))
                {
                    type+=t[k++];
                }
                types_mapping.push_back(std::stoi(type));
                k+=1;
            }

            }
            
            for (int i = 0; i < db_species.size(); i++)
            {
                bool found = false;
                for (int j=0; j < types_mapping.size(); j++)
                    if (db_species[i]==types_mapping[j])
                        found=true;

                if (!found)
                    ERROR("Not all the species were mentioned in [--elements_order] option: " + std::to_string(db_species[i]) + " atomic number is missing.");
            }
        ifs.close();
        ifs.open(args[0], ios::binary);
        }
        
        read_success = true;
        

        while (read_success) {
            Configuration cfg;

            // Read cfg
            if (db.size() == 0) {
                if (opts["input_format"] == "txt" || opts["input_format"] == "bin")
                    read_success = cfg.Load(ifs);
                else
					{
					  Warning("no confgiurations will be saved");
					  read_success=false;
					}
                    //ERROR("unknown file format");
            } else {
                read_success = (count < db.size());
                if (read_success)
                    cfg = db[count];
            }
            if (!read_success) break;

            if (opts["fix_lattice"] != "") cfg.CorrectSupercell();

            // Save cfg
            if (opts["last"] == "") {
                if (opts["output_format"] == "txt")
                    cfg.Save(ofs);
                if (opts["output_format"] == "bin")
                    cfg.SaveBin(ofs);
                if (opts["output_format"] == "vasp_poscar") {
                    if (count != 0) {
                        if (count == 1) cfg_last.WriteVaspPOSCAR(args[1] + "0",types_mapping);
                        cfg.WriteVaspPOSCAR(args[1] + to_string(count),types_mapping);
                    }
                }
                if (opts["output_format"] == "lammps_datafile") {
                    if (count != 0) {
                        if (count == 1) cfg_last.WriteLammpsDatafile(args[1] + "0");
                        cfg.WriteLammpsDatafile(args[1] + to_string(count));
                    }
                }
            }

            cfg_last = cfg;
            count++;
        }
        if (opts["last"] == "") {
            if (opts["output_format"] == "vasp_poscar" && count == 1)
                cfg_last.WriteVaspPOSCAR(args[1],types_mapping);
            if (opts["output_format"] == "lammps_datafile" && count == 1)
                cfg_last.WriteLammpsDatafile(args[1]);
        } else {
            // --last option given
            if (opts["output_format"] == "txt")
                cfg_last.Save(ofs);
            if (opts["output_format"] == "bin")
                cfg_last.SaveBin(ofs);
            if (opts["output_format"] == "vasp_poscar")
                cfg_last.WriteVaspPOSCAR(args[1],types_mapping);
            if (opts["output_format"] == "lammps_datafile")
                cfg_last.WriteLammpsDatafile(args[1]);
        }

        std::cout << "Processed " << count << " configurations" << std::endl;
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("add_from_outcar",
        "reads one configuration from the OUTCAR(s)",
        "mlp add_from_outcar db outcar [outcar2 outcar3 ...]:\n"
        "reads one configuration from each of the outcars\n"
        "and APPENDs to the database db\n"
        ) {

        if (args.size() < 2) {
            cout << "mlp add_from_outcar: minimum 2 arguments required\n";
            return 1;
        }

        ofstream ofs(args[0], std::ios::app|std::ios::binary);
        Configuration cfg;
        for (int i = 1; i < args.size(); i++) {
            cfg.LoadFromOUTCAR(args[i],opts["absolute_elements"] == "");

			if (opts["save_nonconverged"] != "")
				cfg.Save(ofs);
			else
				if (cfg.features["EFS_by"] != "VASP_not_converged")
            		cfg.Save(ofs);
				else
					Warning("non-converged OUTCAR detected. Use '--save_noncoverged' option to save it");
        }
        ofs.close();
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("add_from_outcar_dyn",
        "reads dynamics from the OUTCAR(s)",
        "mlp add_from_outcar_dyn db outcar [outcar2 outcar3 ...]:\n"
        "reads dynamics configuration from each of the outcars\n"
        "each configuration from the dynamics is APPENDed to the database\n"
        ) {

        if (args.size() < 2) {
            cout << "mlp add_from_outcar_dyn: minimum 2 arguments required\n";
            return 1;
        }

        ofstream ofs(args[0], ios::app|ios::binary);
        for (int i = 1; i < args.size(); i++) {
            vector<Configuration> db;
            if (!Configuration::LoadDynamicsFromOUTCAR(db, args[i],opts["save_nonconverged"] != "",opts["absolute_elements"] == ""))
                cerr << "Warning: OUTCAR is broken" << endl;
            for (Configuration cfg : db)
                cfg.Save(ofs);
        }
        ofs.close();
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("add_from_outcar_last",
        "reads the dynamics from OUTCAR(s) and gets the last configuration from each OUTCAR",
        "mlp add_from_outcar_last db outcar [outcar2 outcar3 ...]:\n"
        "reads the dynamics from OUTCAR(s) and gets the last configuration from each OUTCAR.\n"
        "APPENDs all last configurations to the database db\n"
        ) {

        if (args.size() < 2) {
            std::cout << "mlp add_from_outcar: minimum 2 arguments required\n";
            return 1;
        }

        ofstream ofs(args[0], ios::app|ios::binary);
        Configuration cfg;
        for (int i = 1; i < args.size(); i++) {
            if (!cfg.LoadLastFromOUTCAR(args[i],opts["absolute_elements"] == ""))
                cerr << "Warning: OUTCAR is broken" << endl;
            cfg.Save(ofs);
        }
        ofs.close();
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("to_bin",
        "converts a database to the binary format",
        "mlp to_bin db.cfgs db.bin.cfgs:\n"
        "Reads the database db.cfgs and saves in a binary format to db.bin.cfgs\n"
        ) {

        if (args.size() < 3) {
            cout << "mlp to_bin: minimum 2 arguments required\n";
            return 1;
        }

        ifstream ifs(args[0], ios::binary);
        ofstream ofs(args[2], ios::binary);
        Configuration cfg;
        int count;
        for (count = 0; cfg.Load(ifs); count++)
            cfg.SaveBin(ofs);
        ofs.close();
        cout << count << " configurations processed.\n";
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("from_bin",
        "converts a database from the binary format",
        "mlp from_bin db.bin.cfgs db.cfgs:\n"
        "Reads the binary database db.bin.cfgs and saves to db.cfgs\n"
        ) {

        if (args.size() < 3) {
            cout << "mlp to_bin: minimum 2 arguments required\n";
            return 1;
        }

        ifstream ifs(args[0], ios::binary);
        ofstream ofs(args[2], ios::binary);
        Configuration cfg;
        int count;

        for (count = 0; cfg.Load(ifs); count++)
            cfg.Save(ofs);

        ofs.close();
        cout << count << " configurations processed.\n";
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("fix_lattice",
        "puts the atoms into supercell (if they are outside) and makes lattice as much orthogonal as possible and lower triangular",
        "mlp fix_lattice in.cfg out.cfg:\n"
        ) {

        if (args.size() < 2) {
            cout << "mlp to_bin: minimum 2 arguments required\n";
            return 1;
        }

        ifstream ifs(args[0], ios::binary);
        ofstream ofs(args[1], ios::binary);
        Configuration cfg;
        int count;
        for (count = 0; cfg.Load(ifs); count++) {
            cfg.CorrectSupercell();
            cfg.Save(ofs);
        }
        ofs.close();
        cout << count << " configurations processed.\n";
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("mindist",
        "reads a cfg file and saves it with added mindist feature",
        "mlp mindist db.cfg [Options]\n"
        "  Options can be given in any order. Options include:\n"
        "  --no_overwrite: do not change db.cfg, only print global mindist\n"
        "  --no_types: add to file mindist information for each type of atoms"
        ) 
    {

        if (args.size() != 1) {
            cout << "mlp mindist: 1 arguments required\n";
            return 1;
        }

        auto cfgs = MPI_LoadCfgs(args[0]);
        bool no_overwrite = opts.count("no_overwrite") > 0;
        bool no_types = opts.count("no_types") > 0;

        if (!no_overwrite) { ofstream ofs(args[0]); }

        map<pair<int, int>, double> global_mindists;
        double global_md = HUGE_DOUBLE;
        for (auto& cfg : cfgs)
        {
            auto typeset = cfg.GetTypes();
            auto mdm = cfg.GetTypeMindists();
            
            double local_md = HUGE_DOUBLE;
            for (int t1 : typeset)
                for (int t2 : typeset)
                    local_md = __min(local_md, mdm[make_pair(t1, t2)]);
            global_md = __min(local_md, global_md);

            for (auto& md : mdm)
                if (global_mindists.find(md.first) != global_mindists.end())
                    global_mindists[md.first] = __min(md.second, global_mindists[md.first]);
                else
                    global_mindists[md.first] = md.second;

            if (!no_overwrite)
            {
                if (!no_types)
                {
                    string str = "";
                    for (auto& md : mdm)
                        if (md.first.first <= md.first.second)
                            str += "md(" + to_string(md.first.first) + "," + to_string(md.first.second) + ")=" + to_string(md.second) + ";";
                    str = str.substr(0, str.length()-1);
                    cfg.features["mindist"] = str;
                    cfg.AppendToFile(args[0]);
                }
                else
                {
                    cfg.features["mindist"] = to_string(local_md);
                    cfg.AppendToFile(args[0]);
                }
            }
        }

        cout << "Global mindist: " << global_md << endl;
        for (auto& md : global_mindists)
            if (md.first.first <= md.first.second)
                cout << "mindist(" << md.first.first << ", " << md.first.second << ") = " << md.second << endl;

    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("fix_mindist",
        "fixes too short minimal interatomic distance in all configurations from a file by relaxation with repulsive pair potential",
        "mlp fix_mindist 1.6 input.cfg output.cfg\n"
        "the first argument is new minimal allowed interatomic distance\n"
    ) {

        if (args.size() != 3) ERROR("mlp fix_mindist: 3 arguments required\n");

        Settings settings;
        settings["init_mindist"] = args[0];
        settings["iteration_limit"] = "0";
        if (!opts["log"].empty())
            settings["log"] = opts["log"];

        Relaxation rlx(settings);

        ifstream ifs(args[1], ios::binary);
#ifndef MLIP_MPI
        ofstream ofs(args[2], ios::binary);
#else 
        ofstream ofs(args[2]+'_'+to_string(mpi.rank), ios::binary);
#endif

        if (!ifs.is_open()) ERROR("input file is not open");
        if (!ofs.is_open()) ERROR("output file is not open");

        Configuration cfg;
        for (int cntr=1; cfg.Load(ifs); cntr++)
        {
            if ((cntr-1) % mpi.size != mpi.rank)
                continue;

            rlx.cfg = cfg;

            try { rlx.Run(); }
            catch (MlipException& e)
            {
                Warning("Relaxation failed: " + e.message);
            }

            rlx.cfg.Save(ofs);

            if (mpi.rank == 0)
                cout << cntr << endl;
        }
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("subsample",
        "subsamples a database (reduces size by skiping every N configurations)",
        "mlp subsample db_in.cfg db_out.cfg skipN\n"
        ) {

        if (args.size() != 3) {
            cout << "mlp subsample: 3 arguments required\n";
            return 1;
        }

        ifstream ifs(args[0], ios::binary);
        ofstream ofs(args[1], ios::binary);
        Configuration cfg;
        int skipN = stoi(args[2]);
        for (int i = 0; cfg.Load(ifs); i++) {
            if (i % skipN == 0)
                cfg.Save(ofs);
        }
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("fix_cfg",
        "Fixes a database by reading and writing it again",
        "mlp fix_db db.cfg [options]\n"
        "  Options:\n"
        "  --no_loss: writes a file with a very high precision\n"
    ) {
        unsigned int flag = 0U;
        if (opts["no_loss"] != "")
            flag |= Configuration::SAVE_NO_LOSS;

        std::vector<Configuration> db;
        {
            Configuration cfg;
            std::ifstream ifs(args[0], std::ios::binary);
            while (cfg.Load(ifs)) {
                    db.push_back(cfg);
            }
            ifs.close();
        }
        {
            Configuration cfg;
            std::ofstream ofs(args[0], std::ios::binary);
            for (int i = 0; i < db.size(); i++) {
                db[i].Save(ofs, flag);
            }
            ofs.close();
        }
    }
    END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("filter_nonconv",
        "removes configurations with feature[EFS_by] containing not_converged",
        "mlp filter_nonconv db.cfg\n"
        ) {
        if (args.size() != 1) {
        std::cerr << "Error: 1 argument required" << std::endl;
        std::cerr << "Usage: mlp filter_nonconv db.cfg" << std::endl;
        return 1;
    }
    std::vector<Configuration> db;
    {
        Configuration cfg;
        std::ifstream ifs(args[0], std::ios::binary);
        while (cfg.Load(ifs)) {
            if (cfg.features["EFS_by"].find("not_converged") == std::string::npos)
                db.push_back(cfg);
        }
        ifs.close();
    }
    {
        Configuration cfg;
        std::ofstream ofs(args[0], std::ios::binary);
        for (int i = 0; i < db.size(); i++) {
            db[i].Save(ofs);
        }
        ofs.close();
    }
    } END_COMMAND;

    BEGIN_COMMAND("test",
        "performs a number of unit tests",
        "mlp test\n"
    ) {
        if(!self_test()) exit(1);
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("remap_species",
        "changes species numbering",
        "mlp remap_species in.cfg out.cfg n0 n1 ...\n"
    ) {
        if (args.size() <= 3) {
            std::cout << "\tError: at least 3 arguments required\n";
            return 1;
        }

        vector<int> new_species;
        for(int i=2; i < args.size(); i++)
            new_species.push_back(std::stoi(args[i]));

        ifstream ifs(args[0], std::ios::binary);
        ofstream ofs(args[1], std::ios::binary);
        Configuration cfg;
        while (cfg.Load(ifs)) {
            for(int i=0; i<cfg.size(); i++)
                cfg.type(i) = new_species[cfg.type(i)];
            cfg.Save(ofs);
        }
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("select_add_nbhs",
        "selects among neighborhoods",
        "mlp select_add_nbhs pot.mtp train.cfg new.cfg diff.cfg:\n"
    ) {
        if (args.size() != 4) {
            std::cout << "\tError: 4 arguments required\n";
            return 1;
        }

        Message("Initialization");

        const string mtp_filename = args[0];
        const string train_filename = args[1];   //no need since we use state.mvs
        const string new_cfg_filename = args[2];
        const string diff_filename = args[3];

        MLMTPR mtp(mtp_filename);

        if (mpi.rank == 0) {
            cout << "MTPR from " << mtp_filename
                << ", train set: " << train_filename
                << ", add from set: " << new_cfg_filename
                << endl;
        }

        Settings settings;
        settings["update_active_set"]="TRUE";
        settings["site_en_weight"] = "1";
        settings["energy_weight"] = "0";
        settings["force_weight"] = "0";
        settings["stress_weight"] = "0";
        settings["threshold"] = opts["select_threshold"];
        settings["save_state_to"] = opts["save_state_to"];
        settings["save_selected_to"] = opts["save_selected_to"];
        settings["batch_size"] = "99999";
        settings["site_en_weight"] = "1";
        settings["treat_locals"] = "FALSE";

        CfgSelection selector(&mtp, settings);

        Message("Loading trained");
        vector<Configuration> cfgs=MPI_LoadCfgs(train_filename);
        for (Configuration& cfg : cfgs)
        {
            if (cfg.pos(0) != Vector3(0, 0, 0))
                Warning("The first atom is not in (0, 0, 0) position");
            cfg.InitNbhs(mtp.CutOff());
            cfg.nbhs.resize(1);
            cfg.CommGhostData = nullptr;
            selector.Process(cfg);
        }

        Message("Loading new");
        cfgs=MPI_LoadCfgs(new_cfg_filename);
        for (Configuration& cfg : cfgs)
        {
            if (cfg.pos(0) != Vector3(0, 0, 0))
                Warning("The first atom is not in (0, 0, 0) position");
            cfg.InitNbhs(mtp.CutOff());
            cfg.nbhs.resize(1);
            cfg.CommGhostData = nullptr;
            cfg.features["*"] = "*";
            selector.Process(cfg);
        }

        Message("Selecting");
        selector.Select();

        Message("Saving selected");
        { if (mpi.rank == 0) ofstream ofs(diff_filename); }
        for (int i=0; i<mpi.size; i++)
        {
            MPI_Barrier(mpi.comm);
            for (Configuration& cfg : selector.selected_cfgs)
                if (cfg.features.count("*") == 1)
                {
                    cfg.features.erase("*");
                    cfg.AppendToFile(diff_filename);
                }
        }
    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("extract_nbhs",
        "Extracts specified neighborhoods from configurations in the input file, makes a configurations from them, and saves to output file",
        "mlp extract_nbhs in_filename.cfg out_filename cutoff:\n"
    ) {
        if (args.size() != 3) {
            std::cout << "\tError: 3 arguments required\n";
            return 1;
        }

        Message("Initialization");

        const string inp_filename = args[0];
        const string out_filename = args[1];   
        const double cutoff = stod(args[2]);

        if (opts.count("append") == 0)
        { ofstream ofs(out_filename); }

        auto cfgs = MPI_LoadCfgs(inp_filename);
        vector<Configuration> nbhcfgs;

        for (auto& cfg : cfgs)
        {
            std::set<int> inds;
            if (cfg.features.count("nbhs_to_extract") == 0)
                continue;
            string strind = cfg.features.at("nbhs_to_extract");
            while (!strind.empty())
            {
                size_t commapos = strind.find(',');

                int ind = stoi(strind.substr(0, commapos));
                if (ind > cfg.size())
                    ERROR("incorrect nbh index detected!");
                else 
                    cout << "nbh ind #" << ind << endl;

                inds.insert(ind);

                if (commapos != -1)
                    strind = strind.substr(commapos+1);
                else
                    strind.clear();
            }
            for (int ind : inds)
            {
                nbhcfgs.push_back(cfg.GetCfgFromLocalEnv(ind-1, cutoff));
                if (cfg.has_site_energies())
                {
                    nbhcfgs.back().has_energy(true);
                    nbhcfgs.back().energy = cfg.site_energy(ind-1);
                }
            }
        }
        for (int i=0; i<mpi.size; i++)
        {
            if (i == mpi.rank)
            {
                ofstream ofs(args[1], ios::binary | ios::app);
                for (auto& cfg : nbhcfgs)
                {
                    //cfg.features.erase("ghost_atom_numbers");
                    //cfg.ghost_inds.clear();
                    cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
                }
            }
            MPI_Barrier(mpi.comm);
        }

    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("extract-nbhs",
        "Extracts specified neighborhoods from configurations in the input file, makes a configurations from them, and saves to output file",
        "mlp extract-nbhs in_filename.cfg out_filename cutoff:\n"
    ) {
        if (args.size() != 3) {
            std::cout << "\tError: 3 arguments required\n";
            return 1;
        }

        Message("Initialization");

        const string inp_filename = args[0];
        const string out_filename = args[1];   
        const double cutoff = stod(args[2]);

        { ofstream ofs(out_filename); }

        auto cfgs = MPI_LoadCfgs(inp_filename);
        vector<Configuration> nbhcfgs;

        for (auto& cfg : cfgs)
        {
            std::set<int> inds;
            if (cfg.features.count("selected_eqn_inds") == 0)
                continue;
            string strind = cfg.features.at("selected_eqn_inds");
            while (!strind.empty())
            {
                size_t commapos = strind.find(',');
                if (commapos != -1)
                {
                    int ind = stoi(strind.substr(0, commapos));
                    if (ind >= 1+4*(cfg.size()-(int)cfg.ghost_inds.size())+9)
                        ERROR("incorrect equation index detected!");
                    else if (ind >= 1+3*(cfg.size()-(int)cfg.ghost_inds.size())+9)
                    {
                        ind = ind - (1+3*(cfg.size()-(int)cfg.ghost_inds.size())+9);
                        cout << "nbh ind" << endl;
                    }
                    else if (ind >= 1+3*(cfg.size()-(int)cfg.ghost_inds.size()))
                        ERROR("stress equation index detected!");
                    else if (ind > 0)
                    {
                        ind = (ind-1) / 3;
                        cout << "frc ind" << endl;
                    }
                    else 
                        ERROR("energy equation index detected!");                          
                    inds.insert(ind);
                    strind = strind.substr(commapos+1);
                }
                else
                {
                    int ind = stoi(strind);
                    if (ind >= 1+4*(cfg.size()-(int)cfg.ghost_inds.size())+9)
                        ERROR("incorrect equation index detected!");
                    else if (ind >= 1+3*(cfg.size()-(int)cfg.ghost_inds.size())+9)
                    {
                        ind = ind - (1+3*(cfg.size()-(int)cfg.ghost_inds.size())+9);
                        cout << "nbh ind" << endl;
                    }
                    else if (ind >= 1+3*(cfg.size()-(int)cfg.ghost_inds.size()))
                        ERROR("stress equation index detected!");
                    else if (ind > 0)
                    {
                        ind = (ind-1) / 3;
                        cout << "frc ind" << endl;
                    }
                    else
                        ERROR("energy equation index detected!");
                    inds.insert(ind);
                    strind.clear();
                }
            }
            for (int ind : inds)
                nbhcfgs.push_back(cfg.GetCfgFromLocalEnv(ind, cutoff));
        }
        for (int i=0; i<mpi.size; i++)
        {
            if (i == mpi.rank)
            {
                ofstream ofs(args[1], ios::binary | ios::app);
                for (auto& cfg : nbhcfgs)
                    cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
            }
            MPI_Barrier(mpi.comm);
        }

    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("invert_stress",
        "xxx",
        "mlp invert_stress input.cfg:\n"
    ) {

        if (args.size() != 1) {
            std::cout << "\tError: 1 arguments required\n";
            return 1;
        }

        const string cfg_filename = args[0];

        auto cfgs = LoadCfgs(cfg_filename);
        
        { ofstream ofs(cfg_filename); }

        int counter=0;
        for (auto cfg : cfgs)
        {
            cfg.stresses *= -1;
            cfg.AppendToFile(cfg_filename);
            cout << ++counter << endl;
        }

    } END_COMMAND;

    BEGIN_UNDOCUMENTED_COMMAND("recognize",
        "XXX",
        "mlp recognize MTP100.mtp input.cfg --train=out.mvs:\n"
    ) 
    {
        if (args.size() != 2) {
            std::cout << "\tError: 2 arguments required\n";
            return 1;
        }

        const string mtp_filename = args[0];
        const string cfg_filename = args[1];

        MTP mtp(mtp_filename);

        if (!opts["train"].empty())
        {
            Settings set;
            set["site_en_weight"] = "1";
            set["energy_weight"] = "0";
            set["force_weight"] = "0";
            set["stress_weight"] = "0";
            set["weight_scaling"] = "0";
                
            CfgSelection selector(&mtp, set);

            selector.LoadSelected(args[1]);

            selector.Save(opts["train"]);
        }
        else
        {
            auto cfgs = MPI_LoadCfgs(args[1]);

            for (auto& cfg : cfgs)
                for (int i=0; i< cfg.size(); i++)
                    cfg.type(i) = 0;
            
            Settings set;
            set["threshold"] = "1.";
            set["threshold_break"] = "0";
            set["save_sampled_to"] = "";
            set["add_grade_feature"] = "true";
            set["log"] = "stdout";

            CfgSampling samp(&mtp, mtp_filename, set);

            ofstream ofs("recognition.txt");

            int cntr = 0;
            for (auto& cfg : cfgs)
            {
                samp.Grade(cfg);

                // find two atoms with maximal grades
                int i1=-1, i2=-1;
                double g1=-1, g2=-1;
                for (int i=0; i<cfg.size(); i++)
                    if (cfg.nbh_grades(i) > g1)
                    {
                        i1 = i;
                        g1 = cfg.nbh_grades(i);
                    }
                for (int i=0; i<cfg.size(); i++)
                    if (cfg.nbh_grades(i) > g2 && i != i1)
                    {
                        i2 = i;
                        g2 = cfg.nbh_grades(i);
                    }

                //
                bool high_grades = false;
                for (int i=0; i<cfg.size(); i++)
                    if (cfg.nbh_grades(i) > g2/5.0 && i!=i1 && i!=i2)
                        high_grades = true;
                bool low_grades = false;
                if (g2 < 11.0)
                    low_grades = true;

                //
                Vector3 l1(cfg.lattice[0][0], cfg.lattice[0][1], cfg.lattice[0][2]);
                Vector3 l2(cfg.lattice[1][0], cfg.lattice[1][1], cfg.lattice[1][2]);
                Vector3 l3(cfg.lattice[2][0], cfg.lattice[2][1], cfg.lattice[2][2]);
                bool is_lattice_ortog = (fabs(l1*l2) + fabs(l2*l3) + fabs(l1*l3) < 0.1);
                double a = (l1.Norm() + l2.Norm() + l3.Norm()) / 3/4;
                //cout << " lat_dist=" << a << ' ';
                bool is_lattice_cubic = ((l1-l2).Norm() + (l2-l3).Norm() + (l1-l3).Norm() < 0.1*a);
                //
                Vector3 d12 = cfg.pos(i1) - cfg.pos(i2);
                bool wrong_orientation = false;
                bool wrong_dist = false;
                if (d12.Norm() < 0.85*1.45*sqrt(2)*a/3 || d12.Norm() > 1.15*1.45*sqrt(2)*a/3)
                    wrong_dist = true;
                //cout << ' ' << d12.Norm() << ' ' << sqrt(2)*a/3 << ' ' << d12.Norm()/(sqrt(2)*a/3) << ' ';
                if (fabs(d12*Vector3(1,1,0)) < 0.9)
                    wrong_orientation = true;

                bool shifted = false;
                if (cfg.pos(i1, 0) < 0.05*a ||
                    cfg.pos(i1, 1) < 0.05*a ||
                    cfg.pos(i1, 2) < 0.4*a  ||
                    cfg.pos(i1, 0) > 0.95*a ||
                    cfg.pos(i1, 1) > 0.95*a ||
                    cfg.pos(i1, 2) > 0.6*a  ||
                    cfg.pos(i2, 0) < 0.05*a ||
                    cfg.pos(i2, 1) < 0.05*a ||
                    cfg.pos(i2, 2) < 0.4*a  ||
                    cfg.pos(i2, 0) > 0.95*a ||
                    cfg.pos(i2, 1) > 0.95*a ||
                    cfg.pos(i2, 2) > 0.6*a)
                    shifted = true;

                ofs << "cfg# " << ++cntr 
                    << "\ni1= " << i1 << " g1=" << g1 << "\ti2= " << i2  << " g2=" << g2
                    << "\nhigh_grades_check " << high_grades
                    << "\nlow_grades_check " << low_grades
                    << "\nlattice_ortogonality_check " << is_lattice_ortog
                    << "\nlattice_cubic_check " << is_lattice_cubic
                    << "\nwrong_orientation_check " << wrong_orientation
                    << "\nwrong_distance_check " << wrong_dist
                    << "\ndisplscement_check " << shifted
                    << "\n"
                    << endl;

            }
        }

    } END_COMMAND;

    return is_command_found;
}
