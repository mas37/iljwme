#ifdef MLIP_MPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <algorithm>
#include <iostream>

#include "mlip_handler.h"

#include "../../error_monitor.h"
#include "../../mtpr_trainer.h"
#include "../../common/utils.h"
#include "../../drivers/basic_drivers.h"
#include "../../drivers/relaxation.h"
#include "../../mlp/mlp_commands.cpp"
#include "../../mlp/mlp.cpp"

#include <sstream>

using namespace std;

void init() {
//#ifdef MLIP_MPI
//   MPI_Init(NULL,NULL);
//   mpi.InitComm(MPI_COMM_WORLD);
//#endif
   SetTagLogStream("dev", &std::cout);
}

#ifdef MLIP_MPI
void init_mpi(MPI_Comm comm) {
  mpi.InitComm(comm);
  SetTagLogStream("dev", &std::cout);
}
#endif

void _set_cfg(Configuration& cfg, const cfg_data& cfg_item)					//converts from cfg_item (python-compatible container) to MLIP cfg class
{
    const int& size = cfg_item.size;
    Vector3* cfg_pos = reinterpret_cast<Vector3*>(cfg_item.pos);

	cfg.ClearNbhs();
    cfg.resize(size);
    cfg.has_energy(true);
    if (nullptr != cfg_item.forces) cfg.has_forces(true);
    if (nullptr != cfg_item.stresses) cfg.has_stresses(true);
    cfg.lattice = Matrix3(reinterpret_cast<const double(&)[][3]>(*cfg_item.lat));
    for (int i = 0; i < size; ++i)
    {
        int& type = cfg.type(i) = cfg_item.types[i];
        Vector3& pos = cfg.pos(i) = cfg_pos[i];
        if (nullptr != cfg_item.forces)
            Vector3& forces = cfg.force(i) = (reinterpret_cast<Vector3*>(cfg_item.forces))[i];
    }
    cfg.energy = cfg_item.energy;
    if (nullptr != cfg_item.stresses)
    {
        cfg.stresses[0][0] = cfg_item.stresses[0];
        cfg.stresses[1][1] = cfg_item.stresses[1];
        cfg.stresses[2][2] = cfg_item.stresses[2];
        cfg.stresses[1][2] = cfg_item.stresses[3];
        cfg.stresses[0][2] = cfg_item.stresses[4];
        cfg.stresses[0][1] = cfg_item.stresses[5];
        cfg.stresses[1][0] = cfg.stresses[0][1];
        cfg.stresses[2][1] = cfg.stresses[1][2];
        cfg.stresses[2][0] = cfg.stresses[0][2];
    }
}

void _set_cfg_item(cfg_data& cfg_item, const Configuration& cfg)		//converts from MLIP cfg class to cfg_item (python-compatible container)
{
    int& size = cfg_item.size;
    size = cfg.size();
    Vector3* cfg_pos = reinterpret_cast<Vector3*>(cfg_item.pos);

    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
            cfg_item.lat[3*j+i] = cfg.lattice[j][i];

    for (int i = 0; i < size; ++i)
    {
        long& type = cfg_item.types[i];
        type = cfg.type(i);
        Vector3& pos = cfg_pos[i];
        pos = cfg.pos(i);
        if (cfg.has_forces()) 
        {
            Vector3& forces = (reinterpret_cast<Vector3*>(cfg_item.forces))[i];
            forces = cfg.force(i);
        }
    }
    if (cfg.has_energy())
        cfg_item.energy = cfg.energy;	

    if (cfg.has_stresses())
    {
        cfg_item.stresses[0] = cfg.stresses[0][0];
        cfg_item.stresses[1] = cfg.stresses[1][1];
        cfg_item.stresses[2] = cfg.stresses[2][2];
        cfg_item.stresses[3] = cfg.stresses[0][1];
        cfg_item.stresses[4] = cfg.stresses[1][2];
        cfg_item.stresses[5] = cfg.stresses[0][2];
    }
}

double _cfg_ene(cfg_data &atom_cfg)
{
	return atom_cfg.energy;
}
double _cfg_frc(cfg_data &atom_cfg, int n, int a)
{
	return (reinterpret_cast<Vector3*>(atom_cfg.forces))[n][a];
}
double _cfg_str(cfg_data &atom_cfg, int n)
{
	return atom_cfg.stresses[n];
}

void _train( void* pot_addr,
    const int train_size, const cfg_data *train_cfgs, 
    map<string,string> opts)
{
    if (nullptr == pot_addr)
      return;
    MLMTPR* pMtpr = static_cast<MLMTPR*>(pot_addr);
    vector<Configuration> train_set;

	SetTagLogStream("fit",&std::cout);

#ifdef MLIP_MPI
	  int cntr = 0;
      for (cntr=0 ; cntr < train_size; cntr++) {
          Configuration cfg;
          cfg.features["ID"] = to_string(cntr);
		  _set_cfg(cfg, train_cfgs[cntr]);	
			if (cntr%mpi.size==mpi.rank)
			{         		
          		cfg.has_energy(train_cfgs[cntr].has_energy==1);	
				train_set.emplace_back(cfg);
			}			
      }
		if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
          train_set.emplace_back(Configuration());
#else
	for (int i =0 ; i < train_size; i++) {
          Configuration cfg;
          cfg.features["ID"] = to_string(i);
          	_set_cfg(cfg, train_cfgs[i]);
			cfg.has_energy(train_cfgs[i].has_energy==1);
			train_set.emplace_back(cfg);         		
      }
#endif
    try
    {

#ifdef MLIP_MPI
    MPI_Barrier(mpi.comm);
#endif

    	Settings settings;
		settings.Modify(opts);
		
/*      settings["energy_weight"] = opts["energy_weight"]; 
        settings["force_weight"] = opts["force_weight"]; 
        settings["stress_weight"] = opts["stress_weight"];
        settings["scale_by_force"] = opts["scale_by_force"];
        settings["valid_cfgs"] = opts["valid_cfgs"];
        settings["max_iter"] = opts["max_iter"];
        settings["save_to"] = "";
        settings["bfgs_conv_tol"] = opts["bfgs_conv_tol"];
        settings["weighting"] = opts["weighting"];
        settings["skip_preinit"] = opts["skip_preinit"];
        settings["no_mindist_update"] = opts["no_mindist_update"];	
		settings["init_random"] = opts["init_random"];	
			           
        settings["log"]="stdout";

    	if (settings["energy_weight"] == "")
        	settings["energy_weight"]="1";
    	if (settings["force_weight"] == "")
        	settings["force_weight"]="0.01";
        if (settings["stress_weight"] == "")
        	settings["stress_weight"]="0.001";

		if (opts["weighting"]=="structures")
		{
			settings["weight_scaling_energy"]="2";
			settings["weight_scaling_stress"]="2";
			settings["weight_scaling_forces"]="1";

		}
		else if (opts["weighting"]=="molecules")
		{
			settings["weight_scaling_energy"]="0";
			settings["weight_scaling_stress"]="0";
			settings["weight_scaling_forces"]="0";

		}
		else			//vibrations 
		{
			settings["weight_scaling_energy"]="1";
			settings["weight_scaling_stress"]="1";
			settings["weight_scaling_forces"]="0";

		}
	*/
	
	//pMtpr->inited=false;
    MTPR_trainer trainer(pMtpr,settings); 
    trainer.Train(train_set);
	
	   	string mvs_filename = "new_state.mvs";
        if (opts["save_state_to"] != "")
            mvs_filename = opts["save_state_to"];
        
		settings["select:batch_size"] = "0";
		settings["log"]="";

        settings["energy_weight"]="1";
        settings["force_weight"]="0";
        settings["stress_weight"]="0";

        CfgSelection selector(pMtpr, settings);     
		selector.wgt_scale_power = 1;  // default

		if (opts["weighting"] == "structures" )   
            selector.wgt_scale_power = 2;
        else if ( opts["weighting"] == "vibrations" )
            selector.wgt_scale_power = 1;
        else if ( opts["weighting"] == "molecules" )
            selector.wgt_scale_power = 0;
    
        for (auto& cfg : train_set)
			selector.AddForSelection(cfg);

        selector.Select();

		if (mpi.rank == 0) cout << "Writing maxvol state to the file..." << endl;    
        MPI_Barrier(mpi.comm);
        
        if (mvs_filename != "")
			selector.Save(mvs_filename);
    }
    catch (MlipException& e) {
        Warning("Training failed: " + e.message);
    }

}

void _relax( void* pot_addr,
        const int relax_size, cfg_data *relax_cfgs, 
        int& relaxed_size, int *relaxed, map<string,string> opts,map<string,string> relax_opts)
{
    relaxed_size = 0;

    if ((nullptr == pot_addr)&&(opts["iteration_limit"]!="0"))
      return;

    MLMTPR* pMtpr = static_cast<MLMTPR*>(pot_addr);
    
    try
    {
  		if (opts["iteration_limit"]!="0")
		{
			opts["select:batch_size"]="0";
        	opts["abinitio"] = "FALSE";
        	opts["calculate_efs"] = "TRUE";
			opts["mlip"]="mtpr";
			opts["mlip:load_from"]="temp_relax.mtp";
		}
		
		Settings settings;
        settings.Modify(opts);

		if (settings["log"]=="")
            settings["log"]="stdout";

#ifdef MLIP_MPI
		MPI_Barrier(mpi.comm);
		if (mpi.rank == 0)
			remove("temp_relax.mtp");
#else
		remove("temp_relax.mtp");
#endif
        vector<Configuration> cfgs;

        int count = 0;
		Configuration cfg;
#ifdef MLIP_MPI
         for (count=0; count < relax_size; count++)
         {
			_set_cfg(cfg, relax_cfgs[count]);
            cfg.features["ID"] = to_string(count);
			cfg.features["rank"] = to_string(mpi.rank);
            if (count % mpi.size == mpi.rank)
                cfgs.emplace_back(cfg);
         }

		 if ((count % mpi.size != 0) && mpi.rank >= (count % mpi.size))
         	cfgs.emplace_back(Configuration());    
#else
       for (count=0; count < relax_size; count++)
         {	
			_set_cfg(cfg, relax_cfgs[count]);
            cfg.features["ID"] = to_string(count);
            cfgs.emplace_back(cfg);
         }
#endif        

        int offset = cfgs.size();

     	for (int i=0;i<relax_size;i++)
			relaxed[i]=-1;

		vector<int> relax_ranks(relax_size);
		std::fill(relax_ranks.begin(),relax_ranks.end(), -1);

        Configuration cfg_orig;                             //to save this cfg if failed to relax
        count = 0;
        bool error;                                         //was the relaxation successful or not


	if (nullptr != pot_addr)  //typical relaxation
	{
#ifdef MLIP_MPI
		if (mpi.rank == 0)
			pMtpr->Save("temp_relax.mtp");
MPI_Barrier(mpi.comm);
#else
		pMtpr->Save("temp_relax.mtp");
#endif
	
       	Wrapper* mlip_wrapper = new Wrapper(settings);
        Settings driver_settings(relax_opts);
		driver_settings["relax:log"] = opts["log"];
		Relaxation rlx(driver_settings, mlip_wrapper);	

#ifdef MLIP_MPI
		MPI_Barrier(mpi.comm);
		if (mpi.rank == 0)
			remove("temp_relax.mtp");
#else
		remove("temp_relax.mtp");
#endif

		for (auto& cfg : cfgs)
        {
            if (cfg.size() != 0)
            {
                error = false;
                cfg_orig = cfg;
                rlx.cfg = cfg;
                try { rlx.Run(); }
                catch (MlipException& e) {
                    Warning("Relaxation failed: " + e.message);
                    error = true;
                }

                if (!error) {
                      if (rlx.cfg.size() != cfg.size()) 
                          ERROR("\nInternal error!");
                      _set_cfg_item(relax_cfgs[std::stoi(cfg.features["ID"])],rlx.cfg);
                      relaxed[relaxed_size++] = std::stoi(cfg.features["ID"]);
#ifdef MLIP_MPI
					  relax_ranks[std::stoi(cfg.features["ID"])] = mpi.rank;
#endif
                }
            }
            count++;
        }	

	}
	else if (opts["iteration_limit"]=="0")   //repulsion
	{
		Settings driver_settings(relax_opts);
		driver_settings["relax:log"] = opts["log"];
		Relaxation rlx(driver_settings);	

		        for (auto& cfg : cfgs)
        {
            if (cfg.size() != 0)
            {
                error = false;
                cfg_orig = cfg;
                rlx.cfg = cfg;

                try { rlx.Run(); }
                catch (MlipException& e) {
                    Warning("Relaxation failed: " + e.message);
                    error = true;
                }

                if (!error) {
                      if (rlx.cfg.size() != cfg.size()) 
                          ERROR("\nInternal error!");
                      _set_cfg_item(relax_cfgs[std::stoi(cfg.features["ID"])],rlx.cfg);
                      relaxed[relaxed_size++] = std::stoi(cfg.features["ID"]);
#ifdef MLIP_MPI
					  relax_ranks[std::stoi(cfg.features["ID"])] = mpi.rank;
#endif
                }
            }
            count++;
        }
	}

#ifdef MLIP_MPI
  	MPI_Barrier(mpi.comm);
 	for (int i=0; i < relax_size; i++)
		{		
			int rel_rank=-1;
			MPI_Allreduce(&relax_ranks[i], &rel_rank,1,MPI_INT,MPI_MAX,mpi.comm);
			if (rel_rank>=0)
			{	
				int size = (int)(relax_cfgs[i].size);
				MPI_Bcast(&(relax_cfgs[i].pos[0]),size,MPI_DOUBLE,rel_rank,mpi.comm);
				MPI_Bcast(&(relax_cfgs[i].lat[0]),9,MPI_DOUBLE,rel_rank,mpi.comm);
				MPI_Bcast(&(relax_cfgs[i].forces[0]),3*size,MPI_DOUBLE,rel_rank,mpi.comm);
				MPI_Bcast(&(relax_cfgs[i].stresses[0]),6,MPI_DOUBLE,rel_rank,mpi.comm);
				MPI_Bcast(&(relax_cfgs[i].energy),1,MPI_DOUBLE,rel_rank,mpi.comm);
			}
		}
#endif

#ifdef MLIP_MPI
    MPI_Barrier(mpi.comm);
    int maxrel = offset;

    vector<int> sendbuf(maxrel);		//constructing sending buffers of equal sizes for all processes
    std::fill(sendbuf.begin(), sendbuf.end(), -1);   //fill sendbuf with -1
    memcpy(&sendbuf[0], &relaxed[0], maxrel * sizeof(int));   

    vector<int> total_rel(mpi.size*maxrel);   //constructing the receiving buffer
    std::fill(total_rel.begin(), total_rel.end(), -1);
    MPI_Barrier(mpi.comm);

    MPI_Gather(&sendbuf[0], maxrel, MPI_INT, &total_rel[0], maxrel, MPI_INT, 0, mpi.comm);    //fill the array with species from all processes

	relaxed_size=0;
	if (mpi.rank==0)
		for (int i=0;i<maxrel*mpi.size;i++)
			if (total_rel[i]!=-1)
				relaxed[relaxed_size++]=total_rel[i];

	MPI_Barrier(mpi.comm);
	if (mpi.rank==0)
#endif
	cout << "Out of " << relax_size << " configurations " << relaxed_size << " relaxed successfully" << endl;
      }
    catch (MlipException& e) {
        Warning("Relaxation failed: " + e.message);
    }
}

void _select_add( void* pot_addr,const int train_size, const cfg_data *train_cfgs,
    const int new_size, const cfg_data *new_cfgs, int& diff_size, int *diff, map<string,string> opts)
{
    if (nullptr == pot_addr)
      return;
    MLMTPR* pMtpr = static_cast<MLMTPR*>(pot_addr);

    try
    {
		Settings settings;
        settings.Modify(opts);

		if (settings["log"]=="")
            settings["log"]="stdout";

		string mvs_filename = settings["select:save_state_to"];
		int selection_limit = 0;
		if (settings["select:swap_limit"]!="")
			selection_limit = std::stoi(settings["select:swap_limit"]);

		CfgSelection selector(pMtpr, settings);

        for (Configuration& x : selector.selected_cfgs)
              x.features["ID"] = "-1";
       
        int count = 0;
		Configuration cfg,cfg_e;

	if (opts["select:load_state_from"]!="")
		opts["load_state_from"]=opts["select:load_state_from"];

        //Initialize selection with the training set
	if (opts["load_state_from"]=="")
	{
#ifdef MLIP_MPI
         for (count=0; count < train_size; count++)
         {
			_set_cfg(cfg, train_cfgs[count]);
            if (count % mpi.size == mpi.rank)
                selector.AddForSelection(cfg);
         }
        
        if ((count % mpi.size != 0) && mpi.rank >= (count % mpi.size))
            selector.AddForSelection(cfg_e);
#else
       for (count=0; count < train_size; count++)
         {	
			_set_cfg(cfg, train_cfgs[count]);
            selector.AddForSelection(cfg);
         }
#endif 
		selector.Select();
	}
	else 
	{
		selector.Load(opts["load_state_from"]);
	}

        for (Configuration& x : selector.selected_cfgs)
            x.features["ID"] = "-1";

		//Load the new configurations for selection
#ifdef MLIP_MPI
          for (count=0; count < new_size; count++)
         {
			_set_cfg(cfg, new_cfgs[count]);
            cfg.features["ID"] = to_string(count);
            if (count % mpi.size == mpi.rank)
			{
                selector.AddForSelection(cfg);
			}
         }
        
        if ((count % mpi.size != 0) && mpi.rank >= (count % mpi.size))
            selector.AddForSelection(cfg_e);
#else
       for (count=0; count < new_size; count++)
         {	
			_set_cfg(cfg, new_cfgs[count]);
            cfg.features["ID"] = to_string(count);
            selector.AddForSelection(cfg);
         }
#endif        
        if (selection_limit==0)
            selector.Select();
        else
            {
            selector.Select(selection_limit);
            cout << "Swap limit = " << selection_limit << endl;
            }

        {
            count = 0;
            for (Configuration& cfg : selector.selected_cfgs)
                if (cfg.size() > 0) count++;

            //cout << count << " configurations selected from both sets\n" << std::flush;
        }

        vector<int> valid_to_train;
        std::set<int> unique_cfg;
        count = 0;
        int idiff = 0;
        for (Configuration& x : selector.selected_cfgs) {
            if (stoi(x.features["ID"]) >= 0 && unique_cfg.count(stoi(x.features["ID"])) == 0)
			{
                valid_to_train.push_back(stoi(x.features["ID"]));
                if (count < diff_size)
                	diff[idiff++] = stoi(x.features["ID"]);
                unique_cfg.insert(stoi(x.features["ID"]));
                count++;
            }
        }
        diff_size = idiff;

#ifdef MLIP_MPI
    MPI_Barrier(mpi.comm);
    int maxlen = 0;

    MPI_Allreduce(&idiff, &maxlen, 1, MPI_INT, MPI_MAX, mpi.comm);                     //finding maximum lengths of db_species among the processes

    vector<int> sendbuf(maxlen);		//constructing a sending buffers of equal sizes for all processes
    std::fill(sendbuf.begin(), sendbuf.end(), -1);   //fill sendbuf with -1
    memcpy(&sendbuf[0], &diff[0], idiff * sizeof(int));  //fill some parts of the sendbuf with actual species detected for each process (for a part of the training set)  

    vector<int> total_diff(mpi.size*maxlen);   //constructing the receiving buffer
    std::fill(total_diff.begin(), total_diff.end(), -1);
    MPI_Barrier(mpi.comm);

    MPI_Gather(&sendbuf[0], maxlen, MPI_INT, &total_diff[0], maxlen, MPI_INT, 0, mpi.comm);    //fill the array with species from all processes

	diff_size=0;
	if (mpi.rank==0)
		for (int i=0;i<maxlen*mpi.size;i++)
			if (total_diff[i]!=-1)
				diff[diff_size++]=total_diff[i];

	MPI_Barrier(mpi.comm);
#endif

#ifdef MLIP_MPI
	if (mpi.rank==0)
#endif
    cout << "TS increased by " << diff_size << " configs" << endl;
	if (mvs_filename!="")
		selector.Save(mvs_filename);
    }
    catch (MlipException& e) {
		cout << e.message << endl;
		cout.flush();
        ERROR("Selection failed");
    }
}

map<string,string> _calcerrors(void* pot_addr, const int size, const cfg_data *cfgs, bool on_screen)
{
	map<string,string> dict  = {};

	try{
		MLMTPR* pMtpr = static_cast<MLMTPR*>(pot_addr);
		vector<Configuration> check_set;
		
#ifdef MLIP_MPI
	  int cntr = 0;
      for (cntr=0 ; cntr < size; cntr++) {
          Configuration cfg;
          cfg.features["ID"] = to_string(cntr);
		  _set_cfg(cfg, cfgs[cntr]);	
			if (cntr%mpi.size==mpi.rank)
			{         		
          		cfg.has_energy(cfgs[cntr].has_energy==1);	
				check_set.emplace_back(cfg);
			}			
      }
		if ((cntr % mpi.size != 0) && mpi.rank >= (cntr % mpi.size))
          check_set.emplace_back(Configuration());
#else
	for (int i =0 ; i < size; i++) {
          Configuration cfg;
          cfg.features["ID"] = to_string(i);
          	_set_cfg(cfg, cfgs[i]);
			cfg.has_energy(cfgs[i].has_energy==1);
			check_set.emplace_back(cfg);         		
      }
#endif
		
		Configuration cfg;
		ErrorMonitor errmon;
		cout.precision(10);
		errmon.Reset();
		double errmax = 0;
		
		for (Configuration& cfg_orig : check_set)
		{
			cfg = cfg_orig;
			pMtpr->CalcEFS(cfg);
			errmon.AddToCompare(cfg_orig, cfg);
		}

		std::string report="";
#ifdef MLIP_MPI
		MPI_Barrier(mpi.comm);
		if (mpi.rank == 0)
		{
#endif		
		if (on_screen)
		{
		cout << std::endl << "\t\t* * * MTP ERRORS * * *" << std::endl << std::endl; 			
		errmon.GetReport();
		cout << report.c_str(); 
		}

		dict["Energy: Maximal absolute difference"] = to_string(errmon.ene_all.max.delta);
		dict["Energy: Average absolute difference"] = to_string(errmon.ene_aveabs());
		dict["Energy: RMS     absolute difference"] = to_string(errmon.ene_rmsabs());

		dict["Energy per atom: Maximal absolute difference"] = to_string(errmon.epa_all.max.delta);
		dict["Energy per atom: Average absolute difference"] = to_string(errmon.epa_aveabs());
		dict["Energy per atom: RMS absolute difference"] = to_string(errmon.epa_rmsabs());

		dict["Forces: Maximal absolute difference"] = to_string(errmon.frc_all.max.delta);
		dict["Forces: Average absolute difference"] = to_string(errmon.frc_aveabs());
		dict["Forces: RMS absolute difference"] = to_string(errmon.frc_rmsabs());
		dict["Forces: Max(ForceDiff) / Max(Force)"] = to_string(errmon.frc_all.max.delta / 
                                                     (errmon.frc_all.max.value + 1.0e-300));
		dict["Forces: RMS(ForceDiff) / RMS(Force)"] = to_string(errmon.frc_rmsrel());

		dict["Stresses: Maximal absolute difference"] = to_string(errmon.str_all.max.delta);
		dict["Stresses: Average absolute difference"] = to_string(errmon.str_aveabs());
		dict["Stresses: RMS absolute difference"] = to_string(errmon.str_rmsabs());
		dict["Stresses: Max(StressDiff) / Max(Stress)"] = to_string(errmon.str_all.max.delta / 
                                                     (errmon.str_all.max.value + 1.0e-300));
		dict["Stresses: RMS(StressDiff) / RMS(Stress)"] = to_string(errmon.str_rmsrel());
#ifdef MLIP_MPI
		}
		MPI_Barrier(mpi.comm);
#endif
		return dict; 
    }
    catch (MlipException& e) {
		cout << e.message << endl;
		cout.flush();
        ERROR("Error calculation failed");
    }
	return dict;
}

int _run_command(const string& infname, const string& outfname, int cfg_pos, const string& elements)
{
    std::vector<std::string> args;
    std::map<std::string, std::string> opts;
	std::string command = "convert_cfg";

	if (cfg_pos==1)
	{
		opts["elements_order"]=elements;
		opts["output_format"]="vasp_poscar";
	}
	else
	{
		opts["absolute_elements"]='1';
		opts["input_format"]="vasp_outcar";
	}

	args.push_back(infname);
	args.push_back(outfname);
	
    //ParseOptions(argc-1, argv+1, args, opts);

    try {
        ExecuteCommand(command, args, opts);
    }
    catch (const MlipException& e) { std::cerr << e.What(); return 1; }

	return 0;
}
/////////////////
// POT_HANDLER //
/////////////////
int pot_handler::calc_cfg_efs(cfg_data *atom_cfg)
{
	try{
	Configuration cfg;
	_set_cfg(cfg, atom_cfg[0]);

	//pMtpr->CalcEFS(cfg);
	potWrapper->Process(cfg);

	_set_cfg_item(atom_cfg[0],cfg);
	}
	catch (MlipException& e) {
        Warning("Calc EFS failed: " + e.message);
		return -1;
    }
	return 0;
}

void pot_handler::init_wrapper(map<string,string> opts)
{
try {

		opts["mlip:load_from"]="temp_wrap.mtp";
#ifdef MLIP_MPI
		if (mpi.rank == 0)
			pMtpr->Save("temp_wrap.mtp");
MPI_Barrier(mpi.comm);
#else
		pMtpr->Save("temp_wrap.mtp");
#endif
        potWrapper = new Wrapper(opts);

#ifdef MLIP_MPI
		MPI_Barrier(mpi.comm);
		if (mpi.rank == 0)
			remove("temp_wrap.mtp");
#else
		remove("temp_wrap.mtp");
#endif
	}
 catch (MlipException& e) {
        Warning("MTP initialization failed: " + e.message);
    }
}

pot_handler::pot_handler() : pMtpr(nullptr)//,pWrapper(nullptr)
{ }

pot_handler::~pot_handler() 
{
    delete pMtpr;
//     delete pWrapper;
}

void pot_handler::load_potential(const string& filename)
{
    try {
        pMtpr = new MLMTPR(filename);
        std::stringstream slog;
        slog << "MTPR is loaded from " << filename << endl;
        MLP_LOG("mlippy",slog.str());
    }
    catch (MlipException& e) {
        Warning("Loading MTP potential failed: " + e.message);
    }
    
    n_types = pMtpr->species_count;
    n_coeffs = pMtpr->CoeffCount();
}

void pot_handler::save_potential(const string& filename)
{
    try {
        pMtpr->Save(filename);
        std::stringstream slog;
        slog << "MTPR is saved to " << filename << endl;
        MLP_LOG("mlippy",slog.str());
    }
    catch (MlipException& e) {
        Warning("Saving MTP potential failed: " + e.message);
    }
}

vector<int> pot_handler::species_avail()
{
	return pMtpr->atomic_numbers;   
}

/*
int pot_handler::load_settings(const string&  settings_fn)
{   
    try{
    if( nullptr != pWrapper) delete pWrapper;
    pWrapper = new MLIP_Wrapper(LoadSettings(settings_fn));
    }
    catch (MlipException& e) {
        Warning("MLIP Wrapper loading failed: " + e.message);
    }

    return 0;
}   
    
int pot_handler::pass_settings(map<string,string>& setting_map)
{
    
    try{
    Settings settings(setting_map);
    if( nullptr != pWrapper) delete pWrapper;
    pWrapper = new MLIP_Wrapper(setting_map);

    }
    catch (MlipException& e) {
                    Warning("MLIP Wrapper loading failed: " + e.message);
    }
    
    return 0;
}
*/
