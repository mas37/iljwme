/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../../control_unit.h"
          

using namespace std;


int main(int argc, char* argv[])
{
	try
	{
		if (argc != 2)
			ERROR("wrong argument count");

		int pop_size = stoi(argv[1]);
		//double add_rlx_trshld = stod(argv[2]);

		ControlUnit cu("MLIP-settings.ini");
		Relaxation* p_rlx = (Relaxation*)cu.p_driver;

		if (p_rlx1==nullptr)
			ERROR("Relaxation is not initialized");

		for (int i=1; i<=pop_size; i++)
		{
			string fnm = to_string(i) + ".cfg";
			if (system(("mv for_relax/" + fnm + " temp/" + fnm).c_str()) == 0)
			{
				Message("Reading structures from " + fnm);
				ifstream ifs("temp/" + fnm);
				Configuration cfg;
				while (cfg.Load(ifs))
				{
					if (cfg.features.count("pressure") == 0)
						cfg.features["pressure"] = "0";
					p_rlx->pressure = stod(cfg.features["pressure"]);
					
					cout << "Relaxation structure#" << i
						<< " with " << cfg.size() << " atoms. ";
					cout.flush();

					p_rlx->cfg = cfg;
					p_rlx->Run();
					cfg = p_rlx1->cfg;
					cout << "Energy after relaxation: " << cfg.energy << ", status: " << cfg.features["from"];
					cout.flush();
					
					system(("rm -f relaxed/" + fnm).c_str());
					//rlx.cfg.features.erase("pressure");
					cfg.AppendToFile("relaxed/" + fnm);
					system(("rm -f temp/" + fnm).c_str());
					Message("Relaxed structure saved to " + fnm);
				}
			}
		}
	}
	catch (MlipException& ex)
	{
		cout << ex.What() << endl;
		cerr << ex.What() << endl;
		return 1;
	}
}

