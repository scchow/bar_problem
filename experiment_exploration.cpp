#include <iostream>
#include <vector>
#include <string>

#include "MultiNightBar.hpp"

int main(){

    std::cout << "MultiNightBar Experiment" << std::endl;

    int nAgents = 100;
    int nNights = 10;
    int cap = 10;
    int runFlag = 1; //Run "fixed" trial with no agents disabled
    int tau = 200;
    bool learnTypeD = true;
    bool impactTypeD = true;
    double learningRate = 0.1;
    std::string base_path = "Results/explore/";


    std::vector<double> explorationTrials = {0.05, 0.1, 0.2, 0.5, 0.7, 0.9};
    int numRuns = 10;
    int numEpochs = 3000;

    for (size_t i = 0; i < explorationTrials.size(); ++i){

        for (int j = 0; j < numRuns; ++j){

            std::string path = base_path + "exploration_" + std::to_string(explorationTrials[i]) + "/run_" + std::to_string(j) + "/";

            MultiNightBar barProblem(nAgents, nNights, cap, runFlag, tau, 
                            learnTypeD, impactTypeD, 
                            learningRate, explorationTrials[i],
                            path);

            // barProblem.fixAgents(fixedAgent[i]);

            for (int k = 0; k < numEpochs; ++k){
                barProblem.simulateEpoch(k);
            }

            barProblem.logQTables();
        }

    }

}