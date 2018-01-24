#include <iostream>
#include <vector>
#include <string>

#include "MultiNightBar.hpp"

int main(){

    std::cout << "MultiNightBar Experiment" << std::endl;

    int nAgents = 100;
    int nNights = 10;
    int cap = 10;
    int runFlag = 1;
    int tau = 200;
    bool learnTypeD = true;
    bool impactTypeD = true;
    double learningRate = 0.1;
    double exploration = 0.01;
    std::string base_path = "Results/Test2/";


    std::vector<int> fixedAgent = {0, 20, 50, 70, 90};
    int numRuns = 2;
    int numEpochs = 3000;

    for (size_t i = 0; i < fixedAgent.size(); ++i){

        for (int j = 0; j < numRuns; ++j){

            std::string path = base_path + "fixedagent_" + std::to_string(fixedAgent[i]) + "/run_" + std::to_string(j) + "/";

            MultiNightBar barProblem(nAgents, nNights, cap, runFlag, tau, 
                            learnTypeD, impactTypeD, 
                            learningRate, exploration,
                            path);

            // barProblem.fixAgents(fixedAgent[i]);

            for (int k = 0; k < numEpochs; ++k){
                barProblem.simulateEpoch(k);
            }

            barProblem.logQTables();
        }

    }

}