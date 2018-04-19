#include <iostream>
#include <vector>
#include <string>

#include "MultiNightBar.hpp"

int main(){

    std::cout << "MultiNightBar Reward Experiment" << std::endl;

    int nAgents = 100;
    int nNights = 10;
    int cap = 10;
    int runFlag = 3;
    double tau = 1;
    bool learnTypeD = true;
    bool impactTypeD = true;
    double learningRate = 0.1;
    double exploration = 0.01;
    std::string base_path = "Results/random_100/";


    // std::vector<double> taus = {0.1, 0.5, 1.0, 1.5, 2.0, 5.0}; //for D
    std::vector<double> learnProb = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3}; //for G

    int numRuns = 100;
    int numEpochs = 3000;
    std::vector<int> gracePeriods = {0};

    for (size_t g = 0; g < gracePeriods.size(); ++g){

        for (size_t i = 0; i < learnProb.size(); ++i){

            for (int j = 0; j < numRuns; ++j){

                std::string path = base_path + "prob_" + std::to_string(learnProb[i]) + "/grace_" + std::to_string(gracePeriods[g]) + "/run_" + std::to_string(j) + "/";

                MultiNightBar barProblem(nAgents, nNights, cap, runFlag, tau, 
                                learnTypeD, impactTypeD, 
                                learningRate, exploration,
                                path, gracePeriods[g]);

                for (int k = 0; k < numEpochs; ++k){
                    barProblem.simulateEpoch(k, learnProb[i]);
                }

                barProblem.logQTables();
            }

        }
    }

}