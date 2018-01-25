#ifndef MULTI_NIGHT_BAR_H_
#define MULTI_NIGHT_BAR_H_

#include <string>
#include <vector>

#include <random> // for rng
#include <cmath>  // for exp, abs functions
#include <limits> // for infinity

// #include <boost/lexical_cast.hpp>

#include <iostream>
// #include <fstream>
// #include <sstream>

#include <assert.h>

#include <stddef.h> // for int
// #include <experimental/filesystem>


#include "BarAgent.hpp"

// namespace fs = std::experimental::filesystem;


class MultiNightBar
{
public:
    // Constructor
    MultiNightBar(int nAgents, int nNights, int cap, int runFlag, double tau, 
                    bool learnTypeD, bool impactTypeD, 
                    double learningRate, double exploration,
                    std::string path);

    // Destructor
    ~MultiNightBar();

    // Sets numFixedAgents to not Learning
    // This is to be used for testing fixed number of non-learning agents
    void fixAgents(int numFixedAgents);

    // Simulates a single epoch
    void simulateEpoch(int epochNumber, double learnProb = 0.0);

    // Simulates a single epoch: fixed agent learning
    void simulateEpochFixed(int epochNumber);

    // Simulates a single epoch: agent learning based on impact
    void simulateEpochImpact(int epochNumber);

    // Simulates a single epoch: agent learning based on random prob
    void simulateEpochRandom(int epochNumber, double learnProb);

    // Polls each agent for an action. Uses the default exploration rate.
    // For fixed agents, the previous action taken by that agent is used
    std::vector<int> getActions();

    // Polls each agent for an action. Uses the provided exploration rate.
    // Returns a vector with the i-th term is the night the i-th agent attends
    // For fixed agents, the previous action taken by that agent is used
    std::vector<int> getActions(double exploration);

    // Computes the attendance given a vector of agent actions
    std::vector<int> computeAttendance(const std::vector<int>& actions);

    // Computes the reward for a single night
    double computeRewardSingle(int numAttend);

    // Computes the rewards for all nights based on attendance
    // Calls computeRewardSingle() for each night
    std::vector<double> computeRewardMulti(const std::vector<int>& attendance);

    // Computes the global reward based all the rewards
    double computeG(const std::vector<double>& rewardPerNight);

    // Computes the difference reward for each agent based on actions on a particular night
    std::vector<double> computeD(const std::vector<int>& actions, const std::vector<int>& attendance);

    // Computes the impact given the global reward for each agent
    // For fixed agents, the previous impact is used
    std::vector<double> computeImpacts(double G);

    // Computes the impact given the difference rewards for each agent
    // For fixed agents, the previous impact is used
    std::vector<double> computeImpacts(const std::vector<double>& D);

    // Compute probability of learning based on impact
    std::vector<double> computeProbLearning(int epochNumber, const std::vector<double>& impacts);

    // Returns a bool vector defining which agents should and should not learn
    std::vector<bool> computeLearningStatus(const std::vector<double>& probLearning);

    // Sets the Learning Status for the agents (uses output of computeLearningStatus())
    // Also records the impact factor/action for agents who have stopped learning for use in relearning
    void setLearningStatus(const std::vector<bool>& learningStatus, const std::vector<int>& actions, const std::vector<double>& impacts);

    // Updates the Q-Tables of the agents with the same reward (used for Global Reward)
    void updateQTables(const std::vector<int>& actions, double reward);

    // Updates the Q-Tables of the agents with their personalized reward (used for Difference Rewards)
    void updateQTables(const std::vector<int>& actions, const std::vector<double>& rewards);

    // Saves the Previous D values for impact calculation
    void updatePrevD(const std::vector<double>& newD);

    // Saves the previous G value for impact calculations
    void updatePrevG(double newG);

    // constant temperature function - returns the inverse of temperature value to save computation
    double constInvTemp(int epochNumber);

    // creates ofstreams for each of the loggers based on the string path provided
    void setupLoggers();

    // Logs and return the number of agents learning at each epoch 
    int logNumLearning(int epochNumber);

    // Log the performance at each epoch
    void logPerformance(int epochNumber, double G);

    // Log the learning of each agent
    void logLearningStatus();

    // log the actions of each agent
    void logAgentActions(const std::vector<int>& actions);

    // log the final q table values of each agent
    void logQTables();

    // Log the run parameters in this readme
    void logReadMe();

    // Log the final actions
    void logAttendance(const std::vector<int> attendance);

private:

    /* Bar Domain Parameters */

    // number of agents in domain
    int numAgents;
    // number of nights bar is open
    int numNights;
    // capacity of each night
    int capacity;

    // Type of Run
    // 1 - Fixed agent
    // 2 - Impact Calculation
    // 3 - Random - with probability passed into simulateEpoch
    int runType;

    // inverse of tau - temperature value
    double invTemp;

    // learning type (True if agents learn using D, else agents learning using G)
    bool learningD;
    // impact calculation type (True if impact is computed using difference in D, else impact uses difference in G between epochs)
    bool impactD;

    // flag for computing D
    bool useD;


    /* Agent Params */

    // default learning rate for the agents
    double alpha;
    // default exploration rate for the agents
    double epsilon;
    // vector of Q-Learning Bar Agent Pointers
    std::vector<BarAgent*> agentVec;
    // learning status vector - bool vector, 0 = nonlearning, 1 = learning
    std::vector<bool> learningStatus;
    // remember previous actions for non-learning agents
    std::vector<double> prevActions;
    // remember previous impacts for non-learning agents
    std::vector<double> prevImpacts;


    /* Vars for memory of previous runs for impact/non-learning agents */

    // previous G
    double prevG;
    // previous D
    std::vector<double> prevD;


    /* Random number generation variables */

    // random seed - c++ makes a "random seed" somehow
    std::random_device rd;
    // Random Number Generator initialized with seed from rd
    std::mt19937_64 generator{rd()}; 
    // Random number distribution from 0 to 1
    std::uniform_real_distribution<> distReal{0.0, 1.0}; 

    
    /* Log variables */

    // path to log files
    std::string logPath;

    // ofstreams to write data to files
    std::ofstream numLearningFile;
    std::ofstream performanceFile;
    std::ofstream learningStatusFile;
    std::ofstream agentActionFile;
    std::ofstream qTableFile;
    std::ofstream readmeFile;
    std::ofstream attendanceFile;


};

template <class T>
void printVector(std::vector<T> v){

    std::cout << v[0];

    for (size_t i = 1; i < v.size(); ++i){
        std::cout << ", " << v[i];
    }
    std::cout << "\n";

}

#endif // MULTI_NIGHT_BAR_H_