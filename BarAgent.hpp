#ifndef BAR_AGENT_H_
#define BAR_AGENT_H_

// #include <algorithm>

#include <string>
#include <vector>

#include <random> // for rng
#include <cmath>  // for exp, abs functions

#include <iostream>
#include <fstream>
#include <sstream>


#include <stddef.h> // for int

class BarAgent{

public:

    /* Constructor */
    BarAgent(double learningRate, double exploration, int numNights, double maxReward);

    /* Destructor */
    ~BarAgent();

    // returns the night the agent will go to, computed with default exploration rate
    int getAction();

    // Do the exploitive action and choose the best action
    // If there are multiple best actions, randomly choose among them
    int getBestAction();

    // returns the night the agent will go to, computed with provided exploration rate
    int getAction(double exploration);

    // updates the q table for a given action, reward pair
    // Also updates the recorded change in Q (ie. deltaPi)
    void updateQTable(int action, double reward);

    // sets the learning rate
    void setLearningRate(double learningRate);

    // sets the exploration rate
    void setExploration(double exploration);

    // returns how much the policy has changed
    double getDeltaPi();

    // returns the Q-Table
    std::vector<double> getQTable();


private:

    // Q-Table
    std::vector<double> Q;

    // learning rate (alpha)
    double alpha;

    // Note: because bar problem is a single state problem, no lookahead so no discount term
    //// discount factor (gamma)
    //double gamma;

    // exploration rate (epsilon)
    double epsilon;

    // change in policy
    double deltaPi;

    // number of actions (nights)
    double numActions;

    /* Random Number Stuff */
    std::random_device rd; /// Seed Generator
    std::mt19937_64 generator{rd()}; /// generator initialized with seed from rd
    std::uniform_real_distribution<> distReal{0.0, 1.0}; /// Random number distribution from 0 to 1
    std::uniform_int_distribution<> distRandNight{0, 2}; /// Random int distribution (to be overwritten upon actual init)

};

#endif // BAR_AGENT_H_