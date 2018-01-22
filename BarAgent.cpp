#include "BarAgent.hpp"

/* Constructor */
BarAgent::BarAgent(double learningRate, double exploration, int numNights, double maxReward):
                    alpha(learningRate), epsilon(exploration), numActions(numNights){

    // populate the Q-Table using optimistic initialization
    for (int i = 0; i < numNights; ++i){
        Q.push_back(maxReward);
    }

    // Random number distribution that chooses an int representing a random night
    distRandNight = std::uniform_int_distribution<>(0, numActions-1);

}

/* Destructor */
BarAgent::~BarAgent(){
}

// returns the night the agent will go to, computed with default exploration rate
int BarAgent::getAction(){

    // determine if agent explores or exploits
    double rand = distReal(generator);

    // Probability Epsilon: Explore
    if (rand < epsilon){
        return distRandNight(generator);
    }

    // Otherwise Exploit
    else{
        return getBestAction();
    }
}

// returns the night the agent will go to, computed with provided exploration rate
int BarAgent::getAction(double exploration){

    // determine if agent explores or exploits
    double rand = distReal(generator);

    // Probability Epsilon: Explore
    if (rand < exploration){
        return distRandNight(generator);
    }

    // Otherwise Exploit
    else{
        return getBestAction();
    }
}

// Do the exploitive action and choose the best action
// If there are multiple best actions, randomly choose among them
int BarAgent::getBestAction(){
    /* Check for multiple best actions */

    // Keep track of max value/indices
    int maxValue = Q[0];
    std::vector<int> maxIndices = {0};

    // Loop through the max indices
    for (size_t i = 1; i < Q.size(); ++i){
        // If we encounter a new max, clear the indices list, and update
        if (Q[i] > maxValue){
            maxIndices.clear();
            maxValue = Q[i];
            maxIndices.push_back(i);
        }
        // If we encounter entry with (about the) same value as max
        else if (std::abs(Q[i] - maxValue) < 1e-9){
            maxIndices.push_back(i);
        }
    }

    // If there is one clear max value, take that action
    if (maxIndices.size() == 1){
        return maxIndices[0];
    }

    // Otherwise we tie-break
    else{
        std::uniform_int_distribution<> dist(0, maxIndices.size()-1);
        int randInt = dist(generator);
        return maxIndices[randInt];
    }

}

// updates the q table for a given action, reward pair
// Also updates the recorded change in Q (ie. deltaPi)
void BarAgent::updateQTable(int action, double reward){

    double newQ = ((1-alpha) * Q[action]) + (alpha * reward);
    deltaPi = std::abs(newQ - Q[action]);
    Q[action] = newQ;
}

// sets the learning rate
void BarAgent::setLearningRate(double learningRate){
    alpha = learningRate;
}

// sets the exploration rate
void BarAgent::setExploration(double exploration){
    epsilon = exploration;
}

// returns how much the policy has changed
double BarAgent::getDeltaPi(){
    return deltaPi;
}

std::vector<double> BarAgent::getQTable(){
    return Q;
}