#include "MultiNightBar.hpp"

bool debug = false;

// Constructor
MultiNightBar::MultiNightBar(int nAgents, int nNights, int cap, int runFlag, double tau,
                             bool learnTypeD, bool impactTypeD,
                             double learningRate, double exploration,
                             std::string path):
                             numAgents(nAgents), numNights(nNights), capacity(cap), runType(runFlag),
                             learningD(learnTypeD), impactD(impactTypeD), 
                             alpha(learningRate), epsilon(exploration), logPath(path)
{
    // save temperature as inverse to save computation
    invTemp = 1/tau;

    // Random number distribution that chooses an int representing a random night
    std::uniform_int_distribution<> distRandNight(0, numNights-1);

    // compute best possible reward per night, for use in optimistic initialization of bar agents
    double maxReward = numNights * capacity;
    if (learningD){
        maxReward = capacity;
    }

    // create a vector of bar agents
    // set initial learning status
    for (int i = 0; i < numAgents; ++i){
        BarAgent* newAgent = new BarAgent(learningRate, exploration, numNights, maxReward);
        agentVec.push_back(newAgent);

        // default to let the agent learn
        learningStatus.push_back(true);

        // default previous action to be random night
        // this allows random initialization of fixed agents
        prevActions.push_back(distRandNight(generator));

        // let previous probability of learning be 1
        prevProbs.push_back(1.0);

        // default difference reward
        prevD.push_back(0);
    }
    
    // default global reward
    prevG = 0.0;

    // check whether there is reason to compute D
    if (learnTypeD or impactTypeD){
        useD = true;
    }
    else{
        useD = false;
    }


    // make directory if it does not exist
    std::string mkdir = "mkdir -p " + path;
    system(mkdir.c_str());

    // setup logging files
    setupLoggers();

    // make a README file
    logReadMe();

}

// Destructor
MultiNightBar::~MultiNightBar(){
    // deallocate all the agents
    for (int i = 0; i < numAgents; ++i){
        delete agentVec[i];
        agentVec[i] = nullptr;
    }

    // close the ofstreams
    numLearningFile.close();
    performanceFile.close();
    // finalActionFile.close();
    agentActionFile.close();
    qTableFile.close();
    readmeFile.close();

}

// Sets numFixedAgents to not Learning
// This is to be used for testing fixed number of non-learning agents
// Usage: run fixAgents(num agents to fix) after initializing the MultiNightBar
// class with runType flag = 1
void MultiNightBar::fixAgents(int numFixedAgents){
    for (int i = 0; i < numFixedAgents; ++i)
    {
        learningStatus[i] = false;
    }
}

// Simulates a single epoch
void MultiNightBar::simulateEpoch(int epochNumber, double learnProb){
    switch(runType){
        case 1: simulateEpochFixed(epochNumber);
                break;
        case 2: simulateEpochImpact(epochNumber);
                break;
        case 3: simulateEpochRandom(epochNumber, learnProb);
                break;
    }
}

// Simulates a single epoch: fixed agent learning
void MultiNightBar::simulateEpochFixed(int epochNumber){

    std::cout << "Simulating Epoch " << epochNumber << std::endl;

    // poll each agent for an action (get previous action for paused agent)
    std::vector<int> actions = getActions();

    // compute the attendance
    std::vector<int> attendance = computeAttendance(actions);

    // compute reward per night
    std::vector<double> rewardPerNight = computeRewardMulti(attendance);

    // compute global reward
    double G = computeG(rewardPerNight);

    // if necessary compute D
    std::vector<double> D;
    if (useD){
        D = computeD(actions, attendance);
        // std::cout << "D:";
        // for (int i = 0; i < D.size(); ++i){
        //     std::cout << D[i] << ",";
        // }
        // std::cout << "\n";
    }

    // update Q tables of learning agents
    if (learningD){
        if (debug){
            std::cout << "learning with D" << std::endl;
        }
        updateQTables(actions, D);
    }
    else{
        if (debug){
            std::cout << "learning with G" << std::endl;
        }
        updateQTables(actions, G);
    }

    // Logs the number of agents learning at each epoch
    logNumLearning(epochNumber);

    // Log the performance at each epoch
    logPerformance(epochNumber, G);

    // Log the learning of each agent
    logLearningStatus();

    // log the actions of each agent
    logAgentActions(actions);

    // log the attendance
    logAttendance(attendance);


    std::cout << "Global Reward = " << G << std::endl;

    std::cout << "Attendance = "; 
    std::cout << attendance[0];

    for (int i = 1; i < numNights; ++i){
        std::cout << ", " << attendance[i];
    }

    std::cout << "\n\n";

}

// Simulates a single epoch: agent learning based on impact
void MultiNightBar::simulateEpochImpact(int epochNumber){

    if (debug)
    {
        std::cout << "Computing Impact Run: Epoch " << epochNumber << std::endl;
    }

    // for the first 10 epochs everyone learns regardless
    if (epochNumber < 10){
        simulateEpochFixed(epochNumber);
        return;
    }

    // poll each agent for an action (get previous action for paused agent)
    std::vector<int> actions = getActions();

    // compute the attendance
    std::vector<int> attendance = computeAttendance(actions);

    // compute reward per night
    std::vector<double> rewardPerNight = computeRewardMulti(attendance);

    // compute global reward
    double G = computeG(rewardPerNight);

    // if necessary compute D
    std::vector<double> D;
    if (useD){
        D = computeD(actions, attendance);
    }

    // compute impact (get previous impact for paused agents)
    std::vector<double> impacts;
    if (impactD){
        impacts = computeImpacts(D);
    }
    else{
        impacts = computeImpacts(G);
    }

    if (debug)
    {
        std::cout << "D: \n";
        printVector(D);

        std::cout << "Prev D: \n";
        printVector(prevD);

        std::cout << "Impacts: \n"; 
        printVector(impacts);
    }

    // compute the probability of learning for each agent
    std::vector<double> probLearning = computeProbLearning(epochNumber, impacts);

    // compute learning status of the agents via probability
    std::vector<bool> newLearningStatus = computeLearningStatus(probLearning);

    // set the learning status of agents, recording probability of learning and action for all learners
    setLearningStatus(newLearningStatus, actions, probLearning);

    // update Q tables of learning agents
    if (learningD){
        updateQTables(actions, D);
    }
    else{
        updateQTables(actions, G);
    }

    // save G and/or D for future impact calculation
    updatePrevG(G);

    if (useD){
        updatePrevD(D);
    }

    // Logs the number of agents learning at each epoch
    int numLearning = logNumLearning(epochNumber);

    // Log the performance at each epoch
    logPerformance(epochNumber, G);

    // Log the learning of each agent
    logLearningStatus();

    // log the actions of each agent
    logAgentActions(actions);

    std::cout << "Global Reward = " << G << std::endl;

    std::cout << "NumLearning = " << numLearning << std::endl;

    std::cout << "Attendance = "; 
    printVector(attendance);


    std::cout << "\n";
}

// Simulates a single epoch: agent learning based on random prob
void MultiNightBar::simulateEpochRandom(int epochNumber, double learnProb){

    // poll each agent for an action (get previous action for paused agent)
    std::vector<int> actions = getActions();

    // compute the attendance
    std::vector<int> attendance = computeAttendance(actions);

    // compute reward per night
    std::vector<double> rewardPerNight = computeRewardMulti(attendance);

    // compute global reward
    double G = computeG(rewardPerNight);

    // if necessary compute D
    std::vector<double> D;
    if (useD){
        D = computeD(actions, attendance);
    }

    // the probability of each agent learning is fixed by learnProb
    std::vector<double> probLearning(numAgents, learnProb);

    // compute learning status of the agents via probability
    // and directly set without considering other impact
    learningStatus = computeLearningStatus(probLearning);

    // update Q tables of learning agents
    if (learningD){
        updateQTables(actions, D);
    }
    else{
        updateQTables(actions, G);
    }

    // save G and/or D for future impact calculation
    updatePrevG(G);

    if (useD){
        updatePrevD(D);
    }
}

// Polls each agent for an action. Uses the default exploration rate.
// For fixed agents, the previous action taken by that agent is used
std::vector<int> MultiNightBar::getActions(){

    std::vector<int> actions(numAgents);

    for (int i = 0; i < numAgents; ++i){
        // If agent is learning, get action straight from agent
        if (learningStatus[i]){
            // std::cout << "getting action from agent \n";
            actions[i] = agentVec[i]->getAction();
            // std::cout << "Got action: " << actions[i] << std::endl;
        }
        // otherwise get its action from previous action table
        else{
            actions[i] = prevActions[i];
        }
    }

    return actions;
}

// Polls each agent for an action. Uses the provided exploration rate.
// For fixed agents, the previous action taken by that agent is used
std::vector<int> MultiNightBar::getActions(double exploration){

    std::vector<int> actions(numAgents);

    for (int i = 0; i < numAgents; ++i){
        // If agent is learning, get action straight from agent
        if (learningStatus[i]){
            actions[i] = agentVec[i]->getAction(exploration);
        }
        // otherwise get its action from previous action table
        else{
            actions[i] = prevActions[i];
        }
    }

    return actions;
}

// Computes the attendance given a vector of agent actions
std::vector<int> MultiNightBar::computeAttendance(const std::vector<int>& actions){
    
    std::vector<int> attendance(numNights, 0);

    for (int i = 0; i < numAgents; ++i){
        attendance[actions[i]]++;
    }

    return attendance;
}

// Computes the reward for a single night
double MultiNightBar::computeRewardSingle(int numAttend){

    return (double)capacity * std::exp(-0.1 * std::pow((double)numAttend - (double)capacity,2));
}

// Computes the rewards for all nights based on attendance
// Calls computeRewardSingle() for each night
std::vector<double> MultiNightBar::computeRewardMulti(const std::vector<int>& attendance){

    std::vector<double> rewardPerNight(numNights, 0);

    for (int i = 0; i < numNights; ++i){
        rewardPerNight[i] = computeRewardSingle(attendance[i]);
    }

    return rewardPerNight;
}

// Computes the global reward given the reward each night
double MultiNightBar::computeG(const std::vector<double>& rewardPerNight){
    
    double G = 0.0;

    for (int i = 0; i < numNights; ++i){
        G += rewardPerNight[i];
    }

    return G;
}

// Computes the difference reward for each agent based on attendance on a particular night
std::vector<double> MultiNightBar::computeD(const std::vector<int>& actions, const std::vector<int>& attendance){

    assert((int) actions.size() == numAgents);
    assert((int) attendance.size() == numNights);

    std::vector<double> D(numAgents, 0);
    // std::cout << "D = ";

    for (int i = 0; i < numAgents; ++i){

        // agent i attended this night
        int nightAttended = actions[i];

        // the original attendance of that night
        int numAttend = attendance[nightAttended];

        // Suppose agent i did not attend that night, compute the difference in reward
        D[i] = computeRewardSingle(numAttend) - computeRewardSingle(numAttend-1);
        // std::cout << D[i]<< ", ";
    }
    // std::cout << "\n";

    return D;
}

// Computes the impact given the global reward for each agent
// For fixed agents, the previous impact is used
std::vector<double> MultiNightBar::computeImpacts(double G){

    std::vector<double> impacts(numAgents, 0);

    double deltaG = std::abs(G - prevG);

    for (int i = 0; i < numAgents; ++i)
    {
        // if agent is learning, compute change in reward/change in policy
        if (learningStatus[i]){
            impacts[i] = deltaG / agentVec[i]->getDeltaPi();
        }
    }

    if (debug){

        std::vector<double> deltaPis(numAgents, 0);

        for (int i = 0; i < numAgents; ++i)
        {
            deltaPis[i] = agentVec[i]->getDeltaPi();
        }

        std::cout << "deltaG: " << deltaG << std::endl;
        std::cout << "delta Pi:\n";
        printVector(deltaPis);
    }

    return impacts;
}

// Computes the impact given the difference rewards for each agent
// For fixed agents, the previous impact is used
std::vector<double> MultiNightBar::computeImpacts(const std::vector<double>& D){

    std::vector<double> impacts(numAgents, 0);

    for (int i = 0; i < numAgents; ++i)
    {
        // if agent is learning, compute change in reward/change in policy
        if (learningStatus[i]){

            impacts[i] = std::abs(D[i] - prevD[i]) / agentVec[i]->getDeltaPi();

        }

    }

    if (debug){

        std::vector<double> deltaDs(numAgents, 0);
        std::vector<double> deltaPis(numAgents, 0);

        for (int i = 0; i < numAgents; ++i)
        {
            deltaDs[i] = D[i] - prevD[i];
            deltaPis[i] = agentVec[i]->getDeltaPi();
        }
        std::cout << "\ndelta D:\n";
        printVector(deltaDs);
        std::cout << "\ndelta Pi:\n";
        printVector(deltaPis);        

    }

    return impacts;
}

std::vector<double> MultiNightBar::computeProbLearning(int epochNumber, const std::vector<double>& impacts){

    std::vector<double> probLearning(numAgents, 0);

    for (int i = 0; i < numAgents; ++i)
    {
        if (learningStatus[i]){
            probLearning[i] = 1.0 - std::exp((-1.0 * impacts[i] * constInvTemp(epochNumber)));
        }
        else{
            probLearning[i] = prevProbs[i];
        }
    }
    
    return probLearning;
}

// Returns a bool vector defining which agents should and should not learn based on probability of learning
std::vector<bool> MultiNightBar::computeLearningStatus(const std::vector<double>& probLearning){

    std::vector<bool> newLearningStatus(numAgents, false);

    for (int i = 0; i < numAgents; ++i){
        
        // randomly generate a number from 0.0 to 1.0
        double rand = distReal(generator);

        // set learning status flag based on rng
        newLearningStatus[i] = (rand < probLearning[i]);
    }

    return newLearningStatus;
}

// Sets the Learning Status for the agents (uses output of computeLearningStatus())
// Also records the impact factor/action for agents who have stopped learning for use in relearning
void MultiNightBar::setLearningStatus(const std::vector<bool>& newLearningStatus, const std::vector<int>& actions, const std::vector<double>& probs){

    prevActions = actions;
    prevProbs = probs;
    learningStatus = newLearningStatus;

    // for (int i = 0; i < numAgents; ++i){
    //     // if the agent is no longer learning, save its action and impact
    //     // if (!newLearningStatus[i]){
    //         prevActions[i] = actions[i];
    //         prevProbs[i] = probs[i];
    //     // }

    //     learningStatus[i] = newLearningStatus[i];
    // }
}


// Updates the Q-Tables of the agents with the same reward (used for Global Reward)
void MultiNightBar::updateQTables(const std::vector<int>& actions, double reward){

    for (int i = 0; i < numAgents; ++i){

        // if the agent learning, update its Q table
        if (learningStatus[i]){
            agentVec[i]->updateQTable(actions[i], reward);
        }
    }
}

// Updates the Q-Tables of the agents with their personalized reward (used for Difference Rewards)
void MultiNightBar::updateQTables(const std::vector<int>& actions, const std::vector<double>& rewards){

    for (int i = 0; i < numAgents; ++i){

        // if the agent learning, update its Q table
        if (learningStatus[i]){
            agentVec[i]->updateQTable(actions[i], rewards[i]);
        }
    }
}

// Saves the Previous D values for impact calculation
void MultiNightBar::updatePrevD(const std::vector<double>& newD){
    prevD = newD;
}

// Saves the previous G value for impact calculations
void MultiNightBar::updatePrevG(double newG){
    prevG = newG;
}

// constant temperature function
double MultiNightBar::constInvTemp(int epochNumber){
    return invTemp;
}


// creates ofstreams for each of the loggers based on the string path provided
void MultiNightBar::setupLoggers(){
    numLearningFile.open(logPath+"numLearning.csv");
    performanceFile.open(logPath+"performance.csv");
    learningStatusFile.open(logPath+"learningStatus.csv");
    agentActionFile.open(logPath+"agentActions.csv");
    qTableFile.open(logPath+"qTable.csv");
    readmeFile.open(logPath+"readme.txt");
    attendanceFile.open(logPath+"attendance.csv");
}

// Logs the number of agents learning at each epoch
int MultiNightBar::logNumLearning(int epochNumber){

    int numLearning = 0;

    for (int i = 0; i < numAgents; ++i)
    {
        numLearning += learningStatus[i];
    }

    numLearningFile << epochNumber << ", " << numLearning << "\n";

    return numLearning;
}

// Log the performance at each epoch
void MultiNightBar::logPerformance(int epochNumber, double G){

    performanceFile << epochNumber << ", " << G << "\n";

}

// Log the learning of each agent
void MultiNightBar::logLearningStatus(){
    
    learningStatusFile << learningStatus[0];

    for (int i = 1; i < numAgents; ++i){
        learningStatusFile << ", " << learningStatus[i];
    }

    learningStatusFile << "\n";
}

// log the actions of each agent
void MultiNightBar::logAgentActions(const std::vector<int>& actions){

    agentActionFile << actions[0];

    for (int i = 1; i < numAgents; ++i){
        agentActionFile << ", " << actions[i];
    }

    agentActionFile << "\n";

}

// log the final q table values of each agent
void MultiNightBar::logQTables(){

    for (int i = 0; i < numAgents; ++i){

        std::vector<double> qTable = agentVec[i]->getQTable();

        qTableFile << qTable[0];

        for (size_t i = 1; i < qTable.size(); ++i)
        {
            qTableFile << ", " << qTable[i];
        }

        qTableFile << "\n";
    }
}

// log the attendance of each night
void MultiNightBar::logAttendance(std::vector<int> attendance){

    attendanceFile << attendance[0];

    for (int i = 1; i < numNights; ++i){
        attendanceFile << ", " << attendance[i];
    }

    attendanceFile << "\n";

}

// Log the run parameters in this readme
void MultiNightBar::logReadMe(){
    readmeFile << "Num Agents: " << numAgents << "\n";
    readmeFile << "Num Nights: " << numNights << "\n";
    readmeFile << "Capacity: " << capacity << "\n";

    readmeFile << "Run Type: " << runType << "\n";

    readmeFile << "Temp: " << 1.0/invTemp << "\n";

    readmeFile << "Learning Using D?: " << learningD << "\n";
    readmeFile << "Impact Calculate Using D?: " << impactD << "\n";

    // flag for computing D
    readmeFile << "Using D: " << useD << "\n";

    /* Agent Params */

    readmeFile << "Learning Rate: " << alpha << "\n";
    // default exploration rate for the agents
    readmeFile << "Exploration Rate: " << epsilon << "\n";
}

