#include "experiment.h"
#include "boost/timer.hpp"

using namespace std;

EXPERIMENT::PARAMS::PARAMS()
:   NumRuns(100),
    NumSteps(300),
    SimSteps(1000),
    TimeOut(15000),
    Accuracy(0.01),
    UndiscountedHorizon(1000),
    AutoExploration(true)
{
}

EXPERIMENT::EXPERIMENT(const SIMULATOR& real, const SIMULATOR& simulator, const string& outputFile,
                       EXPERIMENT::PARAMS& expParams, BAMCP::PARAMS& searchParams, SamplerFactory& _samplerFact)
:   Real(real),
    Simulator(simulator),
    ExpParams(expParams),
    SearchParams(searchParams),
    samplerFact(_samplerFact),
    OutputFile(outputFile.c_str())
{
    if (ExpParams.AutoExploration)
    {
        SearchParams.ExplorationConstant = simulator.GetRewardRange();
    }

    BAMCP::InitFastUCB(SearchParams.ExplorationConstant);
}


void EXPERIMENT::Run(std::vector<double>& Rhist)
{
    boost::timer timer;

    BAMCP mcts(Simulator, SearchParams, samplerFact);
    double undiscountedReturn = 0.0;
    double discountedReturn = 0.0;
    double discount = 1.0;
    bool terminal = false;

    uint state = Real.CreateStartState();
    if (SearchParams.Verbose >= 1)
        Real.DisplayState(state, cout);

    for (size_t t = 0; t < ExpParams.NumSteps; t++)
    {
        uint action = mcts.SelectAction(state);// contains UCTSearch() planning

        uint observation;
        double reward;
        terminal = Real.Step(state, action, observation, reward);
        Rhist.push_back(reward);
        
        Results.Reward.Add(reward);
        undiscountedReturn += reward;
        discountedReturn += reward * discount;
        discount *= Real.GetDiscount();

        if (SearchParams.Verbose >= 1)
        {
            Real.DisplayAction(action, cout);
            Real.DisplayState(state, cout);
            Real.DisplayObservation(state, observation, cout);
            Real.DisplayReward(reward, cout);
        }

        if (terminal)
        {
            cout << "Terminated" << endl;
            break;
        }

        mcts.Update(state, action, observation, reward);
        state = observation;//For MDP:

        if (timer.elapsed() > ExpParams.TimeOut)
        {
            cout << "Timed out after " << t << " steps in "
                 << Results.Time.GetTotal() << "seconds" << endl;
            break;
        }
    }

    Results.Time.Add(timer.elapsed());
    Results.UndiscountedReturn.Add(undiscountedReturn);
    Results.DiscountedReturn.Add(discountedReturn);
    cout << "(" << discountedReturn << "," 
         << Results.DiscountedReturn.GetMean() << ":"
         << undiscountedReturn << "," << Results.UndiscountedReturn.GetMean()
         << ") " << flush << endl;
}
