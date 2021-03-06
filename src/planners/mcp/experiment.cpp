#include "experiment.h"
#include "boost/timer.hpp"
#include <string>     // std::string, std::to_string

using namespace std;

EXPERIMENT::PARAMS::PARAMS()
:   NumRuns(100),
    NumSteps(300),
    SimSteps(1000),
    TimeOut(15000),
    Accuracy(0.01),
    UndiscountedHorizon(1000),
    AutoExploration(true),
    OutDirpath(string())
{
}

EXPERIMENT::EXPERIMENT(const SIMULATOR& real, const SIMULATOR& simulator, 
                       EXPERIMENT::PARAMS& expParams, BAMCP::PARAMS& searchParams, 
                       SamplerFactory& _samplerFact)
:   Real(real),
    Simulator(simulator),
    ExpParams(expParams),
    SearchParams(searchParams),
    samplerFact(_samplerFact)
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

    uint S = Real.GetNumObservations();
    uint A = Real.GetNumActions();

    utils::Counts counts(S);// transition counts, size=SAS
    utils::Posteriors currPosteriors(S);// posterior of transition prob.
    utils::Posteriors prevPosteriors(S);// posterior of transition prob.
    for (uint i=0; i<S; ++i) {
        counts.at(i).resize(A);
        currPosteriors.at(i).resize(A);
        prevPosteriors.at(i).resize(A);
        for (uint j=0; j<A; ++j) {
            counts.at(i).at(j).resize(S);
            currPosteriors.at(i).at(j).resize(S);
            prevPosteriors.at(i).at(j).resize(S);
            for (uint k=0; k<S; ++k) prevPosteriors[i][j][k] = samplerFact.getAlphaMean();
        }
    }
    
    std::vector< std::vector<double> > posteriorDistances(ExpParams.NumSteps);
    for(uint i=0; i<ExpParams.NumSteps; ++i) posteriorDistances.at(i).resize(S);

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

        //
        mcts.Update(state, action, observation, reward);
        state = observation;//For MDP:

        mcts.GetCounts(&counts);
        for (uint s=0; s<S; ++s) {
            string filename(ExpParams.OutDirpath+"/counts_at_"+to_string(s));
            utils::dump(counts.at(s),filename);
        }

        if (t>0) {
            utils::updatePosteriors(counts,prevPosteriors,&currPosteriors);
            utils::getPosteriorDistances(currPosteriors,prevPosteriors,
                                         &(posteriorDistances.at(t)));
            prevPosteriors = currPosteriors;
        }

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

    string filename(ExpParams.OutDirpath+"/posteriorDistances.csv");
    utils::dump(posteriorDistances,filename);
}
