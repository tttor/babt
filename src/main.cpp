#include <iostream>
#include <algorithm>

#include "envs/grid.h"
#include "planners/mcp/experiment.h"
#include "planners/mcp/bamcp/bamcp.h"
#include "samplers/FDMsamplerFactory.h"
#include "utils/rng.h"
#include "utils/hr_time.h"
#include "utils/utils.h"

#include <boost/foreach.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/lexical_cast.hpp>

using namespace boost::numeric::ublas;
using namespace boost;
using namespace std;

//Number of trials per MDP
size_t nTrials = 1;

//Number of states
size_t S = 25;
//Number of actions
size_t A = 4;

//Discount factor
double gammadisc = 0.96;

int main(int argc, char* argv[]) {
    //Init-----------------------------------------------------------------------------
    //Grid env
    assert(pow((int)sqrt(S),2) == S); assert(A == 4);
    SIMULATOR* real = 0;
    real = new Grid(sqrt(S),gammadisc); 

    //
    SamplerFactory* samplerFact;
    samplerFact =  new FDMsamplerFactory( 1/(double)S ); 

    //
    EXPERIMENT::PARAMS expParams;
    expParams.NumSteps = 100;//Number of steps to execute
    expParams.AutoExploration = false;
    expParams.TimeOut = 32000;
    expParams.OutDirpath = "/home/tor/abt/xprmt/xprmt-babt";

    BAMCP::PARAMS searchParams;
    searchParams.NumSimulations = 1000;
    searchParams.ExplorationConstant = 5;
    searchParams.RB = -1;
    searchParams.eps = 0.5;
    searchParams.MaxDepth = real->GetHorizon(expParams.Accuracy, 
                                             expParams.UndiscountedHorizon);

    //
    std::vector< std::vector<double> > Rhist(nTrials);
    Rhist = std::vector< std::vector<double> >(nTrials);
    
    EXPERIMENT xprmnt(*real, *real, expParams, searchParams, *samplerFact);

    // Run experiment ---------------------------------------------------------------------
    std::cout << std::endl << "**************" << std::endl;
    std::cout << "--BAMCP --nSims " << nSims << " EC " << MCTSEC << std::endl;

    CStopWatch timer;
    timer.startTimer();
    for(uint i=0;i<nTrials;++i){
        cout << "trial= " << i+1 << " of " << nTrials << endl;

        Rhist[i].reserve(expParams.NumSteps);
        xprmnt.Run(Rhist[i]);
    }
    timer.stopTimer();
    std::cout << "Time: " << timer.getElapsedTime()/(double)nTrials  << " s" << endl;

    //
    delete real;
}
