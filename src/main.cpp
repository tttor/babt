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

//output filename
std::string datafilename = "../out/defaultout";

//Number of steps to execute
size_t nSteps = 100;
//Number of trials per MDP
size_t nTrials = 1;

//Number of states
size_t S = 25;
//Number of actions
size_t A = 4;

//Discount factor
double gammadisc = 0.96;

//BAMCP RB parameter
int MCTSRB = -1;
int nSims = 1000;

//Default exploration constant
double MCTSEC = 5;
double MCTSEPS = 0.5;

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
    EXPERIMENT::PARAMS expParamsBAMCP;
    expParamsBAMCP.NumSteps = nSteps;
    expParamsBAMCP.AutoExploration = false;
    expParamsBAMCP.TimeOut = 32000;

    BAMCP::PARAMS searchParamsBAMCP;
    searchParamsBAMCP.NumSimulations = nSims;
    searchParamsBAMCP.ExplorationConstant = MCTSEC;
    searchParamsBAMCP.RB = MCTSRB;
    searchParamsBAMCP.eps = MCTSEPS;
    searchParamsBAMCP.MaxDepth = real->GetHorizon(expParamsBAMCP.Accuracy, 
                                                  expParamsBAMCP.UndiscountedHorizon);

    //
    std::vector< std::vector<double> > Rhist(nTrials);
    Rhist = std::vector< std::vector<double> >(nTrials);
    
    std::string filename_bmcp_all(datafilename);
    filename_bmcp_all.append("_bmcp_all");
    
    EXPERIMENT xprmnt(*real, *real, filename_bmcp_all, 
                      expParamsBAMCP, searchParamsBAMCP, *samplerFact);

    // Run experiment ---------------------------------------------------------------------
    std::cout << std::endl << "**************" << std::endl;
    std::cout << "--BAMCP --nSims " << nSims << " EC " << MCTSEC << std::endl;

    CStopWatch timer;
    timer.startTimer();
    for(uint i=0;i<nTrials;++i){
        cout << "trial= " << i+1 << " of " << nTrials << endl;

        Rhist[i].reserve(nSteps);
        xprmnt.Run(Rhist[i]);
    }
    timer.stopTimer();

    // std::string filename_bmcp(datafilename);
    // filename_bmcp.append("_bmcp");
    // utils::dumpc(Rhist,filename_bmcp,nSteps);

    std::cout << "Time: " << timer.getElapsedTime()/(double)nTrials  << " s" << endl;
    utils::append(timer.getElapsedTime()/(double)nTrials,filename_bmcp_all);

    //
    delete real;
}
