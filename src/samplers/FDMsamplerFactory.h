#pragma once

#include <cassert>
#include "samplerFactory.h"

//Flat Dirichlet Multinomial sampler factory

class FDMsamplerFactory : public SamplerFactory{
    public:
        FDMsamplerFactory(double priorcount);
    
        Sampler* getTransitionSampler(const uint* counts, uint s, uint a, uint S);
        Sampler* getMDPSampler(const uint* counts, uint S, uint A, double* R, bool rsas, double gamma);
        Sampler* getTransitionParamSampler(const uint* counts, uint s, uint a, uint S);
        
        void reset(){assert(false&&"TODO: implement!");}
        void updateCounts(uint s, uint a, uint obs){assert(false&&"TODO: implement!");};
        double getAlphaMean(){return alpha;}

    protected:
        double alpha;
};
