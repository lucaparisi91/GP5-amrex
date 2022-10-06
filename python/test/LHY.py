from gp import gp
import gpAmreX
import numpy as np

if __name__ == "__main__":

    gpAmreX.initialize()
    shape=( 64,64 )   
    domain=[ [-10,10],[-10,10] ]
    N=100

    phi0=gp.createField( domain,shape,selections=[],dtype="complex" )

    gp.initGaussian(phi0,alphaReal=1/(2*2**2) )
    phi0.norm=[N]

    dt=0.01* phi0[ len(phi0) - 1  ].geo.cellSize[0] **2

    func=gp.LHYDroplet( )
    stepper=gp.stepper( func, kind="RK4",realTime=False )
    stepper.enableNormalization([N])

    sim=gp.simulation(phi0,stepper=stepper, maxStepsPerBlock=10000, maxNBlocks=10, timeStep=dt )
    sim.outputDir="imag"

    sim.run()


    
