import numpy as np
import gpAmreX



class box:
    def __init__(self,extent):
        self.extent=extent

        left=np.array([  dExtent[0]    for dExtent in extent] ).astype(int)
        right=np.array([  dExtent[1]    for dExtent in extent] ).astype(int)
        self._boxCpp=gpAmreX.box(left,right)

    def refine(self, ratio):

        assert(len(ratio)==len(self.shape))
        
        extent2=[ [self.extent[d][0]*ratio[d], (self.extent[d][1]+1)*ratio[d] - 1 ] for d in range(len(ratio) ) ] 

        return box(extent2)



    @property
    def left(self):
        return np.array(self._boxCpp.getLow() )
    @property
    def right(self):
        return np.array(self._boxCpp.getHigh() )
    @property
    def shape(self):
        return self.right - self.left + 1
    def cpp(self):
        return self._boxCpp


class geometry:
    def __init__(self,domain,shape):
        self.domain=domain
        self.shape=shape
        left=np.array( [ extent[0] for extent in self.domain] )
        right=np.array( [ extent[1] for extent in self.domain] )

        self.geometryCpp=gpAmreX.geometry(left,right,self.shape)


    def index(self,x):
        return np.array((np.array(x) - self.left)/self.cellSize).astype(np.int64)

    def selectBox( self, domainSelection):
        
        left=[ext[0] for ext in domainSelection ]
        right=[ext[1] for ext in domainSelection ]
        
        boxLeft=self.index( left )
        boxRight=self.index( right )

        extent=list(zip(boxLeft,boxRight))

        return box(extent)

         


        return box(extent)

    
    def cpp(self):
        return self.geometryCpp
    
    def refine(self, ratio):
        assert(len(ratio) == len( self.shape ))
        shape2=[self.shape[d]*ratio[d] for d in range(len(ratio))  ]
        return geometry(self.domain,shape2)
    
    def domainBox(self):
        extent= [ (0,self.shape[d] - 1) for d in range(len(self.shape)) ]
        return box(extent)

    
    
    @property
    def left(self):
        return np.array(self.geometryCpp.getLeft())
    @property
    def right(self):
        return np.array(self.geometryCpp.getRight() )
    

    @property
    def cellSize(self):
        return (self.right - self.left)/self.shape



def cppLevelType(dtype):
    if not (dtype in ["real","complex"] ):
        raise RuntimeError("level dtype not supported")
    
    return gpAmreX.realLevel if dtype == "real" else gpAmreX.complexLevel 



class level:

    def __init__(self,geo,boxes,dtype="real",nComponents=1):
        self.geo=geo
        self.boxes=boxes
        self.nComponents=nComponents
        self.data=[]
        for currentBox in boxes:
            self.data.append(np.zeros(currentBox.shape) )

        cppBoxes= [ curBox.cpp() for curBox in boxes ]
        realComponents=nComponents
        if (dtype=="complex"):
            realComponents*=2
        

        levelType = cppLevelType(dtype)
    


        self._levelCpp= levelType( geo.cpp(),cppBoxes, nComponents )
        self._dtype=dtype

    @property
    def dtype(self ):
        return self._dtype
    

    @property
    def norm(self):
        return self.cpp().getNorm()

    def cpp(self):
        return self._levelCpp
    

    def __getitem__(self, i):
        data=self.cpp().getData(self.boxes[i].cpp() )
        data=np.squeeze(data)
        return (data)
    

    def __setitem__(self, i,data):
        cBox=self.boxes[i]
        if len(data.shape) == len( cBox.shape):
                data=data.reshape(data.shape + (1,) )
        if self._dtype == complex:
                shape2=data.shape
                shape2[-1]*=2
                data2=np.zeros(shape2)
                for iC in range(self.nComponents):
                    data2[...,iC*2]=np.real(data[...,iC])
                    data2[...,iC*2+1]=np.imag(data[...,iC])
                data=data2
                    
        
        return self.cpp().setData(data, self.boxes[i].cpp() )




class field:
    def __init__(self, levels):
        self._levels=levels
        levelsCpp=[level.cpp() for level in levels]
        self._fieldCpp=gpAmreX.field(levelsCpp)

    
    def averageDown(self):
        self.cpp().averageDown()
    
    def save(self,filename):
        self.cpp().save(filename)
    
    def cpp(self):
        return self._fieldCpp
    def __getitem__(self,i):
        return self._levels[i]
    

    @property
    def norm(self):
        return self.cpp().getNorm()
    
    @norm.setter
    def norm(self,N):
        self.cpp().normalize(N)

class trappedVortex:
    def __init__( self, g=1,omega=[1,1] ):
        self.g=g
        self.omega=omega
        self._trappedVortexCpp=gpAmreX.trappedVortex(g,omega)
        self._initialized=False

    def define(self,phi):
        self.cpp().define(phi.cpp() )
    def cpp( self ):
        return self._trappedVortexCpp

    def apply( self, phiIn,phiOut):
        if not self._initialized:
            self.cpp().define(phiIn.cpp())
            
        self.cpp().apply(phiIn.cpp(),phiOut.cpp() )
    def addVortex(self,center):
        self.cpp().addVortex(center)



class stepper:
    def __init__(self,func):
        self._stepperCpp=gpAmreX.stepper( func.cpp() )
    def cpp(self):
        return self._stepperCpp
    def advance(self,phiOld,phiNew,dt):
        self.cpp().advance(phiOld.cpp(),phiNew.cpp(),dt)
    
    
    
        

def grid( box,geo):
    dims=len(box.shape)
    x=[]
    for d in range(dims):
        x.append( (  np.arange(box.left[d],box.right[d] + 1 ,1 ) + 0.5)*geo.cellSize[d] + geo.left[d] )
    
    return np.meshgrid(*x,indexing="ij")
