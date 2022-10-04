import pathlib
import numpy as np
import gpAmreX
from ruamel.yaml import YAML
import pathlib
import h5py



yaml=YAML()


class box:

    @staticmethod
    def fromYAML(ob):
        return box(ob["extent"])    

    def __init__(self,extent):
        self.extent=extent

        left=np.array([  dExtent[0]    for dExtent in extent] ).astype(int)
        right=np.array([  dExtent[1]    for dExtent in extent] ).astype(int)
        self._boxCpp=gpAmreX.box(left,right)

    def refine(self, ratio):

        assert(len(ratio)==len(self.shape))
        
        extent2=[ [self.extent[d][0]*ratio[d], (self.extent[d][1]+1)*ratio[d] - 1 ] for d in range(len(ratio) ) ] 

        return box(extent2)
    
    def toYAML(self):
        return {"extent": self.extent }



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

    @staticmethod
    def fromYAML( obYAML  ):
        domain=obYAML["domain"]
        shape=obYAML["shape"]
        return geometry( domain,shape )


    def __init__(self,domain,shape):
        self.domain=domain
        self.shape=shape
        left=np.array( [ extent[0] for extent in self.domain] )
        right=np.array( [ extent[1] for extent in self.domain] )

        self.geometryCpp=gpAmreX.geometry(left,right,self.shape)
    
    def toYAML(self):
        return {"domain":self.domain,"shape":self.shape }
    

    def index(self,x):
        return np.array((np.array(x) - self.left)/self.cellSize).astype( int)

    def selectBox( self, domainSelection):
        
        left=[ext[0] for ext in domainSelection ]
        right=[ext[1] for ext in domainSelection ]
        
        boxLeft=self.index( left ).tolist()
        boxRight=self.index( right ).tolist()

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
    return gpAmreX.level 


class level:

    @staticmethod
    def fromYAML(ob):
        """ Warning: does not load the data array """
        geo=geometry.fromYAML(ob["geo"])
        boxes = [ box.fromYAML(b)   for b in ob["boxes"] ]
        dtype=ob["dtype"]
        nComponents=ob["nComponents"]

        return level(geo,boxes,dtype=dtype,nComponents=nComponents )



    def __init__(self,geo,boxes,dtype="real",nComponents=1):
        self.geo=geo
        self.boxes=boxes
        self.nComponents=int(nComponents)
        self.data=[]
        for currentBox in boxes:
            self.data.append(np.zeros(currentBox.shape) )

        cppBoxes= [ curBox.cpp() for curBox in boxes ]
        self.nRealComponents=int(nComponents)
        if (dtype=="complex"):
            self.nRealComponents=2*self.nRealComponents
        

        levelType = cppLevelType(dtype)
    


        self._levelCpp= levelType( geo.cpp(),cppBoxes, self.nRealComponents )
        self._dtype=dtype
    

    def toYAML(self):
        ob={ "geo" : self.geo.toYAML() , 
            "boxes" : [b.toYAML() for b in self.boxes ] ,   
            "dtype" : self.dtype,
            "nComponents" : self.nComponents
            }

        return ob


    @property
    def dtype(self ):
        return self._dtype
    

    def cpp(self):
        return self._levelCpp
    

    def __getitem__(self, i):
        datas=[]
        for c in range(self.nRealComponents):
            data=self.cpp().getData(self.boxes[i].cpp() ,c  )
            datas.append(data)
        
        if ( self.dtype=="real"):
            return (datas)        
        

        if ( self.dtype=="complex"):
            datas_complex=[]
            for c in range( self.nRealComponents//2 ):
                datas_complex.append(datas[2*c] + 1j * datas[2*c+1] )
            return ( datas_complex )
        
    
    def __setitem__(self, i,data):
        cBox=self.boxes[i]
        datas=data
        if len(data) != self.nComponents:
            raise RuntimeError("Should be equal to a list of arrays of length the number of components")
        

        if self.dtype == "complex":
                real_data=[]
                for c in range( self.nComponents ):
                    real_data.append( np.real(data[c] ) )
                    real_data.append( np.imag(data[c] ) )
                
                datas=real_data
        

        for c in range(0,len(datas)):
            
            self.cpp().setData(datas[c], self.boxes[i].cpp() , c )
        

class field:

    @staticmethod
    def h5ToBox( hdf5Box  ):
        dims=len(hdf5Box)//2
        extent=[ [ int(hdf5Box[i]),int(hdf5Box[dims+i]) ]    for i in range(dims) ]
        return( box(extent))



    @staticmethod
    def fromYAML(ob):
        levels=[  level.fromYAML(l) for l in ob["levels"]  ]
        return field( levels)

    @staticmethod
    def load(dirname):
        yaml=YAML()
        path=pathlib.Path(dirname) / "field.yaml"
        with open(path) as f:
            ob=yaml.load(f )

        field2=field.fromYAML(ob)
        field2.loadData(dirname)
        return field2
    

    def loadData(self,dirname):
        datas= [ [] for c in range( len(self))  ] 
        for c in range( self.nRealComponents ):
            phiPath=pathlib.Path( dirname ) / "phi{:d}.h5".format(c)
            f=h5py.File(phiPath)

            for lev in range(len(self)):

                    key_level="level_{:d}".format(lev)
                    boxes=[ field.h5ToBox(b) for b in f[key_level]["boxes"] ]
                    data= np.array(f[key_level]["data:datatype=0"]).reshape(np.flip(boxes[0].shape)).transpose()


                    datas[lev].append(data)
        

        if self.dtype == "complex":
            for lev in range( len(self) ):
                datas[lev]=  [ datas[lev][2*c] + 1j*datas[lev][2*c+1]      for c in range(self.nComponents) ]
        
        for lev in range( len(self) ):
            for c in range(self.nComponents):
                self[lev][0]=datas[lev]
    
        


    

        




    def __init__(self, levels):
        self._levels=levels
        levelsCpp=[level.cpp() for level in levels]
        self._dtype=levels[0].dtype

    
        if levels[0].dtype=="real":
            self._fieldCpp=gpAmreX.realField(levelsCpp)
        else:
            if levels[0].dtype=="complex":
                self._fieldCpp=gpAmreX.complexField(levelsCpp)
            else:
                raise RuntimeError("type of levels should be real or complex")
    

    def toYAML(self):
        return {"levels" : [ l.toYAML() for l in self._levels ]   }
    
    

    def averageDown(self):
        self.cpp().averageDown()
    

    def save(self,dirname):
        self.cpp().save(dirname)
        ob=self.toYAML( )
        
        yaml.dump( ob, pathlib.Path(dirname) / "field.yaml" )






    
    def __len__(self):
        return len(self._levels)
    


    def cpp(self):
        return self._fieldCpp
    def __getitem__(self,i):
        return self._levels[i]
    


    
    @property
    def time(self):
        return self.cpp().getTime()
    
    @property
    def norm(self):
        return self.cpp().getNorm()
    
    @norm.setter
    def norm(self,N):
        for c in range(len(N)):
            self.cpp().normalize(N[c],c)
    @property
    def dtype(self):
        return self._dtype
    

    @property
    def nComponents(self):
        
        return self[0].nComponents


    @property
    def nRealComponents(self):
        return self[0].nRealComponents
    

    


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
    def __init__(self,func,realTime=False,kind="eulero"):
        self._stepperCpp=gpAmreX.stepper( func.cpp() , kind )
        self._realTime=realTime
        self._kind=kind
    def define(self,wave):
        self.cpp().define(wave.cpp())
    
    def enableNormalization(self,N):
        self._N=N
        self.cpp().addNormalizationConstraint(N)
    
    





    def cpp(self):
        return self._stepperCpp
    
    def advance(self,phiOld,phiNew,dt ):

        if phiOld.dtype=="real":
            self.cpp().advance( phiOld.cpp(), phiNew.cpp(), dt )
        else:
            if not self._realTime:
                self.cpp().advanceImaginaryTime( phiOld.cpp(), phiNew.cpp(), dt )
            else:
                self.cpp().advanceRealTime( phiOld.cpp(), phiNew.cpp(), dt )

    
    
    
        

def grid( box,geo):
    dims=len(box.shape)
    x=[]
    for d in range(dims):
        x.append( (  np.arange(box.left[d],box.right[d] + 1 ,1 ) + 0.5)*geo.cellSize[d] + geo.left[d] )
    
    return np.meshgrid(*x,indexing="ij")
