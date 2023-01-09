import pybullet as p
import time
import pybullet_data
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,10]
startOrientation = p.getQuaternionFromEuler([0,0,0])
#boxId = p.loadURDF("cube.urdf")
boxId = p.loadSoftBody("lego_9.obj",
        startPos,  
        useMassSpring=1,
        scale=10,
        useBendingSprings=1,
        mass = 0.1,
        springElasticStiffness =1,
        springBendingStiffness=10,
        frictionCoeff=1,
        useFaceContact=1,
        repulsionStiffness=100
)

bp = [[0.22,0.2,1],[-0.3,-0.22,0.8],[0.2,-0.2,0.8],[-0.2,0.2,1]]
bs = [0.5,0.6,0.5,0.5]

for i in range (10000):
    if i%1000 == 0 and i < 4000 and 0:
        ballId = p.loadSoftBody("cube.obj", 
                basePosition = bp[i//1000], 
                scale = bs[i//1000], 
                mass = 1, 
                useNeoHookean=1,
                useMassSpring=1, 
                useBendingSprings=1, 
                useSelfCollision = 1, 
                frictionCoeff = .5, 
                springElasticStiffness = 0.5, 
                springDampingStiffness=0.5, 
                repulsionStiffness=10000)
    p.stepSimulation()
    time.sleep(1./480.)
print(cubePos,cubeOrn)
p.disconnect()

