from cmath import sqrt
from functools import reduce
import gym
import gym.spaces
import numpy
import random

'''
only implementing for four way intersection each with two lanes at the moment.

There will be 4 lanes for choosing traffic lights 
'''

TOTAL_NUMBER_OF_LANES = 8
NUMBER_OF_LANES_TO_OBSERVE = TOTAL_NUMBER_OF_LANES/2 # the lane number is in anticlockwise order skipping one lane
LANES_CAPACITY = [100, 75, 80, 88]  # capacity of the corresponding lane
PROBABILITY_OF_VEHICLE_OCCURANCE_IN_LANES_PER_SECOND = [0.2, 0.11, 0.5, 0.6]

'''
vehicles in those lanes will move with this speed (in km/hr ) if green light is on
this is the average speed each vehicle will take when passing (assuming ideal case)
'''
VEHICLE_SPEED_OF_LANES = [14, 13, 15, 8]  

MIN_GREEN_TIME = 30   # This corresponds to minimum green light time (in seconds) which is equivalent to one step in our environment

''' 
The possible configuration in 4 way, two lanes intersection is 4 

            |     |      |
            |     |      |
            |  ^  |   |  |
            |  |  |   V  |
            |     |      |
------------              -------------
     ->                         ->
------------              -------------
    <-                          <-
------------              --------------
            |     |      |
            |     |      |
            |  ^  |   |  |
            |  |  |   V  |
            |     |      |


The possible green configurations are:

0)


            |     |      |
            |     |      |
            |     |      |
            |     |      |
            |     |      |
------------              -------------
     ->                         ->          lane 0 horizontal right(up)
------------              -------------     lane 0 vertical up (left)
    <-                          <-          lane 2 horizontal left(down)    
------------              --------------    lane 2 vertical down (right)
            |     |      |
            |     |      |
            |     |      |
            |     |      |
            |     |      |

1)


            |     |      |
            |     |      |
            |  ^  |      |
            |  |  |      |
            | ii  |      |
------------              -------------
     ->  i             
------------              -------------   lane 0 vertical down (right)
                            ii   <-       lane 2 vertical up (left)  
------------              --------------  lane 3 horizontal left (up)  
            |     |      |                lane 1 horizontal right (down)  
            |     |   i  |
            |     |      |
            |     |   |  |
            |     |   V  |


2)

            |     |      |
            |     |   |  |
            |     |   V  |
            |     |      |
            |     |   ii |
------------              ------------- lane 0 vertical up (left)
                            i  - >      lane 1 horizontal left (down)  
------------               ------------ lane 2 vertical down(left) 
    <-   ii                             lane 3 horizontal right (up)
------------               -------------
            |     |      |
            |  i  |      |
            |  ^  |      |
            |  |  |      |
            |     |      |

3)

            |     |      |
            |     |      |
            |  ^  |   |  |
            |  |  |   V  |
            |     |      |
------------              ------------- lane 3 vertical up (left)
                                        lane 1 vertical down (right)
------------              ------------- lane 3 horizontal left (down)
                                        lane 1 horizontal right (up)
------------              --------------
            |     |      |
            |     |      |
            |  ^  |   |  |
            |  |  |   V  |
            |     |      |



'''

POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION = 4

LENGTH_STRAIGHT_IN_METERS = 100
LENGTH_DIAGONAL_IN_METERS = LENGTH_STRAIGHT_IN_METERS * sqrt(2)

PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES = 0.3
PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY = 0.4
ONE_TRAINING_TIME = 5 * 60 # Train for the equivalent of 5 minutes

class TrafficIntersectionEnv(gym.Env):

    def __init__(self):

        # initialising gym stuffs
        
        '''
        action space
        The action which can be performed is changing the current active traffic light configuration
        '''
        self.action_space = gym.spaces.Discrete(POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION)

        # observation space
        # Observation space is the vehicle count in different lanes which need to be observed 
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(int(NUMBER_OF_LANES_TO_OBSERVE), ))

        # keeping track of the reward
        self.collected_reward = -1


        # initialising our environment

        # each element of array corresponds to vehicle count in the corresponding lane
        self.lanes = numpy.zeros(int(NUMBER_OF_LANES_TO_OBSERVE))

        # randomising vehicle count in lanes
        for i, _ in enumerate(self.lanes):
            self.lanes[i] = random.randint(int(0.1 * LANES_CAPACITY[i]), LANES_CAPACITY[i])

        self.state = self.lanes

        self.remaining_time = ONE_TRAINING_TIME

        
    def step(self, action):
        '''
        The possible action is to change the traffic light configuration.
        If configuration is set to the same one, yellow light won't turn on. 
        '''

        done = False
        info = {}

        self.remaining_time -= MIN_GREEN_TIME

        # not implemented yet
        '''
        turnOnYellowLight = True
        if self.state == action:
            turnOnYellowLight = False
        '''

        self.state = action

        vehicleThroughIntersection, vehicleRemaining = self.simulateTraffic()

        if vehicleThroughIntersection==0 or self.remaining_time < 60:
            done = True
        
        reward = self.calculateReward(vehicleThroughIntersection, vehicleRemaining)
        self.collected_reward += reward

        obs = self.lanes

        return obs, self.collected_reward, done, info


    def reset(self):
        for i, _ in enumerate(self.lanes):
            self.lanes[i] = random.randint(int(0.1 * LANES_CAPACITY[i]), LANES_CAPACITY[i])

        self.collected_reward = -1
        self.state = self.lanes

        return self.state

    def simulateTraffic(self):

        trafficLightConfiguration = self.state
        vehicleThroughIntersection = 0
        vehicleRemaining = reduce(lambda x,y: x+y, self.lanes)

        # reward calculation based on number of vehicle passed
        
        if trafficLightConfiguration==0:
           # with average speed, the time taken to travel the gap

           # lane 0     Assuming the round path is one tenth length of straight path
           #            and vehicle speed in turning is 70 percent of straight speed
           timeStraightLane0 = LENGTH_STRAIGHT_IN_METERS / VEHICLE_SPEED_OF_LANES[0] * (10 / 36)
           timeTurningLane0 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[0] * (10 / 36)

           # lane 2     Assuming the round path is one tenth length of straight path
           #            and vehicle speed in turning is 70 percent of straight speed
           timeStraightLane2 = LENGTH_STRAIGHT_IN_METERS / VEHICLE_SPEED_OF_LANES[2] * (10 / 36)
           timeTurningLane2 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[2] * (10 / 36)

           '''
           the vehicles wanting to go straight are the vehicles
           not wanting to go through side lanes and not wanting to go diagonally
           '''
           
           vehicleTurning0 = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[0]
           vehicleTurning2 = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[2]
           vehicleStraight0 = self.lanes[0] - vehicleTurning0 - PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[0]
           vehicleStraight2 = self.lanes[2] - vehicleTurning2 - PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[2]

           # max possible vehicle able to pass within given time
           maxVehiclePassingStraightLane0 = MIN_GREEN_TIME / timeStraightLane0
           maxVehiclePassingTurningLane0 = MIN_GREEN_TIME / timeTurningLane0
           maxVehiclePassingStraightLane2 = MIN_GREEN_TIME / timeStraightLane2
           maxVehiclePassingTurningLane2 = MIN_GREEN_TIME / timeTurningLane2
        
           # actual vehicle passing
           actualVehiclePassingStraight0 = max(vehicleStraight0 - maxVehiclePassingStraightLane0, vehicleStraight0)
           actualVehiclePassingStraight2 = max(vehicleStraight2 - maxVehiclePassingStraightLane2, vehicleStraight2)
           actualVehiclePassingTurning0 = max(vehicleTurning0 - maxVehiclePassingTurningLane0, vehicleTurning0)
           actualVehiclePassingTurning2 = max(vehicleTurning2 - maxVehiclePassingTurningLane2, vehicleTurning2)

           #updating the vehicles in lanes
           self.lanes[0] = int(self.lanes[0] - actualVehiclePassingStraight0 - actualVehiclePassingTurning0)
           self.lanes[2] = int(self.lanes[2] - actualVehiclePassingStraight2 - actualVehiclePassingTurning2)
           
           vehicleThroughIntersection = int(actualVehiclePassingStraight0 + actualVehiclePassingTurning0 + actualVehiclePassingStraight2 + actualVehiclePassingTurning2)
           vehicleRemaining = int(reduce(lambda x,y: x+y, self.lanes) - vehicleThroughIntersection)


        elif trafficLightConfiguration==1:
            # with average speed, the time taken to travel the gap

           #    Assuming the round path speed is nine tenth  of straight path speed
           timeDiagonalLane0 = LENGTH_DIAGONAL_IN_METERS / 0.9 * VEHICLE_SPEED_OF_LANES[0] * (10 / 36)
           timeTurningLane1 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[1] * (10 / 36)

           #     Assuming the round path speed is nine tenth of straight path speed
           timeDiagonalLane2 = LENGTH_DIAGONAL_IN_METERS / 0.9 * VEHICLE_SPEED_OF_LANES[2] * (10 / 36)
           timeTurningLane3 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[3] * (10 / 36)

           vehicleDiagonal0 = self.lanes[0] * PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY
           vehicleTurning1 = self.lanes[1] * PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES
           vehicleDiagonal2 = self.lanes[2] * PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY
           vehicleTurning3 = self.lanes[3] * PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES

           # max possible vehicle able to pass within given time
           maxVehiclePassingDiagonalLane0 = MIN_GREEN_TIME / timeDiagonalLane0
           maxVehiclePassingTurningLane1 = MIN_GREEN_TIME / timeTurningLane1
           maxVehiclePassingDiagonalLane2 = MIN_GREEN_TIME / timeDiagonalLane2
           maxVehiclePassingTurningLane3 = MIN_GREEN_TIME / timeTurningLane3
        
           # actual vehicle passing
           actualVehiclePassingDiagonal0 = max(vehicleDiagonal0 - maxVehiclePassingDiagonalLane0, vehicleDiagonal0)
           actualVehiclePassingDiagonal2 = max(vehicleDiagonal2 - maxVehiclePassingDiagonalLane2, vehicleDiagonal2)
           actualVehiclePassingTurning1 = max(vehicleTurning1 - maxVehiclePassingTurningLane1, vehicleTurning1)
           actualVehiclePassingTurning3 = max(vehicleTurning3 - maxVehiclePassingTurningLane3, vehicleTurning3)

           #updating the vehicles in lanes
           self.lanes[0] = int(self.lanes[0] - actualVehiclePassingDiagonal0)
           self.lanes[2] = int(self.lanes[2] - actualVehiclePassingDiagonal2)
           self.lanes[1] = int(self.lanes[1] - actualVehiclePassingTurning1)
           self.lanes[3] = int(self.lanes[3] - actualVehiclePassingTurning3)
           
           vehicleThroughIntersection = int(actualVehiclePassingDiagonal0 + actualVehiclePassingDiagonal2 + actualVehiclePassingTurning1 + actualVehiclePassingTurning3)
           vehicleRemaining = int(reduce(lambda x,y: x+y, self.lanes) - vehicleThroughIntersection)

        elif trafficLightConfiguration==2:
            # with average speed, the time taken to travel the gap

           
           timeTurningDownLane1 = LENGTH_DIAGONAL_IN_METERS / VEHICLE_SPEED_OF_LANES[1] * (10 / 36)
           timeTurningUpLane0 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[0] * (10 / 36)

           timeTurningUpLane3 = LENGTH_DIAGONAL_IN_METERS / VEHICLE_SPEED_OF_LANES[3] * (10 / 36)
           timeTurningDownLane2 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[2] * (10 / 36)
           
           vehicleTurning0 = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[0]
           vehicleTurning2 = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[2]
           vehicleTurningDiagonal1 = PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[1]
           vehicleTurningDiagonal3 = PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[3]

           # max possible vehicle able to pass within given time
           maxVehiclePassingDiagonalLane1 = MIN_GREEN_TIME / timeTurningDownLane1
           maxVehiclePassingTurningLane0 = MIN_GREEN_TIME / timeTurningUpLane0
           maxVehiclePassingDiagonalLane3 = MIN_GREEN_TIME / timeTurningUpLane3
           maxVehiclePassingTurningLane2 = MIN_GREEN_TIME / timeTurningDownLane2
        
           # actual vehicle passing
           actualVehiclePassingDiagonal1 = max(vehicleTurningDiagonal1 - maxVehiclePassingDiagonalLane1, vehicleTurningDiagonal1)
           actualVehiclePassingDiagonal3 = max(vehicleTurningDiagonal3 - maxVehiclePassingDiagonalLane3, vehicleTurningDiagonal3)
           actualVehiclePassingTurning0 = max(vehicleTurning0 - maxVehiclePassingTurningLane0, vehicleTurning0)
           actualVehiclePassingTurning2 = max(vehicleTurning2 - maxVehiclePassingTurningLane2, vehicleTurning2)

           #updating the vehicles in lanes
           self.lanes[0] = int(self.lanes[0] - actualVehiclePassingTurning0)
           self.lanes[2] = int(self.lanes[2] - actualVehiclePassingTurning2)
           self.lanes[1] = int(self.lanes[1] - actualVehiclePassingDiagonal1)
           self.lanes[3] = int(self.lanes[3] - actualVehiclePassingDiagonal3)
           
           vehicleThroughIntersection = int(actualVehiclePassingTurning0 + actualVehiclePassingDiagonal1 + actualVehiclePassingTurning2 + actualVehiclePassingDiagonal3)
           vehicleRemaining = int(reduce(lambda x,y: x+y, self.lanes) - vehicleThroughIntersection)

           # adding vehicles to the lanes according to the probability
           for j,probability in enumerate(PROBABILITY_OF_VEHICLE_OCCURANCE_IN_LANES_PER_SECOND):
            self.lanes[j] += int(probability * MIN_GREEN_TIME)

        elif trafficLightConfiguration==3:
            # with average speed, the time taken to travel the gap

           # lane 1     Assuming the round path is one tenth length of straight path
           #            and vehicle speed in turning is 70 percent of straight speed
           timeStraightLane1 = LENGTH_STRAIGHT_IN_METERS / VEHICLE_SPEED_OF_LANES[1] * (10 / 36)
           timeTurningLane1 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[1] * (10 / 36)

           # lane 3     Assuming the round path is one tenth length of straight path
           #            and vehicle speed in turning is 70 percent of straight speed
           timeStraightLane3 = LENGTH_STRAIGHT_IN_METERS / VEHICLE_SPEED_OF_LANES[3] * (10 / 36)
           timeTurningLane3 = 0.1 * LENGTH_STRAIGHT_IN_METERS / 0.7 * VEHICLE_SPEED_OF_LANES[3] * (10 / 36)

           '''
           the vehicles wanting to go straight are the vehicles
           not wanting to go through side lanes and not wanting to go diagonally
           '''
           
           vehicleTurning1 = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[1]
           vehicleTurning3 = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[1]
           vehicleStraight1 = self.lanes[1] - vehicleTurning1 - PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[1]
           vehicleStraight3 = self.lanes[3] - vehicleTurning3 - PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[3]

           # max possible vehicle able to pass within given time
           maxVehiclePassingStraightLane1 = MIN_GREEN_TIME / timeStraightLane1
           maxVehiclePassingTurningLane1 = MIN_GREEN_TIME / timeTurningLane1
           maxVehiclePassingStraightLane3 = MIN_GREEN_TIME / timeStraightLane3
           maxVehiclePassingTurningLane3 = MIN_GREEN_TIME / timeTurningLane3
        
           # actual vehicle passing
           actualVehiclePassingStraight1 = max(vehicleStraight1 - maxVehiclePassingStraightLane1, vehicleStraight1)
           actualVehiclePassingStraight3 = max(vehicleStraight3 - maxVehiclePassingStraightLane3, vehicleStraight3)
           actualVehiclePassingTurning1 = max(vehicleTurning1 - maxVehiclePassingTurningLane1, vehicleTurning1)
           actualVehiclePassingTurning3 = max(vehicleTurning3 - maxVehiclePassingTurningLane3, vehicleTurning3)

           #updating the vehicles in lanes
           self.lanes[1] = int(self.lanes[1] - actualVehiclePassingStraight1 - actualVehiclePassingTurning1)
           self.lanes[3] = int(self.lanes[3] - actualVehiclePassingStraight3 - actualVehiclePassingTurning3)
           
           vehicleThroughIntersection = int(actualVehiclePassingStraight1 + actualVehiclePassingTurning1 + actualVehiclePassingStraight3 + actualVehiclePassingTurning3)
           vehicleRemaining = int(reduce(lambda x,y: x+y, self.lanes) - vehicleThroughIntersection) 


        return vehicleThroughIntersection, vehicleRemaining

    # dummy reward function for now
    def calculateReward(self, numberOfVehiclePassed, numberOfVehicleRemaining):
        '''
        reward should be proportional to number of vehicle passed through the intersection
        '''
        reward = (numberOfVehiclePassed) / (numberOfVehiclePassed + numberOfVehicleRemaining) * 100
        if reward >= 50:
            return reward
        else:
            return reward * -1
