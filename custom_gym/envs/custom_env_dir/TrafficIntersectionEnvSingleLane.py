from functools import reduce
import gym
import gym.spaces
import numpy
import random


'''
This models a type of intersection in which vechicle in a lane can move if corresponding 
traffic light is green i.e. if light 1 is green then vehicle of lane 1 is free to move anywhere but 
all other lanes is stopped. Such system will only have one lane for each direction (left and right)
'''


TOTAL_NUMBER_OF_LANES = 8
NUMBER_OF_LANES_TO_OBSERVE = TOTAL_NUMBER_OF_LANES/2 # the lane number is in anticlockwise order skipping one lane
LANES_CAPACITY = [100, 75, 80, 88]  # capacity of the corresponding lane
PROBABILITY_OF_VEHICLE_OCCURANCE_IN_LANES_PER_SECOND = [0.2, 0.11, 0.29, 0.4]

'''
vehicles in those lanes will move with this speed (in km/hr ) if green light is on
this is the average speed each vehicle will take when passing (assuming ideal case)
'''
VEHICLE_SPEED_OF_LANES = [14, 13, 15, 8]  

MIN_GREEN_TIME = 60   # This corresponds to minimum green light time (in seconds) which is equivalent to one step in our environment

''' 
The possible configuration in 4 way, two lanes intersection is 4 

            |     |      |
            |     |      |
            |     |      |
            |     |      |
            |     |      |
------------              -------------
 
------------              -------------

------------              --------------
            |     |      |
            |     |      |
            |     |      |
            |     |      |
            |     |      |


The possible green configurations are:

0)


            |     |      |
            |     |      |
            |  ^  |      |
            |  |  |      |
            |     |      |
------------              -------------
     ->                         ->          
------------              -------------     
                                               
------------              --------------    
            |     |      |
            |     |      |
            |     |   |  |
            |     |   V  |
            |     |      |

1)


            |     |      |
            |     |      |
            |     |   |  |
            |     |   V  |
            |     |      |
------------              -------------
                                ->
------------              -------------   
       <-
------------              --------------    
            |     |      |                 
            |     |      |
            |     |      |
            |     |   |  |
            |     |   V  |


2)

            |     |      |
            |     |      |
            |  ^  |      |
            |  |  |      |
            |     |      |
------------              ------------- 
                                  
------------               ------------ 
    <-                        <-       
------------               -------------
            |     |      |
            |     |   |  |
            |     |   V  |
            |     |      |
            |     |      |

3)

            |     |      |
            |     |      |
            |  ^  |      |
            |  |  |      |
            |     |      |
------------              ------------- 
                             ->           
------------              ------------- 
        <-                                    
------------              --------------
            |     |      |
            |     |      |
            |  ^  |      |
            |  |  |      |
            |     |      |



'''

POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION = 4

LENGTH_STRAIGHT_IN_METERS = 15
LENGTH_DIAGONAL_IN_METERS = LENGTH_STRAIGHT_IN_METERS * 1.41

PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES = 0.3
PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY = 0.2
ONE_TRAINING_TIME = 5 * 60 # Train for the equivalent of 5 minutes

class TrafficIntersectionEnvSingleLane(gym.Env):

    def __init__(self):

        # initialising gym stuffs
        
        '''
        action space
        The action which can be performed is changing the current active traffic light configuration
        '''
        self.action_space = gym.spaces.Discrete(POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION)

        # observation space
        # Observation space is the vehicle count in different lanes which need to be observed 
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(int(NUMBER_OF_LANES_TO_OBSERVE), ), dtype=numpy.float64)

        # keeping track of the reward
        # self.collected_reward = -1

        self.lanes = numpy.zeros(int(NUMBER_OF_LANES_TO_OBSERVE))

        # initializing our environment
        self.reset()

        
    def step(self, action):
        '''
        The possible action is to change the traffic light configuration.
        If configuration is set to the same one, yellow light shouldn't turn on. 
        '''

        done = False
        info = {}

        self.remaining_time -= MIN_GREEN_TIME

        # copying the observation to use for reward generation. since, observation will be changed after traffic is simulated
        last_observation_leading_to_predicted_action = self.state.copy()

        vehicleThroughIntersection, vehicleRemaining = self.simulateTraffic(action)

        if vehicleThroughIntersection==0 or self.remaining_time < MIN_GREEN_TIME:
            done = True
        
        reward = self.calculateReward(vehicleThroughIntersection, vehicleRemaining, last_observation_leading_to_predicted_action)
        #self.collected_reward += reward

        obs = self.lanes
        self.state = self.lanes

        if done==True:
            self.reset()

        return obs, reward, done, info


    def reset(self):

        # randomising vehicle count in lanes
        for i in range(int(NUMBER_OF_LANES_TO_OBSERVE)):
            self.lanes[i] = random.randint(int(0.1 * LANES_CAPACITY[i]), LANES_CAPACITY[i])

        #self.collected_reward = -1
        self.state = self.lanes

        self.percentageOfVehiclePassingThroughTheIntersectionLastTime = 0

        self.remaining_time = ONE_TRAINING_TIME

        return self.state

    def simulateTraffic(self, action):

        trafficLightConfiguration = action
        vehicleThroughIntersection = 0
        totalVehicle = reduce(lambda x,y: x+y, self.lanes)

        # reward calculation based on number of vehicle passed
        
        # since vehicles can move randomly (in the real world), we can set any type of logic here, but keeping it a bit realistic

        vehiclesTurning = PERCENTAGE_OF_VEHICLE_GOING_THROUGH_SIDE_LANES * self.lanes[trafficLightConfiguration]
        vehiclesTurningDiagonally = PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY * self.lanes[trafficLightConfiguration]
        vehiclesGoingStraight = self.lanes[trafficLightConfiguration] - vehiclesTurning - vehiclesTurningDiagonally

        timeTakenForTurning = 0.1 * LENGTH_STRAIGHT_IN_METERS / (0.7 * VEHICLE_SPEED_OF_LANES[trafficLightConfiguration] * (10/36))
        timeTakenForDiagonalTurning = LENGTH_DIAGONAL_IN_METERS / (VEHICLE_SPEED_OF_LANES[trafficLightConfiguration]* (10/36))
        timeTakenStraight = LENGTH_STRAIGHT_IN_METERS / (VEHICLE_SPEED_OF_LANES[trafficLightConfiguration]* (10/36))

        maxVehicleTurning = MIN_GREEN_TIME / timeTakenForTurning
        maxVehicleTurningDiagonally = MIN_GREEN_TIME / timeTakenForDiagonalTurning
        maxVehicleStraight = MIN_GREEN_TIME / timeTakenStraight

        actualVehicleTurning = min(maxVehicleTurning, vehiclesTurning)
        actualVehicleTurningDiagonally = min(maxVehicleTurningDiagonally, vehiclesTurningDiagonally)
        actualVehicleGoingStraight = min(maxVehicleStraight, vehiclesGoingStraight)
        
        vehicleThroughIntersection = int(actualVehicleTurning + actualVehicleTurningDiagonally + actualVehicleGoingStraight)

        self.lanes[trafficLightConfiguration] -= vehicleThroughIntersection
        vehicleRemaining = totalVehicle - vehicleThroughIntersection
        
        # adding vehicles to the lanes according to the probability
        for j,probability in enumerate(PROBABILITY_OF_VEHICLE_OCCURANCE_IN_LANES_PER_SECOND):
            self.lanes[j] += int(probability * MIN_GREEN_TIME)


        return vehicleThroughIntersection, vehicleRemaining

    # dummy reward function for now
    # this doesn't work
    def calculateReward(self, numberOfVehiclePassed, numberOfVehicleRemaining, last_observation_leading_to_predicted_action):

        reward = numberOfVehiclePassed / max(last_observation_leading_to_predicted_action)

        return reward