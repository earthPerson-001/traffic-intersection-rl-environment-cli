from cmath import sqrt
from functools import reduce
from sys import maxunicode
from time import time
import gym
import gym.spaces
import numpy
import random

'''
only implementing for four way intersection each with two lanes at the moment.

There will be 4 lanes for choosing traffic lights 
'''

TOTAL_NUMBER_OF_LANES = 16
NUMBER_OF_LANES_TO_OBSERVE = TOTAL_NUMBER_OF_LANES/2 # the lane number is in anticlockwise order skipping one lane
LANES_CAPACITY = [100, 75, 80, 88, 73, 87, 78, 91]  # capacity of the corresponding lane
PROBABILITY_OF_VEHICLE_OCCURANCE_IN_LANES_PER_SECOND = [0.15, 0.25, 0.22, 0.5, 0.8, 0.86, 0.77, 0.67]

'''
vehicles in those lanes will move with this speed (in km/hr ) if green light is on
this is the average speed each vehicle will take when passing (assuming ideal case)
'''
VEHICLE_SPEED_OF_LANES = [14, 13, 15, 8, 12, 11, 9.5, 12.5]  

MIN_GREEN_TIME = 60   # This corresponds to minimum green light time (in seconds) which is equivalent to one step in our environment

''' 
The possible configuration in 4 way, 4 lanes intersection is 4 

            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
------------                            -------------

------------                            -------------

-_-_-_-_-_-_-                           -_-_-_-_-_-_-_-

------------                            -------------
     
------------                            --------------
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |


The possible green configurations are:

0)

            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
------------                            -------------

------------                            -------------

-_-_-_-_-_-_-                           -_-_-_-_-_-_-_-

------------                            -------------
     
------------                            --------------
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |

1)


            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
------------                            -------------

------------                            -------------

-_-_-_-_-_-_-                           -_-_-_-_-_-_-_-

------------                            -------------
     
------------                            --------------
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |


2)

            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
------------                            -------------

------------                            -------------

-_-_-_-_-_-_-                           -_-_-_-_-_-_-_-

------------                            -------------
     
------------                            --------------
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |

3)

 
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
------------                            -------------

------------                            -------------

-_-_-_-_-_-_-                           -_-_-_-_-_-_-_-

------------                            -------------
     
------------                            --------------
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |
            |     |      ||     |       |



'''

POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION = 4

LENGTH_STRAIGHT_IN_METERS = 15
LENGTH_DIAGONAL_IN_METERS = LENGTH_STRAIGHT_IN_METERS * 1.41

PERCENTAGE_OF_VEHICLE_GOING_Left = 0.3
PERCENTAGE_OF_VEHICLE_GOING_DIAGONALLY = 0.2
PERCENTAGE_OF_VEHICLE_GOING_STRAIGHT = 0.5
ONE_TRAINING_TIME = 5 * 60 # Train for the equivalent of 5 minutes

class TrafficIntersectionEnvDoubleLane(gym.Env):

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

        # Resetting our environment, rather resetting it
        self.reset()

        
    def step(self, action):
        '''
        The possible action is to change the traffic light configuration.
        If configuration is set to the same one, yellow light won't turn on. 
        '''

        done = False
        info = {}

        self.remaining_time -= MIN_GREEN_TIME

        vehicleThroughIntersection, vehicleRemaining = self.simulateTraffic(action)

        # copying the observation to use for reward generation. since, observation will be changed after traffic is simulated
        last_observation_leading_to_predicted_action = self.state.copy()

        if vehicleThroughIntersection==0 or self.remaining_time < 60:
            done = True
            self.reset()
        
        reward = self.calculateReward(vehicleThroughIntersection, vehicleRemaining, last_observation_leading_to_predicted_action)
        # self.collected_reward += reward

        obs = self.lanes
        self.state = self.lanes

        if done==True:
            self.reset()

        return obs, reward, done, info


    def reset(self):
        for i, _ in enumerate(self.lanes):
            self.lanes[i] = random.randint(int(0.3 * LANES_CAPACITY[i]), LANES_CAPACITY[i])

        # self.collected_reward = -1
        self.state = self.lanes

        self.remaining_time = ONE_TRAINING_TIME

        self.percentageOfVehiclePassingThroughTheIntersectionLastTime = 0

        return self.state

    def simulateTraffic(self, action):

        trafficLightConfiguration = action
        vehicleThroughIntersection = 0
        totalVehicle = reduce(lambda x,y: x+y, self.lanes)

        # reward calculation based on number of vehicle passed
        
        if trafficLightConfiguration==0:
            first_lanes_to_subtract_certain_percentage_traffic = [0,4]
            second_lanes_to_subtract_certain_percentage_traffic = [1,5]
            percentage_to_subtract = [1,0.5]

           
        elif trafficLightConfiguration==1:
            first_lanes_to_subtract_certain_percentage_traffic = [2,6]
            second_lanes_to_subtract_certain_percentage_traffic = [3,7]
            percentage_to_subtract = [1,0.5]


        elif trafficLightConfiguration==2:
            first_lanes_to_subtract_certain_percentage_traffic = [0,4]
            second_lanes_to_subtract_certain_percentage_traffic = [1,5]
            percentage_to_subtract = [0.3,0.2]

            
        elif trafficLightConfiguration==3:
            first_lanes_to_subtract_certain_percentage_traffic = [2,6]
            second_lanes_to_subtract_certain_percentage_traffic = [3,7]
            percentage_to_subtract = [0.3,0.2]
        
        else:
            first_lanes_to_subtract_certain_percentage_traffic = []
            second_lanes_to_subtract_certain_percentage_traffic = []
            percentage_to_subtract = []
        
        if (len(first_lanes_to_subtract_certain_percentage_traffic) == 0) or (len(second_lanes_to_subtract_certain_percentage_traffic) == 0) or (len(percentage_to_subtract)==0):
            return 0,0
        
        no_of_vehicle_to_remove_in_each_lanes = numpy.zeros(len(self.lanes))
        lane_arrays = [first_lanes_to_subtract_certain_percentage_traffic, second_lanes_to_subtract_certain_percentage_traffic]
        for j in range(len(lane_arrays)):
            for i in range(len(lane_arrays[j])):
                no_of_vehicle_to_remove_in_each_lanes[lane_arrays[j][i]] = percentage_to_subtract[j] * self.lanes[lane_arrays[j][i]]

        speed_of_vehicle_removal_in_each_lane = numpy.zeros(len(no_of_vehicle_to_remove_in_each_lanes))
        for i in range(len(speed_of_vehicle_removal_in_each_lane)):
            if(no_of_vehicle_to_remove_in_each_lanes[i] != 0) and i%2==0:
                speed_of_vehicle_removal_in_each_lane[i] = 0.35 * VEHICLE_SPEED_OF_LANES[i] * (10/36) # taking average between straight speed and turning speed
            elif (no_of_vehicle_to_remove_in_each_lanes[i] != 0) and i%2!=0:
                speed_of_vehicle_removal_in_each_lane[i] = 0.85 * VEHICLE_SPEED_OF_LANES[i] * (10/36) # taking average between straight speed and diagonal speed
            else:
                speed_of_vehicle_removal_in_each_lane[i] = 0

        time_required_for_one_vehicle_removal_in_each_lane = numpy.zeros(len(no_of_vehicle_to_remove_in_each_lanes))
        for i in range(len(speed_of_vehicle_removal_in_each_lane)):
            if(no_of_vehicle_to_remove_in_each_lanes[i] != 0) and i%2==0:
                time_required_for_one_vehicle_removal_in_each_lane[i] = 0.9 * LENGTH_STRAIGHT_IN_METERS / speed_of_vehicle_removal_in_each_lane[i] # taking average between straight distance and turning distance
            elif (no_of_vehicle_to_remove_in_each_lanes[i] != 0) and i%2!=0:
                time_required_for_one_vehicle_removal_in_each_lane[i] = 1.205 * LENGTH_DIAGONAL_IN_METERS / speed_of_vehicle_removal_in_each_lane[i] # taking average between straight distance and diagonal distance
            else:
                time_required_for_one_vehicle_removal_in_each_lane[i] = 0

        max_number_of_vehicles_removed_in_each_lane = numpy.zeros(len(no_of_vehicle_to_remove_in_each_lanes))
        for i in range(len(time_required_for_one_vehicle_removal_in_each_lane)):
            if(time_required_for_one_vehicle_removal_in_each_lane[i] != 0):
                max_number_of_vehicles_removed_in_each_lane[i] = MIN_GREEN_TIME / time_required_for_one_vehicle_removal_in_each_lane[i]

        actual_number_of_vehicle_removed_in_each_lane = numpy.zeros(len(max_number_of_vehicles_removed_in_each_lane))
        for i in range(len(actual_number_of_vehicle_removed_in_each_lane)):
            if(no_of_vehicle_to_remove_in_each_lanes[i] != 0):
                # selecting whichever is the lowest either max possible to remove or maximum available to remove
                actual_number_of_vehicle_removed_in_each_lane[i] = min(max_number_of_vehicles_removed_in_each_lane[i], no_of_vehicle_to_remove_in_each_lanes[i])
        
        # Removing the vehicle from the lanes
        for laneNo in range(len(self.lanes)):
            self.lanes[laneNo] -= actual_number_of_vehicle_removed_in_each_lane[laneNo]

        vehicleThroughIntersection = reduce(lambda x,y: x + y, actual_number_of_vehicle_removed_in_each_lane)

        vehicleRemaining = totalVehicle - vehicleThroughIntersection

        # adding vehicles to the lanes according to the probability
        for j,probability in enumerate(PROBABILITY_OF_VEHICLE_OCCURANCE_IN_LANES_PER_SECOND):
            self.lanes[j] += int(probability * MIN_GREEN_TIME)


        return vehicleThroughIntersection, vehicleRemaining

    # dummy reward function for now
    def calculateReward(self, numberOfVehiclePassed, numberOfVehicleRemaining, last_observation_leading_to_predicted_action):
        
        percentage_of_vehicle_passing_through_the_intersection = numberOfVehiclePassed/(numberOfVehiclePassed + numberOfVehicleRemaining) * 100

        if numberOfVehiclePassed < min(last_observation_leading_to_predicted_action):
            reward = (percentage_of_vehicle_passing_through_the_intersection - self.percentageOfVehiclePassingThroughTheIntersectionLastTime) * 100
        elif numberOfVehiclePassed > max(last_observation_leading_to_predicted_action):
            reward = 1000
        else:
            reward = (1 - (max(last_observation_leading_to_predicted_action) - numberOfVehiclePassed)/max(last_observation_leading_to_predicted_action)) * 200

        self.percentageOfVehiclePassingThroughTheIntersectionLastTime = percentage_of_vehicle_passing_through_the_intersection

        return reward
