from __future__ import print_function
from __future__ import absolute_import
from genericpath import exists

import os
import sys
import optparse
from time import sleep

from generateRouteFile import generate_routefile

TRAFFIC_INTERSECTION_TYPE="double"
TOTAL_TIMESTEPS=5000
GENERATE_CUSTOM_ROUTE=True

# checking for sumo_home variable and exiting if it is not found
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumo
import sumo.tools.sumolib as sumolib
from sumo.tools import traci

import gym
from custom_gym.envs.custom_env_dir import TrafficIntersectionEnvSingleLane
from stable_baselines3 import PPO

env = gym.make('TrafficIntersectionEnv{}Lane-v1'.format(TRAFFIC_INTERSECTION_TYPE.capitalize()))

# Here, formatting is done as to create error if wrong model is selected
# as, there won't be same model trained at exact same time and upto same timesteps
model = PPO.load("/home/bishal/Programming/reinforcement-learning-env-cli/models/2022-08-18 19:02:03.388610-TrafficIntersection-{}Lane-ppo-1000000".format(TRAFFIC_INTERSECTION_TYPE.capitalize()))

def run():
    step = 0
    
    while step < TOTAL_TIMESTEPS:

        if step == 0: # setting the initial configuration
            if TRAFFIC_INTERSECTION_TYPE == "single":
                traci.trafficlight.setPhase("J9", 4)
            elif TRAFFIC_INTERSECTION_TYPE == "double":
                traci.trafficlight.setPhase("J11", 4)

            traci.simulationStep()
            step += 1
            continue

        sleep(0.08)

        # Selection traffic configuration once every 30 timesteps
        
        simulation_time = traci.simulation.getTime()
        if  simulation_time % 30 == 0:

            if TRAFFIC_INTERSECTION_TYPE == "single":

                junction_with_lights = "J9"

                current_state = traci.trafficlight.getPhase(junction_with_lights)

                # starting from left-upper lane, skipping one lane
                vehicle_count_lane_0 = traci.lane.getLastStepVehicleNumber("-E8_0")
                vehicle_count_lane_1 = traci.lane.getLastStepVehicleNumber("-E9_0")
                vehicle_count_lane_2 = traci.lane.getLastStepVehicleNumber("-E10_0")
                vehicle_count_lane_3 = traci.lane.getLastStepVehicleNumber("E7_0")

                lanes_observation = [vehicle_count_lane_0, vehicle_count_lane_1, vehicle_count_lane_2, vehicle_count_lane_3]

                lane_to_have_next_green_light, _state = model.predict(lanes_observation, deterministic=True)

                if lane_to_have_next_green_light == current_state / 2: # dividing by 2 as green phases is 0, 2, 4, 6
                    traci.trafficlight.setPhase(junction_with_lights, current_state)
                else:
                    # Trying to turn on yellow light
                    # This doesn't work for now, we need to figure out a way to send the next traffic light configuration after turning on yellow light
                    traci.trafficlight.setPhase(junction_with_lights, current_state + 1)

                    # changing phase here makes the above phase change obsolete
                    traci.trafficlight.setPhase(junction_with_lights, lane_to_have_next_green_light * 2)

            elif TRAFFIC_INTERSECTION_TYPE == "double":

                junction_with_lights = "J11"

                current_state = traci.trafficlight.getPhase(junction_with_lights)

                vehicle_count_lane_0 = traci.lane.getLastStepVehicleNumber("E9_0")
                vehicle_count_lane_1 = traci.lane.getLastStepVehicleNumber("E9_1")
                vehicle_count_lane_2 = traci.lane.getLastStepVehicleNumber("E8_0")
                vehicle_count_lane_3 = traci.lane.getLastStepVehicleNumber("E8_1")
                vehicle_count_lane_4 = traci.lane.getLastStepVehicleNumber("-E10_0")
                vehicle_count_lane_5 = traci.lane.getLastStepVehicleNumber("-E10_1")
                vehicle_count_lane_6 = traci.lane.getLastStepVehicleNumber("-E11_0")
                vehicle_count_lane_7 = traci.lane.getLastStepVehicleNumber("-E11_1")

                lanes_observation = [vehicle_count_lane_0, vehicle_count_lane_1, vehicle_count_lane_2, vehicle_count_lane_3, vehicle_count_lane_4, vehicle_count_lane_5, vehicle_count_lane_6, vehicle_count_lane_7]
                next_configuration, _state = model.predict(lanes_observation, deterministic=True)

                if next_configuration == current_state / 2: # dividing by 2 as green phases is 0, 2, 4, 6
                    traci.trafficlight.setPhase(junction_with_lights, current_state)
                    
                    for id in traci.route.getIDList():
                        print(traci.route.getEdges(id))
                        print("\n")
                    
                else:
                    # Trying to turn on yellow light
                    # This doesn't work for now, we need to figure out a way to send the next traffic light configuration after turning on yellow light
                    traci.trafficlight.setPhase(junction_with_lights, current_state + 1)

                    # changing phase here makes the above phase change obsolete
                    traci.trafficlight.setPhase(junction_with_lights, next_configuration * 2)

        traci.simulationStep()
        step += 1

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()

    file_separator = os.path.sep

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = sumolib.checkBinary('sumo')
    else:
        sumoBinary = sumolib.checkBinary('sumo-gui')

    current_directory = os.getcwd()
    sumo_files_directory = current_directory + file_separator + "sumo-files" + file_separator

    if not exists(sumo_files_directory):
        sys.exit("Couldn't find sumo files directory, please place it in proper place.")
    

    # Generating custom route file
    if GENERATE_CUSTOM_ROUTE and TRAFFIC_INTERSECTION_TYPE == "double":
        route_file = generate_routefile()
    else:
        route_file = "{}small-map-{}-lane.rou.xml".format(sumo_files_directory, TRAFFIC_INTERSECTION_TYPE)

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, 
    "-c", "{}small-map-{}-lane.sumocfg".format(sumo_files_directory, TRAFFIC_INTERSECTION_TYPE),
    "-n", "{}small-map-{}-lane.net.xml".format(sumo_files_directory, TRAFFIC_INTERSECTION_TYPE),
    "-r", route_file,
    '--start', '--quit-on-end',
    "--time-to-teleport", "-1"])

    run()

    traci.close()