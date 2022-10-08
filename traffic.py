from __future__ import print_function
from __future__ import absolute_import
from genericpath import exists

import os
import sys
import optparse
from pathlib import Path
from generateRouteFile import generate_routefile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # traffic-intersection-rl-environment-cli root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

TRAFFIC_INTERSECTION_TYPE="double"
TOTAL_TIMESTEPS=50000
GENERATE_CUSTOM_ROUTE=True
time_to_teleport="-1"       # setting time to teleport to -1 will make vehicles not teleport

# sumo stuffs
SUMO_DIRECTORY= Path(str(ROOT) + "/sumo-files").resolve()
net_file=Path(str(SUMO_DIRECTORY) + "/small-map-{}-lane.net.xml".format(TRAFFIC_INTERSECTION_TYPE)).resolve()
route_file=Path(str(SUMO_DIRECTORY) + "/small-map-{}-lane.rou.xml".format(TRAFFIC_INTERSECTION_TYPE)).resolve()
sumocfg_file=Path(str(SUMO_DIRECTORY) + "/small-map-{}-lane.sumocfg".format(TRAFFIC_INTERSECTION_TYPE)).resolve()

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
models = Path(str(ROOT) + "/models").resolve()
model = PPO.load(Path(str(models) + "/2022-10-08 14:55:51.640820-TrafficIntersection-{}Lane-ppo-150000".format(TRAFFIC_INTERSECTION_TYPE.capitalize())).resolve())

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

    # Generating custom route file
    if GENERATE_CUSTOM_ROUTE and TRAFFIC_INTERSECTION_TYPE == "double":
        route_file = generate_routefile(routefilePath=route_file.parents[0])

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, 
    "-c", sumocfg_file,
    "-n", net_file,
    "-r", route_file,
    '--start', '--quit-on-end',
    "--time-to-teleport", time_to_teleport])

    run()

    traci.close()