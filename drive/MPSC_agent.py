import os, sys
import numpy as np
import torch
import glob
import logging

import carla

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
from coilutils.drive_utils import checkpoint_parse_configuration_file
from configs import g_conf, merge_with_yaml
from network import CoILModel

class MPSCAgent(AutonomousAgent):
	def setup(self, path_to_config_file):

		yaml_conf, checkpoint_number = checkpoint_parse_configuration_file(path_to_config_file)

		# Take the checkpoint name and load it
		checkpoint = torch.load(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
											  '_logs',
											 yaml_conf.split('/')[-2], yaml_conf.split('/')[-1].split('.')[-2]
											 , 'checkpoints', str(checkpoint_number) + '.pth'))

		# merge the specific agent config with global config _g_conf
		merge_with_yaml(os.path.join('/', os.path.join(*os.path.realpath(__file__).split('/')[:-2]),
									 yaml_conf))

		self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
		# TODO: retrain the model with MPSC
		self._model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
		self.first_iter = True
		logging.info("Setup Model")
		# Load the model and prepare set it for evaluation
		self._model.load_state_dict(checkpoint['state_dict'])
		self._model.cuda()
		self._model.eval()
		self.latest_image = None
		self.latest_image_tensor = None
		# We add more time to the curve commands
		self._expand_command_front = 5
		self._expand_command_back = 3
		# RH: add more sensors compared to coil
		# TODO: check the diff betw ALL_SENS & ALL_SENS_HDMAP_WP
		# self.track = Track.CAMERAS	
		self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS

	def sensors(self):
		# currently give the full suite of available sensors
		# TODO: check the config/installation of the sensors
		sensors = [{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll':0.0, 'pitch':0.0, 'yaw': 0.0,
					'width': 800, 'height': 600, 'fov':100, 'id': 'Center'},
				   {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
					'yaw': -45.0, 'width': 800, 'height': 600, 'fov': 100, 'id': 'Left'},
				   {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
					'width': 800, 'height': 600, 'fov': 100, 'id': 'Right'},
				   {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
					'yaw': -45.0, 'id': 'LIDAR'},
				   {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
				   {'type': 'sensor.can_bus', 'reading_frequency': 25, 'id': 'can_bus'},
				   {'type': 'sensor.hd_map', 'reading_frequency': 1, 'id': 'hdmap'},
				  ]
		return sensors

	def run_step(self, input_data, timestamp):
		# the core method
		# TODO
		# 1. request current localization 
		for key, value in input_data.items():
			print("input_data ", key)
		localization = input_data['GPS']
		directions = self._get_current_direction(input_data['GPS'][1])
		logging.debug("Directions {}".format(directions))


		# 2. get recommended action from the NN controller (copy from CoILBaseline)
        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = input_data['can_bus'][1]['speed'] / g_conf.SPEED_FACTOR
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])
        # Compute the forward pass processing the sensors got from CARLA.
        model_outputs = self._model.forward_branch(self._process_sensors(input_data['rgb'][1]),
                                                   norm_speed,
                                                   directions_tensor)
        steer, throttle, brake = self._process_model_outputs(model_outputs[0])

		# 3. use inner-loop to simulate/approximate vehicle model
		# save the NN output as vehicle control
		sim_control = carla.VehicleControl()
		sim_control.steer = float(steer)
		sim_control.throttle = float(throttle)
		sim_control.brake = float(brake)
		logging.debug("inner loop for sim_control", sim_control)
		# TODO
		# create a "virtual agent" that has the same state with ego_vehicle
		sim_ego = self.world.create_ego_vehicle(current_ego_states)
		# pass the sim_control to virtual agent and run T timesteps
		sim_ego.apply_control(sim_control)
		# use current model to predict the following state-action series
		MPSC_controls = [] # TODO: check where u should init it
		for i in range(T):
			sim_ego.run_step() # TODO def run_step, update for sim_ego
			sim_ego.update()
			# 4. use MPSC to check safety at each future timestep
			safe = MPSC.check_safety(sim_ego.state, safety_boundary)
			
			if not safe:
				# if not safe, obtain MPSC control output
				logging.debug("use MPSC controller")
				control = MPSC_control
				MPSC_controls.append(MPSC_control) #  collect all "safe" o/p
				# 7. execute MPSC control and add it to new dataset
				break
			else:
				if i < T-1:
					continue
				else: # final step
					# if safe within all T timesteps, proceed to  use NN control output
					logging.debug("use NN controller")
					control = sim_control
		# 8. retrain the network and/or do policy aggregation
		if len(MPSC_controls):
			self.model.train(self.model, MPSC_controls)

        logging.debug("Control output ", control)
        # There is the posibility to replace some of the predictions with oracle predictions.
        self.first_iter = False
        return control


    # TODO: copy from CoIL, do doubel check
    def _get_current_direction(self, vehicle_position):

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        min_distance = 100000
        for index in range(len(self._global_plan)):

            waypoint = self._global_plan[index][0]

            computed_distance = distance_vehicle(waypoint, vehicle_position)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index

        #print("Closest waypoint {} dist {}".format(closest_id, min_distance))
        direction = self._global_plan[closest_id][1]
        print ("Direction ", direction)
        if direction == RoadOption.LEFT:
            direction = 3.0
        elif direction == RoadOption.RIGHT:
            direction = 4.0
        elif direction == RoadOption.STRAIGHT:
            direction = 5.0
        else:
            direction = 2.0

        return direction