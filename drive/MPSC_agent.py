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
		# check map waypoint format => carla_data_provider & http://carla.org/2018/11/16/release-0.9.1/
		# e.g. from map.get_waypoint Waypoint(Transform(Location(x=338.763, y=226.453, z=0), Rotation(pitch=360, yaw=270.035, roll=0)))
		self.track = Track.ALL_SENSORS_HDMAP_WAYPOINTS # specify available track info, see autonomous_agent.py

	def sensors(self):
		# currently give the full suite of available sensors
		# check the config/installation of the sensors => https://carla.readthedocs.io/en/latest/cameras_and_sensors/
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
		# input_data is obtained from sensors. => autonomous_agent.py def __call__(self)
		for key, value in input_data.items():
			print("input_data ", key, value)


# 		  
# ======[Agent] Wallclock_time = 2019-07-08 14:26:54.522155 / Sim_time = 1.4500000216066837
# input_data key  GPS (3755, array([49.00202793,  8.00463308,  1.58916414]))
# input_data key  can_bus (43, {'moi': 1.0, 'center_of_mass': {'x': 60.0, 'y': 0.0, 'z': -60.0}, 'linear_velocity': array([[<carla.libcarla.Vector3D object at 0x7fb4fa0e2348>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e2450>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e2608>],
#        [<carla.libcarla.Vector3D object at 0x7fb4fa0e22f0>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e2870>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e26b8>],
#        [<carla.libcarla.Vector3D object at 0x7fb4fa0e2500>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e2818>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0ddfa8>]], dtype=object), 'speed': -1.6444947256841175e-06, 'lateral_speed': array([[<carla.libcarla.Vector3D object at 0x7fb4fa0e4ad8>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e49d0>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e23a0>],
#        [<carla.libcarla.Vector3D object at 0x7fb4fa0e48c8>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e4ce8>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e23f8>],
#        [<carla.libcarla.Vector3D object at 0x7fb4fa0e4d40>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e4c90>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0e28c8>]], dtype=object), 'transform': <carla.libcarla.Transform object at 0x7fb4fa0de3f0>, 'damping_rate_zero_throttle_clutch_disengaged': 0.3499999940395355, 'max_rpm': 6000.0, 'clutch_strength': 10.0, 'drag_coefficient': 0.30000001192092896, 'linear_acceleration': array([[<carla.libcarla.Vector3D object at 0x7fb4fa0dd0e0>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0ddf50>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0d58c8>],
#        [<carla.libcarla.Vector3D object at 0x7fb4fa0dd088>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0dd138>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0d5088>],
#        [<carla.libcarla.Vector3D object at 0x7fb4fa0dd1e8>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0f6d98>,
#         <carla.libcarla.Vector3D object at 0x7fb4fa0d5920>]], dtype=object), 'damping_rate_full_throttle': 0.15000000596046448, 'use_gear_autobox': True, 'torque_curve': [{'x': 0.0, 'y': 400.0}, {'x': 1890.7607421875, 'y': 500.0}, {'x': 5729.57763671875, 'y': 400.0}], 'dimensions': {'width': 0.9279687404632568, 'height': 0.6399999856948853, 'length': 2.4543750286102295}, 'steering_curve': [{'x': 0.0, 'y': 1.0}, {'x': 20.0, 'y': 0.8999999761581421}, {'x': 60.0, 'y': 0.800000011920929}, {'x': 120.0, 'y': 0.699999988079071}], 'mass': 1850.0, 'wheels': [{'tire_friction': 3.5, 'steer_angle': 70.0, 'damping_rate': 0.25, 'disable_steering': False}, {'tire_friction': 3.5, 'steer_angle': 70.0, 'damping_rate': 0.25, 'disable_steering': False}, {'tire_friction': 3.5, 'steer_angle': 0.0, 'damping_rate': 0.25, 'disable_steering': False}, {'tire_friction': 3.5, 'steer_angle': 0.0, 'damping_rate': 0.25, 'disable_steering': False}]})
# input_data key  rgb (3753, array([[[135, 118, 110, 255],
#         [135, 118, 110, 255],
#         [136, 119, 110, 255],
#         ...,
#  		  [[114, 108, 105, 255],
#         [110, 105, 102, 255],
#         [112, 106, 104, 255],
#         ...,
#         [118, 112, 109, 255],
#         [118, 112, 109, 255],
#         [121, 115, 113, 255]]], dtype=uint8))
# Direction  RoadOption.LANEFOLLOW
# ego_trans
# Transform(Location(x=338.763, y=226.453, z=-0.0109183), Rotation(pitch=0.000136604, yaw=-89.9654, roll=-0.000274658))
# 1.9995784804148584/0.0

		localization = input_data['GPS']
		directions = self._get_current_direction(input_data['GPS'][1])
		logging.debug("Directions {}".format(directions))


		# 2. get recommended action from the NN controller (copy from CoILBaseline)
		# Take the forward speed and normalize it for it to go from 0-1
		norm_speed = input_data['can_bus'][1]['speed'] / g_conf.SPEED_FACTOR
		norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
		directions_tensor = torch.cuda.LongTensor([directions])
		# End-to-end part, feed in images from rgb sensor, then parse network output as controller
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
		# copy a "parallel world" and create a "virtual agent" that has the same state with ego_vehicle
		sim_world = self.world # TODO: check how to copy the world, roads info are necessary, the rest optional
		sim_ego = sim_world.create_ego_vehicle(current_ego_states)

		sim_world.agent_instance = getattr(sim_world.module_agent, sim_world.module_agent.__name__)(args.config)
        correct_sensors, error_message = sim_world.valid_sensors_configuration(sim_world.sim_agent, sim_world.track)
		
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