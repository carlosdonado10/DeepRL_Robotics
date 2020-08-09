import pybullet as p
import pybullet_data as pd

# video requires ffmpeg available in path


def load_env(pandaObj, createVideo=False, visualize=True, fps=240, timeStep=1/240):
	if visualize:
		if createVideo:
			p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps="+str(fps) )
		else:
			p.connect(p.GUI)
	else:
		p.connect(p.DIRECT)

	p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
	p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
	p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
	p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35, -0.13, 0])
	p.setAdditionalSearchPath(pd.getDataPath())

	p.setTimeStep(timeStep)
	p.setGravity(0, -9.8, 0)

	panda = pandaObj(p, [0, 0, 0])
	panda.control_dt = timeStep

	# logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
	# panda.bullet_client.submitProfileTiming("start")
	# for i in range(100000):
	# 	panda.bullet_client.submitProfileTiming("full_step")
	# 	panda.step()
	# 	p.stepSimulation()
	# 	if createVideo:
	# 		p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
	# 	if not createVideo:
	# 		time.sleep(timeStep)
	# 	panda.bullet_client.submitProfileTiming()


	return panda