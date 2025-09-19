import airsim

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def takeoff(self):
        print("Arming the drone and taking off...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.hoverAsync().join()

    def land(self):
        print("Landing...")
        self.client.landAsync().join()
        print("Disarming.")
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def move(self, vx, vy, vz, duration):
        """Moves the drone at a specific velocity for a set duration."""
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
    
    def hover(self):
        """Hovers the drone in place."""
        self.client.moveByVelocityAsync(0, 0, 0, 1).join()