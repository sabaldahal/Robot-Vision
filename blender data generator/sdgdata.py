class SDGData():
    def __init__(self, scene, camera, resx, resy, bottom_collection, top_collection, keypoint_collection, obj_controller, lights):
        self.scene = scene
        self.camera = camera
        self.resx = resx
        self.resy = resy
        self.bottom_collection = bottom_collection
        self.top_collection = top_collection
        self.keypoint_collection = keypoint_collection
        self.obj_controller = obj_controller
        self.lights = lights