#!/usr/bin/env python
import sm
import aslam_cv as acv
import aslam_cameras_april as acv_april
import kalibr_common as kc
from kalibr_imu_camera_calibration import *

import tf
import numpy as np
import argparse
import signal
import sys   

import roslib
roslib.load_manifest('aprilgrid_detector')
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy

# make numpy print prettier
np.set_printoptions(suppress=True)

def signal_exit(signal, frame):
    print
    sm.logWarn("Shutting down! (CTRL+C)")
    sys.exit(1)

#helper to constrain certain arguments to be specified only once
class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        if getattr(namespace, self.dest) is not None:
            msg = '{o} can only be specified once'.format(o = option_string)
            raise argparse.ArgumentError(None, msg)
        setattr(namespace, self.dest, values)

def parseArgs():
    class KalibrArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sm.logError('%s' % message)
            sys.exit(2)
        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)
            return formatter.format_help()     
        
    usage = """
    Example usage to detect an april grid pattern and publish its 3D pose.
    
    %(prog)s --cam camchain.yaml --target aprilgrid.yaml
    
    camchain.yaml: is the camera-system calibration output of the multiple-camera
                   calibratin tool (kalibr_calibrate_cameras)
    
    example aprilgrid.yaml:       |  example imu.yaml: (ADIS16448)
        target_type: 'aprilgrid'  |      accelerometer_noise_density: 0.006  
        tagCols: 6                |      accelerometer_random_walk: 0.0002
        tagRows: 6                |      gyroscope_noise_density: 0.0004
        tagSize: 0.088            |      gyroscope_random_walk: 4.0e-06
        tagSpacing: 0.3           |      update_rate: 200.0"""    

    #setup the argument list
    parser = KalibrArgParser(description='Calibrate the spatial and temporal parameters of an IMU to a camera chain.', usage=usage)
    
    #configuration files
    groupCam = parser.add_argument_group('Camera system configuration')
    groupCam.add_argument('--cams', dest='chainYaml', help='Camera system configuration as yaml file', action=Once)
    
    groupTarget = parser.add_argument_group('Calibration target')
    groupTarget.add_argument('--target', dest='target_yaml', help='Calibration target configuration as yaml file', required=True, action=Once)
    
    #optimization options
    groupOpt = parser.add_argument_group('Optimization options')
    groupOpt.add_argument('--time-calibration', action='store_false', dest='no_time', help='Enable the temporal calibration', required=False)      
    groupOpt.add_argument('--max-iter', type=int, default=30, dest='max_iter', help='Max. iterations (default: %(default)s)', required=False)
    groupOpt.add_argument('--recover-covariance', action='store_true', dest='recover_cov', help='Recover the covariance of the design variables.', required=False)

    #Result options  
    outputSettings = parser.add_argument_group('Output options')
    outputSettings.add_argument('--show-extraction', action='store_true', dest='showextraction', help='Show the calibration target extraction. (disables plots)')
    outputSettings.add_argument('--extraction-stepping', action='store_true', dest='extractionstepping', help='Show each image during calibration target extraction  (disables plots)', required=False)
    outputSettings.add_argument('--verbose', action='store_true', dest='verbose', help='Verbose output (disables plots)')
    outputSettings.add_argument('--dont-show-report', action='store_true', dest='dontShowReport', help='Do not show the report on screen after calibration.')
     
    #print help if no argument is specified
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(2)
    
    #Parser the argument list
    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)
                 
    if parsed.verbose:
        parsed.showextraction = True             
    
    #there is a with the gtk plot widget, so we cant plot if we have opencv windows open...
    #--> disable the plots in these special situations
    if parsed.showextraction or parsed.extractionstepping or parsed.verbose:
        parsed.dontShowReport = True
    
    return parsed

def setupCalibrationTarget(targetConfig, 
                           camera,
                           showExtraction=False, 
                           showReproj=False, 
                           imageStepping=False):
        
    #load the calibration target configuration
    targetParams = targetConfig.getTargetParams()
    targetType = targetConfig.getTargetType()
    
    if targetType == 'checkerboard':
        options = acv.CheckerboardOptions(); 
        options.showExtractionVideo = showExtraction;
        grid = acv.GridCalibrationTargetCheckerboard(targetParams['targetRows'], 
                                                     targetParams['targetCols'], 
                                                     targetParams['rowSpacingMeters'], 
                                                     targetParams['colSpacingMeters'],
                                                     options)
    elif targetType == 'circlegrid':
        options = acv.CirclegridOptions(); 
        options.showExtractionVideo = showExtraction;
        options.useAsymmetricCirclegrid = targetParams['asymmetricGrid']
        grid = acv.GridCalibrationTargetCirclegrid(targetParams['targetRows'],
                                                   targetParams['targetCols'], 
                                                   targetParams['spacingMeters'], 
                                                   options)
    elif targetType == 'aprilgrid':
        options = acv_april.AprilgridOptions(); 
        options.showExtractionVideo = showExtraction;
        options.minTagsForValidObs = int( np.max( [targetParams['tagRows'], targetParams['tagCols']] ) + 1 )
            
        grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'],
                                                        targetParams['tagCols'], 
                                                        targetParams['tagSize'], 
                                                        targetParams['tagSpacing'], 
                                                        options)
    else:
        raise RuntimeError( "Unknown calibration target." )
                          
    options = acv.GridDetectorOptions() 
    options.imageStepping = imageStepping
    options.plotCornerReprojection = showReproj
    options.filterCornerOutliers = True
    #options.filterCornerSigmaThreshold = 2.0
    #options.filterCornerMinReprojError = 0.2
    detector = acv.GridDetector(camera.geometry, grid, options)        
    return detector

class AprilgridDetector:
    def __init__(self):
        # Initialize Kalibr detector
        camchain = rospy.get_param('~camchain')
        target = rospy.get_param('~target')

        verbose = False
        show_extraction = False
        extraction_stepping = False
        #logging modess
        if verbose:
            sm.setLoggingLevel(sm.LoggingLevel.Debug)
        else:
            sm.setLoggingLevel(sm.LoggingLevel.Info)
        
        signal.signal(signal.SIGINT, signal_exit)
    
        #load calibration target configuration 
        targetConfig = kc.CalibrationTargetParameters(target)
        
        print "Initializing calibration target:"
        targetConfig.printDetails()
        
        print "Initializing camera chain:"
        chain = kc.CameraChainParameters(camchain)      
        chain.printDetails()   
        
        leftCamConfig = chain.getCameraParameters(0)
        
        left_camera = kc.AslamCamera.fromParameters( leftCamConfig )
        
        self._detector = setupCalibrationTarget(targetConfig,
                                               left_camera,
                                               showExtraction=show_extraction,
                                               showReproj=show_extraction,
                                               imageStepping=extraction_stepping)

        # ROS stuff
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber("image", 
                                          Image, 
                                          self.image_callback,
                                          queue_size=1)
        self._tf_broadcaster = tf.TransformBroadcaster()

    def image_callback(self, ros_image):
        try:
            cv_image = np.squeeze(np.array(self._bridge.imgmsg_to_cv2(ros_image, "mono8")))
        except CvBridgeError as e:
            print(e)
        #cv2.imshow("Image", cv_image)
        #cv2.waitKey(1)
        timestamp = acv.Time(ros_image.header.stamp.secs,
                             ros_image.header.stamp.nsecs)
        success, observation = self._detector.findTarget(timestamp, cv_image)
        if not success:
            return
        transform = np.linalg.inv(observation.T_t_c().T())
        R = transform.copy()
        R[:3, 3] = 0
        q = tf.transformations.quaternion_from_matrix(R)
        t = transform[:3, 3]

        self._tf_broadcaster.sendTransform(t,
                                           q,
                                           ros_image.header.stamp,
                                           "aprilgrid",
                                           "camera")

def main():
    rospy.init_node('aprilgrid_detector', anonymous=True)
    aprilgrid_detector = AprilgridDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
if __name__ == "__main__":
    main()
#     try:
#         main()
#     except Exception,e:
#         sm.logError("Exception: {0}".format(e))
#         sys.exit(-1)
        
