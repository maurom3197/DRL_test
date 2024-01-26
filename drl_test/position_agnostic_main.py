#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from drl_test.task.position_agnostic_crop_follow.position_agnostic_crop_follow import DRLAgent
import tensorflow as tf
import os 

#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

def main(args=None):
    rclpy.init(args=args)

    drl_agent = DRLAgent()

    # drl_agent.get_logger().info('Spinning')
    # rclpy.spin(drl_agent)
    # drl_agent.destroy_node()
    # rclpy.shutdown()

    try:
        # executor.spin()
        drl_agent.execute()
    except KeyboardInterrupt:
        # executor.shutdown()
        drl_agent.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

