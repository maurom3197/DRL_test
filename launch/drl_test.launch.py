import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
   config = os.path.join(
      get_package_share_directory('drl_test'),
      'config',
      'main_params.yaml'
      )

   return LaunchDescription([
      Node(
         package='drl_test',
         executable='position_agnostic_main',
         namespace='',
         name='main_node',
         parameters=[config, {"main_params_path":config}]
      )
   ])