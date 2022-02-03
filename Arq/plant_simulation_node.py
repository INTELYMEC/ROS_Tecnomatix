
# ============================================================================
# IMPORTS
# ============================================================================

import rospy
from material_handling import Material_Handling
from rl_method_1 import RL_Method_1

# ============================================================================


def plant_simulation_node():
    rospy.init_node('plant_simulation_node', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    method = RL_Method_1()
    plant = Material_Handling(method)
    plant.process_simulation()

    rate.sleep()


if __name__ == '__main__':
    try:
        plant_simulation_node()
    except rospy.ROSInterruptException:
        pass
