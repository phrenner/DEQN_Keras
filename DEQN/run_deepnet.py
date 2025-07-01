import tensorflow as tf
import Main

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

# tf.config.run_functions_eagerly(True) #for debugging enable eager execution


#### Run an episode ###
Main.run_cycles()
