import os

from clearml import Task, Logger


params_dictionary = {'lr': 0.001, 'epochs': 5, 'batch_size': 16}


task = Task.init(project_name='ProjectCV', task_name='TEST')

task.connect(params_dictionary)

task.set_user_properties(
  {"name": "backbone", "description": "network type", "value": "mstcn++"}
)

PATH_FEATURES = "/datashare/APAS/features/"
print(os.listdir(PATH_FEATURES))


for epoch in range(params_dictionary['epochs']):
    for batch_idx in range(params_dictionary['batch_size']):
        loss = 0.1
        Logger.current_logger().report_scalar(
            title="train", series="loss", iteration=(epoch * params_dictionary['epochs'] + batch_idx), value=loss)