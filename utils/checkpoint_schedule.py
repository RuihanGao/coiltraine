import os


def maximun_checkpoint_reach(iteration, checkpoint_schedule):
    if iteration >= max(checkpoint_schedule):
        return True
    else:
        return False



""" FUNCTIONS FOR SAVING THE CHECKPOINTS """

def is_ready_to_save(iteration):
    """ Rerturns if the iteration is a iteration for saving a checkpoint

    """
    if iteration in g_conf.param.SAVE_SCHEDULE():
        return True
    else:
        return False

def get_latest_saved_checkpoint(exp_batch, exp_alias):
    """
        Returns the , latest checkpoint number that was saved

    """
    checkpoint_files = sorted(os.listdir(self._config_input.models_path))

    return checkpoint_files[-1]


""" FUNCTIONS FOR GETTING THE CHECKPOINTS"""
def get_latest_evaluated_checkpoint(exp_batch, exp_alias, process_name):

    """
        Get the latest checkpoint that was validated or tested.
    Args:
        exp_batch:
        exp_alias:
        process_name:

    Returns:

    """

    return monitor.get_latest_checkpoint(exp_batch, exp_alias, process_name)

def is_next_checkpoint_ready(exp_batch, exp_alias, process_name, checkpoint_schedule):

    ltst_check = get_latest_evaluated_checkpoint(exp_batch, exp_alias, process_name)

    next_check = checkpoint_schedule[checkpoint_schedule.index(ltst_check)+1]

    return str(next_check) + '.pth' in os.listdir(os.path.join(exp_batch, exp_alias, 'checkpoints'))


def get_next_checkpoint(exp_batch, exp_alias, process_name, checkpoint_schedule):
    ltst_check = get_latest_evaluated_checkpoint(exp_batch, exp_alias, process_name)

    return checkpoint_schedule[checkpoint_schedule.index(ltst_check) + 1]



#
# def next_check_point_ready():
#     """
#     Looks at every checkpoint file in the folder. And for each of
#     then tries to find the one that matches EXACTLY with the one in the schedule
#
#     :return:
#     """
#
#     checkpoint_files = sorted(os.listdir(self._config_input.models_path))
#     for f in checkpoint_files:
#
#         match = re.search('model.ckpt-(\d+)', f)
#         if match:
#             checkpoint_number = match.group(1)
#
#             if int(checkpoint_number) == (self._checkpoint_schedule[self._current_checkpoint_number]):
#                 self._checkpoint_number_to_test = str(self._checkpoint_schedule[self._current_checkpoint_number])
#
#                 return True
#     logging.info('Checkpoint Not Found, Will wait for %d' % self._checkpoint_schedule[self._current_checkpoint_number] )
#     return False
#
# def get_test_name():
#
#     return str(self._checkpoint_number_to_test)
#
# def finish_model():
#     """
#     Increment and go to the next model
#
#     :return None:
#
#     """
#     self._current_checkpoint_number += 1


def is_iteration_for_saving():


    return True