# opencood environment has issues with tensorboard. Activate pynew env then run the following. 

import tensorflow as tf
import torch
import os


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def list_tags(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    tags = event_acc.Tags()
    print("Available Tags:")
    for tag_type, tag_list in tags.items():
        print(f"{tag_type}:")
        for tag in tag_list:
            print(f"  - {tag}")


def find_best_model(event_file):
    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Retrieve scalar data
    validate_loss = event_acc.Scalars('Validate_Loss')

    # Find the minimum validation loss and corresponding step
    min_validate_loss = min(validate_loss, key=lambda x: x.value)
    best_step = min_validate_loss.step
    best_loss = min_validate_loss.value

    print(f"Best model found at step: {best_step}")
    print(f"Minimum validation loss: {best_loss}")

    return best_step, best_loss


def list_all_losses(event_file):
    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Retrieve scalar data
    validate_loss = event_acc.Scalars('Validate_Loss')

    # Print all steps and corresponding validation losses
    print("Step\tValidation Loss")
    for entry in validate_loss:
        print(f"{entry.step}\t{entry.value}")

    # Optionally return the list of steps and losses if needed
    return [(entry.step, entry.value) for entry in validate_loss]

if __name__ == '__main__':
    event_file = "/home/ma906813/projectmulti_agent_perception/OpenCOOD/results/point_pillar_v2xvit_2025_03_10_05_22_10/events.out.tfevents.1741584131.train-v2xvit-nomswin-cd-a6-245jw"
    best_step, best_loss = find_best_model(event_file)
    
    '''
    Or just run 
    tensorboard --logdir='/path/to/event/file' to visualize the event file
    '''