# opencood environment has issues with tensorboard. Activate pynew env then run the following. 

import tensorflow as tf
import torch


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
    event_file = '/home/ma906813/projectmulti_agent_perception/OpenCOOD/nautilus/output/point_pillar_v2xvit_2024_08_12_19_37_52/events.out.tfevents.1723491472.job-train-lite-5f6q4'
    # list_tags(event_file)
    # best_step, best_loss = find_best_model(event_file)
    all_losses = list_all_losses(event_file)




    # for event in tf.compat.v1.train.summary_iterator(event_file):
    #     for value in event.summary.value:
    #         print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value}")
    # for e in tf.compat.v1.train.summary_iterator(event_file):
    #     for v in e.summary.value:
    #         if v.tag == 'your/precision_tag':  # Replace with your specific tag for AP
    #             # The tensor value will contain your average precision
    #             average_precision = tf.make_ndarray(v.tensor)
    #             print(f'average_precision: {average_precision}')





    # ----------------------------------------------------------------
    # model_file = '/home/ma906813/projectmulti_agent_perception/OpenCOOD/opencood/logs/point_pillar_v2xvit_2024_08_07_21_10_56/events.out.tfevents.1723079460.CECSL6YQTDH2'
    # state_dict = torch.load(model_file)

    # print(state_dict.keys())
    # print(f'epoch: {state_dict["epoch"]}')
    # print(f'max_accuracy: {state_dict["max_accuracy"]}')
