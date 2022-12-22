import logging
from tqdm import tqdm

import torch

from smoke.utils import comm
from smoke.utils.timer import Timer, get_time_str
from smoke.data.datasets.evaluation import evaluate


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
        images = images.to(device)
        print("Targets in compute on dataset: ",targets)

        print("Type of Images in compute on dataset: ",type(images))
        print("Type of Targets in compute on dataset",type(targets))
        print("Format Targets in compute on dataset: ",format(targets))
        print("Image IDS: ",image_ids)
        with torch.no_grad():
            if timer:
                timer.tic()
            #print(images[0])
            print('Input to Network: ',images)

            output = model(images, targets)
            print('Prediction Complete ----------------------------------------------------------------')
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = output.to(cpu_device)
        results_dict.update(
            {img_id: output for img_id in image_ids}
        )

    print("Output in compute on dataset: ",output)
    print("Results in compute on dataset: ",results_dict)

    return results_dict

# def my_compute_on_dataset(model, data_loader, device, timer=None):
#     model.eval()
#     results_dict = {}
#     cpu_device = torch.device("cpu")
#     #for _, batch in enumerate(tqdm(data_loader)):
#     images, targets, image_ids = tqdm(data_loader)["images"], tqdm(data_loader)["targets"], tqdm(data_loader)["img_ids"]
    
#     images = images.to(device)
#     with torch.no_grad():
#         if timer:
#             timer.tic()
#         output = model(images, targets)
#         #print(output)
#         if timer:
#             torch.cuda.synchronize()
#             timer.toc()
#         output = output.to(cpu_device)
#     results_dict.update(
#         {img_id: output for img_id in image_ids}
#     )
#     return results_dict


def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,

):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    print('PREDICTIONS in INFERENCE: ',predictions)
    comm.synchronize()

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    if not comm.is_main_process():
        return

    #print("Inference Predictions: ",predictions)
    # print("Inference Output Folder: ",output_folder)
    # print("Inference eval_types: ", eval_types)
    return evaluate(eval_type=eval_types,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder, )
