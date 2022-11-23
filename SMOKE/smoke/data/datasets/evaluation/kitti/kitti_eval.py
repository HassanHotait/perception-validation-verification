import os
import csv
import logging
import subprocess

from smoke.utils.miscellaneous import mkdir

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}


def kitti_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing kitti detection evaluation: ")
        do_kitti_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )


def do_kitti_detection_evaluation(dataset,
                                  predictions,
                                  output_folder,
                                  logger
                                  ):
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(predict_folder, predict_txt)

        generate_kitti_3d_detection(prediction, predict_txt)

    logger.info("Evaluate on KITTI dataset")
    output_dir = os.path.abspath(output_folder)
    print('Current Working Directory: ')
    print(os. getcwd())
    print('Look above me')
    label_dir = getattr(dataset, 'label_dir')
    label_dir='/home/hasan/SMOKE/datasets/kitti/training/label_2'
    os.chdir('./smoke/data/datasets/evaluation/kitti/kitti_eval_11')
    
    print('label_dir: ',label_dir)
    if not os.path.isfile('evaluate_object_3d_offline'):
        subprocess.Popen('g++ -O3 -DNDEBUG -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp', shell=True)
        print('Yes condition is happening')
        
    print('Moved after if condition')
    command = "./evaluate_object_3d_offline {} {}".format(label_dir, output_dir)
    print("Checkpoint 1")
    print('Current Working Directory: ')
    print(os. getcwd())
    output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()
    print("Checkpoint 2")
    logger.info(output)
    print("Checkpoint 3")
    
    #os.chdir('../tools')
    print('Did change directory to tools')


def generate_kitti_3d_detection(prediction, predict_txt):
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)


def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()