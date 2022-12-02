import sys
import argparse
from PyQt5 import QtWidgets

from utils.thread_evaluation import EvaluateThread
from utils.ui_train import TrainingUi
from configparser import ConfigParser


def get_parser():
    parser = argparse.ArgumentParser(
        description="trained model evaluation with plot")
    parser.add_argument('-model_path', required=True, help='model path to be evaluated, \
                                            just copy the relative path of the log')
    parser.add_argument('-eval_eps', required=True, type=int,
                        help='evaluation episode number')

    return parser


def main():

    # set evaluation model path
    eval_path = r'C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2022_09_06_15_05_SimpleMultirotor_No_CNN_SAC'

    # select config file and model name
    config_file = eval_path + '/config/config.ini'
    # config_file = r"D:\OneDrive - mail.nwpu.edu.cn\Github\PhD-thesis-plot\CH3\3_1_simple_training_and_anaylysis\3_1_4_training_analysis_multi\data\2022_08_30_10_43_SimpleMultirotor_No_CNN_SAC\config\config.ini"
    model_file = eval_path + '/models/model_sb3.zip'
    # config_file = r"C:\Users\helei\Documents\GitHub\UAV_Navigation_DRL_AirSim\configs\config_new.ini"
    # model_file = eval_path + '/models/model_200000.zip'
    total_eval_episodes = 50

    # 1. Create the qt thread (is MainThread in fact)
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    # 2. Start training thread
    evaluate_thread = EvaluateThread(
        eval_path, config_file, model_file, total_eval_episodes)
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.has_option('options', 'perception'):
        if cfg.get('options', 'perception') == 'lgmd':
            evaluate_thread.env.lgmd_signal.connect(gui.lgmd_plot_cb)

    evaluate_thread.start()

    # program will not terminate until you closed the GUI
    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    main()
