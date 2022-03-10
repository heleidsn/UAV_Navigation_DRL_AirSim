import sys
import argparse
from PyQt5 import QtWidgets

from utils.thread_evaluation import EvaluateThread
from utils.ui_train import TrainingUi

def get_parser():
    parser = argparse.ArgumentParser(description="trained model evaluation with plot")
    parser.add_argument('-model_path', required=True, help='model path to be evaluated, \
                                            just copy the relative path of the log')
    parser.add_argument('-eval_eps', required=True, type=int, help='evaluation episode number')

    return parser

def main():
    
    eval_path = 'logs_save/2022_03_10_16_06_test'

    config_file = eval_path + '/config/config.ini'
    model_file = eval_path + '/models/model_sb3'
    total_eval_episodes = 10
    
    # 1. Create the qt thread (is MainThread in fact)
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    # 2. Start training thread  
    evaluate_thread = EvaluateThread(config_file, model_file, total_eval_episodes)
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)
    evaluate_thread.start()

    # program will not terminate until you closed the GUI 
    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    main()
    