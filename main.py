import argparse
import sys
from CONST import *

def arg_parser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for DQN-Atari Experiment",
        epilog="python main.py ")
    parser.add_argument(
        '--game', type=str, default='breakout',
        help='Name of the game to run the experiment on.')
    parser.add_argument(
        '--dropouts', type=list, default=DROPS,
        help='List with the probabilities to dropout for each layer of the architecture.')
    parser.add_argument(
        '--memory_size', type=int, default=MEMORY_SIZE,
        help='Size of the memory replay.')
    parser.add_argument(
        '--seed', type=int, default=-1)
    parser.add_argument(
        '--game_actions', type=int, default=4,
        help="An override for the actions to take.")
    parser.add_argument(
        '--steps', type=int, default=UPDATES,
        help='Steps the learner agent for the whole experiment.')
    parser.add_argument(
        '--mini_batch_size', type=int, default=MINI_BATCH_SIZE,
        help='Size of the mini batch from the replay memory.')
    parser.add_argument(
        '--lhist', type=int, default=LHIST,
        help='Number of frames to stack for the history.')
    parser.add_argument(
        '--learning_rate', type=float, default=LEARNING_RATE,
        help='Starting learning rate for the RMSprop optimizer.')
    parser.add_argument(
        '--double', type=bool, default=False,
        help='True if you want the target to be evaluated as Double-DQN.')
    parser.add_argument(
        '--update_target', type=int, default=UPDATE_TARGET,
        help='Steps of the learner in which the target net is updated with the online parameters.')
    parser.add_argument(
        '--test_steps', type=int, default=TEST_STEPS,
        help='Number of steps to execute per test.')
    parser.add_argument(
        '--test_freq', type=int, default=TEST_FREQ,
        help='Number of updates before a test is executed.')
    parser.add_argument(
        '--buffer_size', type=int, default=BUFFER_SIZE,
        help='Number of transitions to store in each actor agent.')
    parser.add_argument(
        '--memory_start', type=int, default=REPLAY_START,
        help='Steps to apply a random policy to fill the buffer before start learning.')
    parser.add_argument(
        '--update_online', type=int, default=UPDATE_FREQ_ONlINE,
        help='Steps in between the environment to update the onlineModel.')
    parser.add_argument(
        '--ray', type=bool, default=False,
        help='Flag to enable the ray behaviour or not for the trainer.')
    parser.add_argument(
        '--play', type=int, default=PLAY_STEPS,
        help='Pass a number of steps for the trained agent to play. Defaults 3000, does not play')
    parser.add_argument(
        '--buffer_samples', type=int, default=BUFFER_SAMPLES,
        help='Number of sampled histories stored on the buffer to show in screen.')
    parser.add_argument(
        '--crop', type=tuple, default=CROP,
        help='The vertical indices to crop the frame of the observation.')
    parser.add_argument(
        '--mode', type=str, default='train',
        help="Mode of the main program. 'train' is the default mode to create and train an agent\
        from  scratch. 'test' is a mode where one can load a pass model and test it.")
    parser.add_argument(
        '--optimizer', type=str, default='rmsprop',
        help="Optimizer to use in the learner. Options rmsprop and adam")
    parser.add_argument(
        '--dropout_exploration', type=bool, default=False,
        help="Enable the actors to do the exploration with dropout.")
    return parser.parse_known_args(args)[0]

from trainer import Trainer

if __name__ == '__main__':
    args = arg_parser(sys.argv[:])
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.fillMemory()
        trainer.sampleBuffer(args.buffer_samples)
        trainer.train()
        trainer.test()
        trainer.close()
        trainer.playTest()
    elif args.mode == 'test':
        modelDir = input("Please type the folder in which the model is in:")
        trainer.loadActor(modelDir)
        tests = int(input("How many test runs?:"))
        for i in range(tests):
            trainer.test()
        trainer.playTest()
