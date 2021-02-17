import numpy as np
from torch import device, cuda
import math

from memory import MemoryReplay
from policy import atariDQN
from actors import ActorDQN, LearnerDQN
from utils import Tocker, goToDir, Saver, timeFormated, time

from tqdm import tqdm
import os

from torch.utils.tensorboard import SummaryWriter

try:
    import ray
    RAY = True
except:
    RAY = False
    print("There's no ray package available to import!")

NCPUS = os.cpu_count()
CUDA = cuda.is_available()
DEVICE = device('cuda') if CUDA else device('cpu')
MEGA = 1024 * 1024
GIGA = MEGA * 1024

class Trainer:
    def __init__(self, args, n_workers = 4, memory_ram = 2, time_saving=45):
        # Getting Ray
        self.Ray = args.ray
        if self.Ray and not RAY:
            print("-- ray is not available --\nNormal initialization in progress")
            self.Ray = False
        
        # Some parameters
        self.time_save = time_saving
        self.seed = args.seed
        self.exp_steps = args.steps
        self.test_steps = args.test_steps
        self.test_freq = args.test_freq
        self.memory_start = args.memory_start
        self.update_online = args.update_online
        self.play_steps = args.play
        self.game = args.game
        self.double = args.double
        self.game_actions = args.game_actions
        self.seed = args.seed

        # Other variables to save progress
        self.time = Tocker()
        self.ckTime = Tocker()
        self.acc_test = 0
        self.mean_test = []
        self.std_test = []
        self.actor_test_episodes = 0
        self.test_episodes = 0
        name_temp = args.game
        name_temp += '_double' if self.double else ''
        name_temp += '_' + args.optimizer +'_lr_' + str(args.learning_rate)
        self.saver = Saver(name_temp)
        name_sum = os.path.join(self.saver.dir,"tensorboard_{}".format(name_temp))
        self.writer = SummaryWriter(name_sum)

        #Generating the actors
        self.policy = atariDQN(args.lhist, args.game_actions, args.dropouts)

        self.memoryReplay = MemoryReplay(capacity=args.memory_size,
                                        LHist=args.lhist, 
                                        )

        self.main_actor = ActorDQN(self.game,
                                    args.game_actions,
                                    self.policy,
                                    lHist=args.lhist,
                                    steps_per_update=args.update_online,
                                    buffer_size=args.buffer_size,
                                    test_steps = self.test_steps,
                                    start_steps = self.memory_start,
                                    device=DEVICE,
                                    seed = args.seed,
                                    writer=self.writer,
                                    )

        self.steps_to_fill_mem = math.ceil(self.memory_start / (args.buffer_size * args.lhist))

        self.n_actors = NCPUS if n_workers >= NCPUS else n_workers
            
        if NCPUS > 1 and self.Ray:
            # Actors with ray are created to speed up
            # the filling and testing of the buffer and net
            actors_start_steps = math.ceil(self.memory_start / self.n_actors)
            self.steps_to_fill_mem = math.ceil(actors_start_steps / (args.buffer_size * args.lhist))
            actors_test_steps = math.ceil(self.test_steps / self.n_actors)

            # ---- Initialize Ray ----
            ray.init(num_cpus=self.n_actors,
                    _memory = memory_ram*GIGA, 
                    object_store_memory = 400*MEGA)
            
            # Buffers for the actors
            #self.buffers = [Buffer(capacity=actors_buffer_size) for _ in range(self.n_actors)]
            # Actors of the ActorDQN to fill and evaluate-only Access to their buffers only
            actor = ray.remote(ActorDQN)
            self.actors = [actor.remote(self.game,
                                        args.game_actions,
                                        self.policy,
                                        lHist=args.lhist,
                                        buffer_size=args.buffer_size,
                                        test_steps = actors_test_steps,
                                        start_steps = actors_start_steps,
                                        seed = args.seed,
                                        ray_actor = True) \
                                                for i in range(self.n_actors)]
            time.sleep(10)
            print("Trainer set with Ray\nRay resources {} workers with {} GB of RAM".format(self.n_actors, memory_ram))
        else:
            print(timeFormated(), "Trainer set")

        self.main_learner = LearnerDQN(self.policy,
                                        self.memoryReplay,
                                        args.mini_batch_size,
                                        learning_rate=args.learning_rate,
                                        update_target=args.update_target,
                                        device=DEVICE,
                                        double=args.double,
                                        optimizer=args.optimizer,
                                        seed=args.seed,
                                        writer= self.writer,
                                        )

    def fillMemory(self):
        I = tqdm(range(self.steps_to_fill_mem), desc='Filling Memory')
        self.time.tick
        if self.Ray:
            print("Ray has started. Filling Memory . . .")
            for i in I:
                buffers = ray.get([actor.fillBuffer.remote() for actor in self.actors])
                self.memoryReplay.combineBuffers(*buffers)
            self.main_actor.steps = self.memory_start
        else:
            for i in I:
                buffer = self.main_actor.fillBuffer()
                self.memoryReplay.combineBuffers(buffer)
        print(timeFormated(), "Memory Filled to {} in {}".format(self.memory_start,self.time.tock))
        # Saving a fixed ammount of frames to Test on.
        s, a, r, s2, t = self.memoryReplay.Sample(500)
        self.main_actor.passTestHistories(s)
        del a, r, s2, t

    def sampleBuffer(self, samples:int = 20):
        print("Displaying {} sample histories from the buffer".format(samples))
        try:
            self.memoryReplay.showBuffer(samples)
        except:
            print("Display samples Stopped")

    def __del__(self):
        print("Trainer Terminated")

    def close(self):
        if self.ckTime.tocktock > 5 : self.saveAll()
        self.writer.flush()
        self.writer.close()
        ray.shutdown()

    def train(self):
        self.main_actor.newGame()
        I = tqdm(range(0, self.exp_steps), 
                        desc='Executing and learning', unit='updates',
                        )
        for i in I:
            bufferReady = self.main_actor.autoStep()
            if bufferReady:
                self.memoryReplay.combineBuffers(self.main_actor.getBuffer())
            self.main_learner.trainStep()
            #self.main_actor.updateModel(self.main_learner.onlineModel()) # They got the same object now
            self.writer.flush()
            if i % self.test_freq == 0:
                self.test()
            if self.ckTime.tocktock >= self.time_save:
                self.saveAll()
                self.ckTime.tick

    def test(self):
        #self.memoryReplay.add(*self.memoryReplay.zeroe)
        # ---- Testing the perfomance ----
        self.time.tick
        q_mean = self.main_actor.testQHistories()
        if self.Ray:
            print("Ray starting test . . . ")
            # ---- Dividing the total test_steps per actor ----
            # Update the online model in the actors
            updatedOnline = self.main_learner.onlineModel(cpu=True)
            ray.get([actor.updateModel.remote(updatedOnline.copy()) for actor in self.actors])
            # Get test results
            testRes = ray.get([actor.testRun.remote() for actor in self.actors])
            # Consolidate results
            episodeRwd = []
            for rE in testRes:
                episodeRwd += rE
            print("Ray testing done")
        else:
            # ------ main actor perfoms all the steps sequentially ------
            episodeRwd = self.main_actor.testRun()
        # Saving results and logging
        len_episodeRwd = len(episodeRwd)
        tot_episodeRwd = sum(episodeRwd)
        self.actor_test_episodes += len(episodeRwd)
        self.mean_test += [np.mean(episodeRwd if len_episodeRwd != 0 else 0.0)]
        self.std_test += [np.std(episodeRwd if len_episodeRwd != 0 else 0.0)]
        self.acc_test += tot_episodeRwd
        self.writer.add_scalar('test/accumulated_reward', tot_episodeRwd, self.test_episodes)
        self.writer.add_scalar('test/mean_reward', self.mean_test[-1], self.test_episodes)
        self.writer.add_scalar('test/std_reward', self.std_test[-1], self.test_episodes)
        self.writer.add_scalar('test/actor_episodes', len_episodeRwd, self.test_episodes)
        self.writer.add_scalar('test/Q-mean', q_mean, self.test_episodes)
        self.writer.flush()
        self.test_episodes += 1
        print(timeFormated(), "Test done in {}. Reward Accumulated:{}, Mean:{}. Q_mean {}".format(self.time.tock,
																						np.round(tot_episodeRwd, 2),
																						np.round(self.mean_test[-1],2),
                                                                                        np.round(q_mean,3)))

    def saveAll(self):
        # --- Saving buffer and trainer ----
        models = dict()
        self.saver.saveObject(self.dictToSave(), "trainer")
        self.saver.saveObject(self.memoryReplay.dictToSave(), "memory")
        models['online'] = self.main_learner.onlineModel()
        models['target'] = self.main_learner.targetModel()
        models['optimizer'] = self.main_learner.optimizerState()
        self.saver.saveModel(models, "models")
    
    def loadActor(self, Dir):
        os.chdir(Dir)
        files = os.listdir()
        print("Files on direction:")
        for n, File in enumerate(files):
            print("{} : {}".format(n, File))
        while 1:
            choice = input("Enter the number for the model to load :")
            choice = int(choice)
            if choice > len(files) or not isinstance(choice, int) or choice < 0:
                print("Number not valid. Please try again.")
            else:
                break
        model = os.path.join(Dir, files[choice])
        models = self.saver.loadModel(model, DEVICE)
        self.main_actor.updateModel(models['online'])
        print("Actor restored from", Dir)

    def dictToSave(self):
        this = dict()
        res = dict()
        res['means'] = self.mean_test
        res['stds'] = self.std_test
        res['acc'] = self.acc_test
        res['episodes'] = self.actor_test_episodes
        this['results'] = res
        this['seed'] = self.seed
        this['Steps'] = self.exp_steps
        this['testSteps'] = self.test_steps
        this['memoryStart'] = self.memory_start
        this['double'] = self.double
        this['game'] = self.game
        this['game_actions'] = self.game_actions
        return this

    def loadFromDict(self, this):
        try:
            res = this['results']
            self.mean_test = res['means']
            self.sdt_test = res['stds']
            self.acc_test = res['acc']
            self.actor_test_episodes = res['episodes'] 
            self.seed = this['seed'] 
            self.exp_steps = this['Steps']
            self.test_steps = this['testSteps']
            self.memory_start = this['memoryStart']
            self.double = this['double']
            self.game_actions = this['game_actions']
            print("Successfully loading Trainer from dict")
        except:
            print("Error loading Trainer loaded from dict")

    def playTest(self):
        try:
            import imageio
            GIF = True
            bufferFrame = []
        except:
            GIF = False
            print("imageio is missing from the packages. A .gif from the run won't be made.")
        if self.play_steps > 0:
            # Start playing sequence
            # --- wait user to watch ----
            a = input("Press any key to initialize test . . .")
            self.main_actor.isTest = True
            self.main_actor.updateModel(self.main_learner.onlineModel())
            self.main_actor.newGame()
            env = self.main_actor.env
            game = self.main_actor.game
            print("Test of the agent in {}".format(game))
            episodes, reward = 0, 0
            I = tqdm(range(0, self.play_steps), 'Test in progress', unit=' plays')
            for _ in I:
                self.time.tick
                if GIF:
                    bufferFrame.append(env._get_image())
                env.render()
                stepRwd = self.main_actor.step()
                #60Hz / Skip_Frame as the environment will do
                self.time.lockHz(15 if game != 'space_invaders' else 20)
                if self.main_actor.done:
                    episodes += 1
                reward += stepRwd
            env.close()
            if GIF:
                imageio.mimsave("./testPlay {} frames {} episodes {} points {} - {}.gif".format(game, self.play_steps, episodes, reward, timeFormated()),
                                    bufferFrame, fps = 15 if game != 'space_invaders' else 20)
            print(timeFormated(), "Test play done. Completed {} episodes and accumulated {} points".format(episodes, reward))
        else:
            None
    
    
