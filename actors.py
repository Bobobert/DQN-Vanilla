from gym.envs.atari import AtariEnv
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from memory import MemoryReplay
from CONST import *
from tqdm import tqdm

makeEnv = lambda game: AtariEnv(game,
                                obs_type= 'image',
                                frameskip= 4 if game != 'space_invaders' else 3)

# TYPES
FRAMETYPE = np.uint8

class ActorDQN():
    """
    Actor to host an excecute a gym environment for atari games.
    
    Paramenters
    -----------
    game: str
        Name of the game to execute.
    gameActions: int
        Number of actions that the agent can execute.
        If doubt check the actions in CONST.ACTION_MEANING
    policy: torch policy
        Object that hosts the policy network to process the
        environment's observations.
    
    """
    def __init__(self,
                game:str,
                gameActions:int,
                policy,
                lHist: int = 4,
                steps_per_update:int = 4,
                test_steps:int = TEST_STEPS,
                epsilon:float = EPSILON_INIT,
                epsilon_final:float = EPSILON_FINAL,
                epsilon_test:float = EPSILON_TEST,
                epsilon_life:int = EPSILON_LIFE,
                epsilon_mode:str = "linear",
                buffer_size:int = BUFFER_SIZE,
                start_steps: int = REPLAY_START,
                start_noop_max:int = NO_OP_MAX,
                clip:float = CLIP_REWARD,
                crop = None,
                seed:int = -1,
                ray_actor:bool = False,
                device = torch.device('cpu'),
                writer = None,
                ):
        
        self.seed = seed if seed >= 0 else None
        self.rg = np.random.Generator(np.random.PCG64(self.seed))

        buffer_size = buffer_size * lHist
        self.sBuffer = np.zeros([buffer_size] + FRAMESIZE, dtype=FRAMETYPE)
        self.aBuffer = np.zeros(buffer_size, dtype=FRAMETYPE)
        self.rBuffer = np.zeros(buffer_size, dtype=np.float32)
        self.tBuffer = np.zeros(buffer_size, dtype=np.bool_)
        self._i = 0
        self.bufferFull = False
        self.buffer_size = buffer_size
        self.crop = crop
        self.lHist = lHist
        self.lastObs = None
        
        assert game in Games or game in GamesExtras, "Game name is not valid."
        self.game = game
        self.n_actions = gameActions
        self.actionsT = torch.tensor(range(gameActions), 
                                        dtype=torch.int64,
                                        device=device).unsqueeze(0)
        self.lives = None
        self.setGame(game)
        self.done = True

        self.ray_actor = ray_actor
        self.device = device
        self.policy = policy.copy().to(device) if ray_actor else policy # If main do not copy the net.
        self.writer = writer

        self._epsilon_start = epsilon
        self._epsilon_end = epsilon_final
        self._epsilon_life = epsilon_life
        self._epsilon_mode = epsilon_mode
        self._epsilon_test = epsilon_test
        self.noop_max = start_noop_max
        self.clip = clip

        self.steps = 0
        self.episodes = 0
        self.steps_per_update = steps_per_update
        self.steps_per_test = test_steps
        self.testSteps = 0
        self.testEpisodes = 0
        self.isTest = False
        self.test_histories_steps = 0
        self.start_steps = start_steps
        self.test_histories = None

    def setGame(self, game):
        self.env = makeEnv(game)
        print("Game {} is set up!".format(game))
        if self.env.action_space.n != self.n_actions:
            print("Actions given {} don't match the environment {}".format(
                            self.n_actions ,self.env.action_space.n))
        if self.seed is not None: 
            self.env.seed(self.seed)
        
    def newGame(self):
        obs = self.env.reset()
        done = False
        if self.isTest:
            for _ in range(self.rg.integers(0, self.noop_max)):
                # Actions with no-op at the start for a random number of steps
                obs, _, done, _ = self.env.step(0)
        self.updateBuffer(obs, 0, 0, done)
        self.lives = self.env.ale.lives()
        self.done = done
    
    def procFrame(self,f):
        if self.crop is not None:
            # Do crop. Self.crop can be a list or tuple
            f = f[self.crop[0]:self.crop[1]]
        return  cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2YUV)[:,:,0], FRAMESIZETP)
    
    def updateBuffer(self, st1, at, rt, terminal):
        """
        Saves the information of the transition from a step of
        the environment, this is s_{t+1}, a_t, r_t, terminal_{t+1}.
        Meaning it saves the new observation from P(s_t, a_t), s_t
        should be the previous item in sBuffer.

        The tBuffer uses not-logic. True when not-terminal and 
        False for terminal states.
        """
        st1 = self.procFrame(st1)
        if self.clip > 0.0:
            rt = np.clip(rt, -self.clip, self.clip)
        self.sBuffer[self._i] = st1
        self.aBuffer[self._i] = at
        self.rBuffer[self._i] = rt
        self.tBuffer[self._i] = not terminal
        self._i = (self._i + 1) % self.buffer_size
        if self._i == 0:
            self.bufferFull = True

    def getTransition(self):
        i = self._i - 1
        transition = (self.sBuffer[i].copy(),
                        self.aBuffer[i].copy(),
                        self.rBuffer[i].copy(),
                        self.tBuffer[i].copy())
        return transition

    def clearBuffer(self):
        self.bufferFull = False
        self._i = 0
        self.tBuffer[:] = False
        self.done = True
        self.lastObs = None

    def getState(self):
        """
        Returns from buffer the last composite history by Phi
        on the time t.
        """
        lastObs = torch.zeros((self.lHist, FRAMESIZE[0], FRAMESIZE[1]),
                                    dtype=torch.float32)
        for i in range(self.lHist):
            j = self._i - i - 1
            if i > 0 and not self.tBuffer[j]:
                break
            lastObs[i] = torch.from_numpy(self.sBuffer[j])
        return lastObs.unsqueeze(0).to(self.device).div(255)

    def updateModel(self, state_dict):
        self.policy.updateState(state_dict)

    @property
    def Steps(self) -> int:
        return max(0, self.steps - self.start_steps)

    @property
    def epsilon(self) -> float:
        """
        Returns the current value of epsilon given the configurations.
        """
        if self.isTest:
            return self._epsilon_test
        steps = self.Steps
        if steps > self._epsilon_life:
            return self._epsilon_end
        if self._epsilon_mode == 'linear':
            m = (self._epsilon_end - self._epsilon_start) / (self._epsilon_life)
            return self._epsilon_start + m * steps
        elif self._epsilon_mode == 'annealed':
            return self._epsilon_end + \
                (self._epsilon_start - self._epsilon_end) * np.exp(-1.0 * steps / self._epsilon_life)

    def step(self, dry=False):
        """
        Main method to do a step on the environment.
        Resets or change environment automatically when
        then current environment is done.
        Adds the current obsevations to the buffer and 
        clips the reward if needed.

        If dry is True always executes a random action.
        Returns
        -------
        reward:float
        """
        if self.done:
            self.newGame()
        if dry:
            action = self.randomAction()
        else:
            action =  self.calculateAction()
        obs, reward, done, info = self.env.step(action)

        obs_lives = info['ale.lives']

        # Reboots in training when it lost a life
        # It does not apply when dry mode or test
        if obs_lives == self.lives or dry or self.isTest:
            self.updateBuffer(obs, action, reward, done)
        else:
            self.updateBuffer(obs, action, reward, True)

        self.lives = obs_lives
        self.done = done

        # Updating counters
        if self.isTest:
            self.testSteps += 1
            self.testEpisodes += 1 if self.done else 0
        else:
            self.steps += 1
            self.episodes += 1 if self.done else 0
        return reward

    def autoStep(self):
        """
        Does n steps_per_update in the environment. Accumulates the experiences
        in its buffer. Returns the flag when the buffer is full
        """ 
        for i in range(self.steps_per_update):
            _ = self.step()
        return self.bufferFull

    def calculateAction(self):
        if self.rg.uniform() <= self.epsilon:
            # Epsilon step
            return self.randomAction()
        else:
            # Greedy step
            lastObs = self.getState()
            with torch.no_grad():
                output = self.policy.forward(lastObs)
                q_max = output.max(1).values.item()
                actions = self.actionsT[output.ge(q_max)]
            del lastObs, output
            return self.rg.choice(actions.to('cpu'))

    def randomAction(self):
        return self.rg.integers(self.n_actions)

    def testRun(self):
        """
        Executes a whole test run with the actual policy state.

        returns
        -------
        rewards_episode: list
            Contains all the accumulate reward by episode.
        """
        self.isTest = True
        self.clearBuffer()
        # Making the iterator
        if self.ray_actor:
            I = range(self.steps_per_test)
        else:
            I = tqdm(range(self.steps_per_test),desc="Testing agent",unit='steps')
        episode_reward, ls = 0, 0
        rewards_episode = []
        for step in I:
            episode_reward += self.step()
            if self.done:
                rewards_episode += [episode_reward]
                # Logging Actor
                if self.writer is not None:
                    self.writer.add_scalar("actor/Episode reward", episode_reward, self.testEpisodes)
                    self.writer.add_scalar("actor/Steps per episode", step - ls, self.testEpisodes)
                    ls = step
                episode_reward = 0
        if not self.ray_actor:
            I.close()
        self.isTest = False
        self.clearBuffer()
        if rewards_episode == []:
            rewards_episode = [episode_reward]
        return rewards_episode
    
    def testQHistories(self):
        if self.test_histories is None:
            return 0.0
        with torch.no_grad():
            model_out = self.policy(self.test_histories)
            mean = torch.mean(model_out.max(1).values).item()
        self.test_histories_steps += 1
        del model_out
        return mean

    def passTestHistories(self, histories):
        self.test_histories = histories.to(self.device)

    def getBuffer(self):
        Buffers = (self.sBuffer.copy(), self.aBuffer.copy(), 
                    self.rBuffer.copy(), self.tBuffer.copy())
        self.bufferFull = False
        return Buffers

    def fillBuffer(self):
        self.clearBuffer()
        for _ in range(self.buffer_size):
            self.step(dry = True)
        return self.getBuffer()

class LearnerDQN():
    """
    Actor that manages the samples and learn the transitions
    with the policy passed.

    Parameters
    ----------
    """
    def __init__(self,
                policy,
                memory:MemoryReplay,
                mini_batch_size:int = MINI_BATCH_SIZE,
                gamma:float = GAMMA,
                update_target:int = UPDATE_TARGET,
                learning_rate:float = LEARNING_RATE,
                momentum:float = G_MOMENTUM,
                clip_error:float = CLIP_ERROR,
                double: bool = True,
                optimizer:str = 'rmsprop',
                seed:int = -1,
                device = torch.device('cpu'),
                writer = None,
                ):

        self.seed = seed if seed >= 0 else None
        self.rg = np.random.Generator(np.random.PCG64(self.seed))

        self.double = double
        self.gamma = gamma
        self.clip = clip_error
        self.miniBatch_size = mini_batch_size

        self.device = device
        self.target_net = policy.copy().to(device)
        self.online_net = policy.to(device)
        self.n_actions = policy.outputs
        
        self.memory = memory
        self.writer = writer
        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.online_net.parameters(), 
                                        lr= learning_rate, alpha=momentum,
                                        eps=0.00001, centered=True)

        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.online_net.parameters(),
                                        lr=learning_rate,)

        self.steps = 0
        self.update_target = update_target

    def Dropout(self,activate:bool=True,mod:int=2):
        """
        Equates to call the model.train() or model.eval()
        depending of activate.
        mod = 0, just set the online_net
        mod = 1, just set the target_net
        mod = 2, set both
        """
        if mod == 0:
            self.online_net.Dropout(activate)
        elif mod == 1:
            self.target_net.Dropout(activate)
        elif mod == 2:
            self.online_net.Dropout(activate)
            self.target_net.Dropout(activate)

    def trainStep(self):
        """
        Executes a simple learning step with a mini batch from
        the memory replay.\n
        It updates its target network automagically.
        """
        s1, a, r, s2, t = self.memory.Sample(self.miniBatch_size, self.device)
        actions_hot = F.one_hot(a, num_classes=self.n_actions)
        q_online = self.online_net.forward(s1)
        q_online = torch.mul(q_online, actions_hot)
        q_target = self.dqnTarget(s2, r, t)
        q_target = torch.mul(q_target, actions_hot)

        # Hubber Loss
        loss = F.smooth_l1_loss(q_online, q_target, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar('train/Loss', loss.item(), self.steps)
            self.writer.add_scalar('train/Mean TD Error', torch.mean(q_target - q_online).item(), self.steps)
        if self.steps % self.update_target == 0:
            # Updates the net
            self.updateTargetModel(self.onlineModel())
        self.steps += 1

    def dqnTarget(self, s2, r, t):
        with torch.no_grad():
            if self.double:
                On_model_out = self.online_net.forward(s2)
                a_greedy = On_model_out.max(1)[1]
            model_out = self.target_net.forward(s2)
            if not self.double:
                Qs2_max = model_out.max(1)[0]  
            else:
                Qs2_max = model_out.gather(1, a_greedy.unsqueeze(1)).squeeze(1)
            target = r + torch.mul(t, Qs2_max).mul(self.gamma).reshape(r.shape)
        return target.unsqueeze(1)

    def onlineModel(self, cpu:bool = False):
        return self.online_net.getState(cpu)

    def updateOnlineModel(self, new_state):
        self.online_net.updateState(new_state)

    def targetModel(self):
        return self.target_net.getState()

    def updateTargetModel(self, new_state):
        self.target_net.updateState(new_state)

    def optimizerState(self):
        return self.optimizer.state_dict()

    def optimizerUpdate(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
