from dm_env import specs, Environment, TimeStep, StepType
from dm_control import suite
from dm_control.suite.wrappers import pixels, action_scale
from collections import deque
import numpy as np
import dm_env

class PixelWrapper(pixels.Wrapper):
    """扩展 pixels.Wrapper 获取 84x84 图像观测"""
    def __init__(self, env):
        super().__init__(
            env,
            pixels_only=True,  # 只返回像素观测
            render_kwargs={
                'width': 84,
                'height': 84,
                'camera_id': 0
            }
        )

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, repeat):
        self._env = env
        self._repeat = repeat

    def step(self, action):
        total_reward = 0
        for _ in range(self._repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward
            if time_step.last():
                break
        return time_step._replace(reward=total_reward)

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

class FrameStackWrapper(dm_env.Environment):
    """堆叠多帧,把像素从 (H, W, C) 转成 (C, H, W)"""
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()
        
        # 从 OrderedDict 中取出 pixels 的规格
        pixel_spec = wrapped_obs_spec['pixels']
        h, w, c = pixel_spec.shape  # 原始形状是 (H, W, C)
        
        # 设置新的观测规格:(C×num_frames, H, W)
        self._obs_spec = specs.BoundedArray(
            shape=(c * num_frames, h, w),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )

    def _transform_observation(self, time_step):
        """把队列里的多帧图像拼接成 (C×frames, H, W)"""
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)  # (9, 84, 84)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        """从 time_step 里取 'pixels' 并转成 (C, H, W)"""
        pixels = time_step.observation['pixels']  # (H, W, C)
        pixels = np.transpose(pixels, (2, 0, 1))  # => (C, H, W)
        return pixels

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        # action spec 也有名字
        action_spec = self._env.action_spec()
        if getattr(action_spec, 'name', None) is None:
            action_spec = specs.BoundedArray(
                shape=action_spec.shape,
                dtype=action_spec.dtype,
                minimum=action_spec.minimum,
                maximum=action_spec.maximum,
                name='action'
            )
        return action_spec

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step)

    def _augment_time_step(self, time_step):
        if time_step.first():
            return TimeStep(
                step_type=time_step.step_type,
                reward=0.0,
                discount=1.0,
                observation=time_step.observation
            )
        return TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=time_step.observation
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

class MaxStepWrapper(dm_env.Environment):
    def __init__(self, env, max_steps=1000):
        self._env = env
        self._max_steps = max_steps
        self._current_step = 0

    def reset(self):
        self._current_step = 0
        return self._env.reset()

    def step(self, action):
        if self._current_step >= self._max_steps:
            print(f"[MaxStepWrapper] step={self._current_step}, reached max_steps={self._max_steps}, returning LAST.")
            return TimeStep(
                step_type=StepType.LAST,
                reward=0.0,
                discount=1.0,
                observation=self._env.reset().observation
            )
        self._current_step += 1

        time_step = self._env.step(action)

        if time_step.last():
            print(f"[MaxStepWrapper] time_step.last() from env at step={self._current_step}, returning LAST.")
            return time_step

        if self._current_step >= self._max_steps:
            print(f"[MaxStepWrapper] step={self._current_step}, forcing LAST.")
            return TimeStep(
                step_type=StepType.LAST,
                reward=time_step.reward,
                discount=time_step.discount,
                observation=time_step.observation
            )

        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

def make(name, frame_stack=3, action_repeat=2, seed=1):
    domain, task = name.split('_', 1)
    
    env = suite.load(domain, task, task_kwargs={'random': seed})
    
    env = PixelWrapper(env)
    env = ActionRepeatWrapper(env, action_repeat) 
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0) 
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    env = MaxStepWrapper(env, max_steps=1000)
    
    return env 