import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from tqdm import tqdm


def episode_len(episode):
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, action=None):
        for spec in self._data_specs:
            spec_name = getattr(spec, 'name', None)
            
            if spec_name == 'observation':
                value = time_step.observation
                if isinstance(value, dict):
                    # 如果是字典,取出 'pixels' 键对应的值
                    value = value['pixels']
                assert value.shape == (9, 84, 84), f"Wrong observation shape: {value.shape}"
                self._current_episode['observation'].append(value)
                
            elif spec_name == 'action':
                if action is None or (isinstance(action, np.ndarray) and action.shape == ()):
                    # 如果没有提供action或action形状不对,创建正确形状的全零动作
                    value = np.zeros(spec.shape, dtype=spec.dtype)
                else:
                    # 检查action是numpy数组且维度正确
                    value = np.asarray(action, dtype=spec.dtype)
                    if value.shape != spec.shape:
                        print(f"Warning: Reshaping action from {value.shape} to {spec.shape}")
                        value = value.reshape(spec.shape)
                self._current_episode['action'].append(value)
                
            elif spec_name == 'reward':
                # 检查奖励是 shape=(1,) 的数组
                value = np.array([time_step.reward], dtype=np.float32)
                self._current_episode['reward'].append(value)
                
            elif spec_name == 'discount':
                # 检查折扣因子是 shape=(1,) 的数组
                value = np.array([time_step.discount], dtype=np.float32)
                self._current_episode['discount'].append(value)
                
            else:
                raise ValueError(f'Unexpected spec name: {spec_name} (spec: {spec})')

        if time_step.last():
            # 如果是最后一步,存储整个 episode
            episode = dict()
            for spec in self._data_specs:
                spec_name = getattr(spec, 'name', None)
                vals_list = self._current_episode[spec_name]
                
                try:
                    episode[spec_name] = np.array(vals_list, dtype=spec.dtype)
                except Exception as e:
                    print(f"Error converting {spec_name} to numpy array: {str(e)}")
                    print(f"Values shapes: {[v.shape if hasattr(v, 'shape') else type(v) for v in vals_list]}")
                    raise
                    
            
            self._current_episode = defaultdict(list)
            
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        
        try:
            save_episode(episode, self._replay_dir / eps_fn)
            print(f"[REPLAY DEBUG] Saved an episode file: {eps_fn}")
        except Exception as e:
            print(f"Error saving episode: {str(e)}")
            
            self._num_episodes -= 1
            self._num_transitions -= eps_len
            raise


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        
        # 减少重复打印的布尔标记
        self._already_logged_no_episodes = False
        
        # 控制 "[REPLAY DEBUG]" 输出频率的计数器
        self._debug_check_counter = 0
        
        # 初始化时就加载一些数据
        self._try_fetch()

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0

        # 单 worker 时可以不用多 worker 判断
        worker_id = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            pass

        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        
        self._debug_check_counter += 1

        if (worker_id == 0
            and (self._debug_check_counter % 10 == 0)
            and (len(eps_fns) > len(self._episodes))):
            print(f"[REPLAY DEBUG] Found new files to load (worker={worker_id})...")
            from tqdm import tqdm
            for eps_fn in tqdm(eps_fns,
                               desc="Loading replay buffer",
                               mininterval=0.5,
                               dynamic_ncols=True,
                               leave=True,
                               position=0,
                               ncols=80):
                self._store_one_file(eps_fn, fetched_size=0)
        else:
            # 静默加载: 不使用进度条，直接处理
            for eps_fn in eps_fns:
                self._store_one_file(eps_fn, fetched_size=0)

    def _load_episodes_by_worker(self, eps_fns, worker_id, show_tqdm=False):
        """
        是否在 worker_id=0 并且 show_tqdm=True
        否则静默加载
        """
        fetched_size = 0

        if worker_id == 0 and show_tqdm:
            # 使用 tqdm
            for eps_fn in tqdm(eps_fns, 
                              desc="Loading replay buffer",
                              mininterval=0.5,        
                              dynamic_ncols=True,     
                              leave=True,            
                              position=0,
                              ncols=80,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]' 
                              ):
                self._store_one_file(eps_fn, fetched_size)

        else:
            # 静默加载
            for eps_fn in eps_fns:
                self._store_one_file(eps_fn, fetched_size)

    def _store_one_file(self, eps_fn, fetched_size):
        """
        处理并存储单个 episode 文件 eps_fn，检查维度并调用 self._store_episode
        """
        try:
            parts = eps_fn.stem.split('_')
            if len(parts) < 4:
                return
            eps_idx = int(parts[-2])
            eps_len = int(parts[-1])

            if eps_fn in self._episodes:
                return
            if fetched_size + eps_len > self._max_size:
                return
            fetched_size += eps_len

            if not self._store_episode(eps_fn):
                return
        except Exception as e:
            print(f"Error processing file {eps_fn}: {str(e)}")
            return

    def _sample(self):
        try:
            self._try_fetch()
        except Exception as e:
            print(f"Error in _try_fetch: {str(e)}")
            traceback.print_exc()
            
        self._samples_since_last_fetch += 1
        
        # 确保有数据可采样
        if not self._episode_fns:
            if not self._already_logged_no_episodes:
                print("No episodes available, returning empty sample")
                self._already_logged_no_episodes = True
            
            empty_sample = (
                np.zeros(self._obs_shape, dtype=np.uint8),
                np.zeros(self._action_dim, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.zeros(self._obs_shape, dtype=np.uint8)
            )
            return empty_sample
        else:
            # 每次一旦真的有数据可用了，就重置该标记
            self._already_logged_no_episodes = False

        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        
        # 获取数据
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        
        # 检查 action 是否有效
        if action is None or action.shape == ():
            print(f"Invalid action shape: {action.shape if action is not None else None}")
            # 返回一个有效形状的空动作
            action = np.zeros(self._action_dim, dtype=np.float32)
        
        # 检查维度
        assert obs.shape == (9, 84, 84), f"Wrong observation shape: {obs.shape}"
        assert len(action.shape) == 1, f"Wrong action shape: {action.shape}"
        assert next_obs.shape == (9, 84, 84), f"Wrong next_obs shape: {next_obs.shape}"
        
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        
        return (obs, action, reward, discount, next_obs)

    def _sample_episode(self):
        if not self._episode_fns:
            raise RuntimeError("No episodes available to sample from")
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            # early_eps_fn.unlink(missing_ok=True)

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            pass

        return True

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(storage, max_size, batch_size, num_workers,
                      save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    # 从 storage 的 data_specs 中获取维度信息
    obs_spec = storage._data_specs[0] 
    action_spec = storage._data_specs[1] 

    class EnhancedReplayBuffer(ReplayBuffer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._obs_shape = obs_spec.shape
            self._action_dim = action_spec.shape[0]

    iterable = EnhancedReplayBuffer(storage._replay_dir,
                                  max_size_per_worker,
                                  num_workers,
                                  nstep,
                                  discount,
                                  fetch_every=1000,
                                  save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       worker_init_fn=_worker_init_fn)
    return loader 