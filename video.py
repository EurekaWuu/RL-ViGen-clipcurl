import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False
        self.warning_printed = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if not self.enabled:
            return
        
        try:
            frame = env.render(mode='rgb_array')
        except:
            try:
                frame = env.render()
            except:
                if not self.warning_printed:
                    print("Warning: Failed to render frame")
                    self.warning_printed = True
                return
                
        if frame is None:
            if not self.warning_printed:
                print("Warning: env.render() returned None")
                self.warning_printed = True
            return
            
        if not isinstance(frame, np.ndarray):
            print(f"Warning: frame is not numpy array, got {type(frame)}")
            return
            
        if frame.ndim < 2:
            print(f"Warning: frame has less than 2 dimensions, shape: {frame.shape}")
            return
            
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = frame.transpose(1, 2, 0)
            
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
            
        self.frames.append(frame)

    def save(self, file_name):
        if not self.enabled or not self.frames:
            return
        path = self.save_dir / file_name
        imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.warning_printed = False

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if not self.enabled:
            return
            
        try:
            frame = env.render(mode='rgb_array')
        except:
            try:
                frame = env.render()
            except:
                if not self.warning_printed:
                    print("Warning: Failed to render frame")
                    self.warning_printed = True
                return
                
        if frame is None:
            if not self.warning_printed:
                print("Warning: env.render() returned None")
                self.warning_printed = True
            return
            
        if not isinstance(frame, np.ndarray):
            print(f"Warning: frame is not numpy array, got {type(frame)}")
            return
            
        if frame.ndim < 2:
            print(f"Warning: frame has less than 2 dimensions, shape: {frame.shape}")
            return
            
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = frame.transpose(1, 2, 0)
            
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
            
        self.frames.append(frame)

    def save(self, file_name):
        if not self.enabled or not self.frames:
            return
        path = self.save_dir / file_name
        imageio.mimsave(str(path), self.frames, fps=self.fps)
