import wrappers.loco_wrapper_pixels as dmc

def test_env():
    env = dmc.make('walker_walk')
    time_step = env.reset()
    obs = time_step.observation
    print(f"Observation shape: {obs.shape}")
    
    print(f"Observation type: {type(obs)}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation min-max: ({obs.min()}, {obs.max()})")
    
    action = env.action_spec().generate_value()
    time_step = env.step(action)
    obs = time_step.observation
    print(f"\nAfter one step:")
    print(f"Observation shape: {obs.shape}")
    
if __name__ == "__main__":
    test_env() 