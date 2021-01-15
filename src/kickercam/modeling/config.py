"""
Simple Standard config 

"""

config =  { "input_size":(1, 256, 144), "use_gray": True, "use_rgb": False, "norm_mean": 0, "norm_scale": 255.,
        "interverted_residual_setting" : [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 32, 4, 2],
            [6, 16, 4, 2],
        ]}
