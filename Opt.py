class Opt():
    n_epochs = 500
    batch_size = 100
    # lr = 0.0004
    lr = 0.0001
    b1 = 0.5
    b2 = 0.999
    n_cpu = 4
    n_classes = 10
    img_size = 28
    img_channels = 1
    z_size = 64
    sample_interval = 200
    lambda_ALM = 1
    mu_ALM = 1.2
    rho_ALM = 1.5
    decay_lr = 0.75
    lambda_mse = 1e-3
    LAMBDA=float(10)
    sigma=float(1)
    # def __init__(self):
    #     print("opt setted")