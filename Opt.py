class opt():
    n_epochs = 200
    batch_size = 100
    lr = 3e-4
    b1 = 0.5
    b2 = 0.999
    n_cpu = 4
    n_classes = 10
    img_size = 28
    img_channels = 1
    z_size = 64
    sample_interval = 500
    lambda_ALM = 1
    mu_ALM = 1.2
    rho_ALM = 1.5
    decay_lr = 0.75
    lambda_mse = 1e-3
    def __init__(self):
        print("opt setted")