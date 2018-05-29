class Kernel(object):
    def __init__(self, kernel, radius=2, sigma_x=1.0, sigma_y=1.0, theta=0):
        self.kernel = kernel
        self.radius = radius
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.theta = theta

    def __repr__(self):
        return 'Kernel <%s %s %s %s>' % (self.kernel, self.sigma_x, self.sigma_y, self.theta)
