import cosmogp
import numpy as np

number_of_point = 40
kernel_amplitude = 0.5
correlation_length = 0.5

x_axis = np.linspace(0,5,number_of_point)
x_axis, y_axis = np.meshgrid(x_axis,x_axis)

X = np.array([x_axis.reshape(number_of_point**2) ,
              y_axis.reshape(number_of_point**2)]).T


kernel = cosmogp.rbf_kernel_2d(X, [kernel_amplitude,correlation_length,correlation_length,0.],
                               nugget=0, floor=0.0, y_err=None)

y = np.random.multivariate_normal(np.zeros(len(X)), kernel)
