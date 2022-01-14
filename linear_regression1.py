# Libraries
import matplotlib.pyplot as plt
import numpy as np

# Data
x_soat = np.array([1.0, 2.0, 3.0])
y_baho = np.array([2.0, 4.0, 6.0])


# Function for correctly calculating
def forward(x):
    return x * w


# (Loss) Function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# Containers for us to create a graph
w_list = []
mse_list = []

# Calculate w in the range 0 to 4
for w in np.arange(0.0, 4.1, 0.1):
    print("w={:.3f}".format(w))
    L_umum = 0

    for x_hb_qiym, y_hb_qiym in zip(x_soat, y_baho):
        y_hb_bash = forward(x_hb_qiym)
        L_hb_qiym = loss(x_hb_qiym, y_hb_qiym)
        L_umum += L_hb_qiym
        print("\t", "{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(x_hb_qiym, y_hb_qiym, y_hb_bash, L_hb_qiym))

    # Calculating MSE for each information
    print("MSE=", L_umum / len(x_soat))  # len(x_soat)--> N
    w_list.append(w)
    mse_list.append(L_umum / len(x_soat))

# Result ( Graph )
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
ax = plt.axes()
ax.set_facecolor('#030101')
plt.show()
